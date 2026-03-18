"""Undo plugin — /undo as a self-contained, loadable nanobot plugin module.

This plugin implements the full undo feature (last-turn revert with optional
file-change rollback) for both the CLI and Telegram channels.  It is
completely self-contained: it works with any nanobot build that has the
plugin hook infrastructure (``register_command_handler`` etc.) but does **not**
require the built-in undo methods to exist on ``AgentLoop`` or
``TelegramChannel``.

Usage — add to ``~/.nanobot/config.json``:

.. code-block:: json

    {
      "agents": {
        "defaults": {
          "plugins": ["nanobot.plugins.undo.UndoPlugin"]
        }
      }
    }

When the plugin is registered, it takes priority over the built-in /undo
handling so it is safe to enable it alongside a fork that still contains the
original built-in implementation.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.plugins.base import NanobotPlugin

if TYPE_CHECKING:
    from nanobot.agent.loop import AgentLoop
    from nanobot.channels.telegram import TelegramChannel

# Tools whose side effects are fully reversible by restoring file content.
_REVERSIBLE_TOOLS: frozenset[str] = frozenset({"write_file", "edit_file", "read_file", "list_dir"})


class UndoPlugin(NanobotPlugin):
    """Plugin that implements /undo (undo the last conversation turn with file revert)."""

    name = "undo"

    def __init__(self) -> None:
        self._loop: AgentLoop | None = None
        self._channel: TelegramChannel | None = None
        # Per-session turn stack: skey → list of {"user_msg_id": int|None, "bot_msg_ids": list[int]}
        self._session_turn_stack: dict[str, list[dict]] = {}
        # Pending undo confirmations waiting for the user to click Confirm/Cancel.
        self._pending_undo: dict[str, dict] = {}

    # -----------------------------------------------------------------------
    # Plugin lifecycle
    # -----------------------------------------------------------------------

    def setup_agent(self, loop: AgentLoop) -> None:
        """Register /undo command handlers with the AgentLoop."""
        self._loop = loop

        # Phase 1 (Telegram two-step): return the plan without executing.
        loop.register_command_handler(
            lambda msg: (
                msg.content.strip().lower() == "/undo"
                and bool((msg.metadata or {}).get("_undo_preview"))
            ),
            self._handle_undo_preview,
        )
        # Phase 2 (Telegram two-step): apply after user confirmed.
        loop.register_command_handler(
            lambda msg: (
                msg.content.strip().lower() == "/undo"
                and (msg.metadata or {}).get("_undo_apply_turn_start") is not None
            ),
            self._handle_undo_apply,
        )
        # Simple /undo (CLI or any other channel without two-step UI).
        loop.register_command_handler(
            lambda msg: msg.content.strip().lower() == "/undo",
            self._handle_undo,
        )

    def setup_telegram(self, channel: TelegramChannel, app) -> None:
        """Register Telegram command/callback handlers and outbound hooks."""
        from telegram.ext import CallbackQueryHandler, CommandHandler

        self._channel = channel

        # Telegram command and callback handlers (registered before built-ins so they win).
        app.add_handler(CommandHandler("undo", self._on_undo_command))
        app.add_handler(CallbackQueryHandler(
            self._on_undo_callback, pattern="^undo_confirm:|^undo_cancel:"
        ))

        # Handle special outbound messages produced by the agent-side handlers.
        channel.register_outbound_handler(
            lambda msg: "_undo_plan" in (msg.metadata or {}),
            self._show_undo_confirmation,
        )
        channel.register_outbound_handler(
            lambda msg: "_undo_result" in (msg.metadata or {}),
            self._handle_undo_result,
        )

        # Track user messages and sent bot messages for deletion on successful undo.
        channel.register_inbound_hook(self._on_user_message)
        channel.register_sent_hook(self._on_bot_message_sent)

    # -----------------------------------------------------------------------
    # Agent-side helpers
    # -----------------------------------------------------------------------

    def _plan_undo(self, session_key: str) -> dict:
        """Compute the undo plan for the last turn without applying it."""
        assert self._loop is not None
        session = self._loop.sessions.get_or_create(session_key)
        turn_start = session.get_last_turn_start_index()
        if turn_start == -1:
            return {"nothing": True}

        last_turn_entries = [
            e for e in session.undo_log if e.get("turn_start_index") == turn_start
        ]
        last_turn_msgs = session.messages[turn_start:]

        non_reversible: set[str] = set()
        for m in last_turn_msgs:
            for tc in m.get("tool_calls") or []:
                tool_name = tc.get("function", {}).get("name", "")
                if tool_name and tool_name not in _REVERSIBLE_TOOLS:
                    non_reversible.add(tool_name)

        reversible_actions: list[str] = []
        for entry in last_turn_entries:
            if entry.get("reversible", True):
                reversible_actions.append(f"{entry['tool_name']} {entry['path']}")

        user_count = sum(1 for m in last_turn_msgs if m.get("role") == "user")
        assistant_count = sum(1 for m in last_turn_msgs if m.get("role") == "assistant")

        return {
            "nothing": False,
            "turn_start_index": turn_start,
            "user_count": user_count,
            "assistant_count": assistant_count,
            "reversible_actions": reversible_actions,
            "non_reversible": sorted(non_reversible),
        }

    # -----------------------------------------------------------------------
    # Agent-side command handlers
    # -----------------------------------------------------------------------

    async def _handle_undo_preview(self, msg: InboundMessage) -> None:
        """Return the undo plan without executing (used for Telegram confirmation UI)."""
        assert self._loop is not None
        plan = self._plan_undo(msg.session_key)
        await self._loop.bus.publish_outbound(OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id,
            content="",
            metadata={**dict(msg.metadata or {}), "_undo_plan": plan},
        ))

    async def _handle_undo_apply(self, msg: InboundMessage) -> None:
        """Apply the undo for a specific planned turn (second phase of Telegram two-step flow).

        Verifies that the session's current last turn still matches the requested
        ``turn_start_index`` — if a new message arrived between preview and confirm the
        index will have shifted, and we return an ``expired`` result without touching
        session state.
        """
        assert self._loop is not None
        requested_turn: int = msg.metadata["_undo_apply_turn_start"]

        # Cancel any in-progress tasks for this session first.
        await self._loop.cancel_session_tasks(msg.session_key)

        session = self._loop.sessions.get_or_create(msg.session_key)

        async def _publish_result(result: dict) -> None:
            await self._loop.bus.publish_outbound(OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id,
                content="",
                metadata={**dict(msg.metadata or {}), "_undo_result": result},
            ))

        current_turn_start = session.get_last_turn_start_index()
        if current_turn_start == -1:
            await _publish_result({"status": "nothing", "turn_start_index": requested_turn})
            return
        if current_turn_start != requested_turn:
            await _publish_result({"status": "expired", "turn_start_index": requested_turn})
            return

        turn_start = current_turn_start
        last_turn_entries = [
            e for e in session.undo_log if e.get("turn_start_index") == turn_start
        ]
        last_turn_msgs = session.messages[turn_start:]

        non_reversible: set[str] = set()
        for m in last_turn_msgs:
            for tc in m.get("tool_calls") or []:
                tool_name = tc.get("function", {}).get("name", "")
                if tool_name and tool_name not in _REVERSIBLE_TOOLS:
                    non_reversible.add(tool_name)

        reverted: list[str] = []
        revert_errors: list[str] = []
        for entry in reversed(last_turn_entries):
            if not entry.get("reversible", True):
                revert_errors.append(
                    f"{entry['tool_name']} {entry['path']}: prior content unavailable, not reverted"
                )
                continue
            try:
                path = Path(entry["path"])
                if entry.get("existed_before"):
                    prev = entry["previous_content"]
                    if not isinstance(prev, str):
                        revert_errors.append(
                            f"{entry['tool_name']} {entry['path']}: unexpected previous_content type, not reverted"
                        )
                        continue
                    path.write_text(prev, encoding="utf-8")
                else:
                    path.unlink(missing_ok=True)
                reverted.append(f"{entry['tool_name']} {entry['path']}")
            except Exception as exc:
                revert_errors.append(f"{entry['tool_name']} {entry['path']}: {exc}")

        session.undo_log = [e for e in session.undo_log if e.get("turn_start_index") != turn_start]
        user_count = sum(1 for m in last_turn_msgs if m.get("role") == "user")
        assistant_count = sum(1 for m in last_turn_msgs if m.get("role") == "assistant")
        session.drop_last_turn()
        self._loop.sessions.save(session)

        await _publish_result({
            "status": "success",
            "turn_start_index": turn_start,
            "user_count": user_count,
            "assistant_count": assistant_count,
            "reverted": reverted,
            "revert_errors": revert_errors,
            "non_reversible": sorted(non_reversible),
        })

    async def _handle_undo(self, msg: InboundMessage) -> None:
        """Cancel active tasks, revert file changes, and remove the last turn (simple flow)."""
        assert self._loop is not None

        await self._loop.cancel_session_tasks(msg.session_key)

        session = self._loop.sessions.get_or_create(msg.session_key)
        turn_start = session.get_last_turn_start_index()
        if turn_start == -1:
            await self._loop.bus.publish_outbound(OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id,
                content="Nothing to undo.",
                metadata=dict(msg.metadata or {}),
            ))
            return

        last_turn_entries = [
            e for e in session.undo_log if e.get("turn_start_index") == turn_start
        ]
        last_turn_msgs = session.messages[turn_start:]

        non_reversible: set[str] = set()
        for m in last_turn_msgs:
            for tc in m.get("tool_calls") or []:
                tool_name = tc.get("function", {}).get("name", "")
                if tool_name and tool_name not in _REVERSIBLE_TOOLS:
                    non_reversible.add(tool_name)

        reverted: list[str] = []
        revert_errors: list[str] = []
        for entry in reversed(last_turn_entries):
            if not entry.get("reversible", True):
                revert_errors.append(
                    f"{entry['tool_name']} {entry['path']}: prior content unavailable, not reverted"
                )
                continue
            try:
                path = Path(entry["path"])
                if entry.get("existed_before"):
                    prev = entry["previous_content"]
                    if not isinstance(prev, str):
                        revert_errors.append(
                            f"{entry['tool_name']} {entry['path']}: unexpected previous_content type, not reverted"
                        )
                        continue
                    path.write_text(prev, encoding="utf-8")
                else:
                    path.unlink(missing_ok=True)
                reverted.append(f"{entry['tool_name']} {entry['path']}")
            except Exception as exc:
                revert_errors.append(f"{entry['tool_name']} {entry['path']}: {exc}")

        session.undo_log = [e for e in session.undo_log if e.get("turn_start_index") != turn_start]
        user_count = sum(1 for m in last_turn_msgs if m.get("role") == "user")
        assistant_count = sum(1 for m in last_turn_msgs if m.get("role") == "assistant")
        session.drop_last_turn()
        self._loop.sessions.save(session)

        lines = ["↩️ Undid last turn"]
        for r in reverted:
            lines.append(f"• Reverted: {r}")
        for e in revert_errors:
            lines.append(f"• Error reverting: {e}")
        msg_parts: list[str] = []
        if user_count:
            msg_parts.append(f"{user_count} user message{'s' if user_count != 1 else ''}")
        if assistant_count:
            msg_parts.append(f"{assistant_count} assistant message{'s' if assistant_count != 1 else ''}")
        if msg_parts:
            lines.append(f"• Removed: {', '.join(msg_parts)}")
        for nr in sorted(non_reversible):
            lines.append(f"• Not reverted: {nr} side effects")

        await self._loop.bus.publish_outbound(OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id,
            content="\n".join(lines),
            metadata={**dict(msg.metadata or {}), "_undo_succeeded": True},
        ))

    # -----------------------------------------------------------------------
    # Telegram-side: inbound / sent hooks for message-deletion tracking
    # -----------------------------------------------------------------------

    def _on_user_message(self, chat_id: str, message_id: int, session_key: str | None) -> None:
        """Track each incoming user message so it can be deleted on a successful undo."""
        skey = session_key or f"telegram:{chat_id}"
        if skey not in self._session_turn_stack:
            self._session_turn_stack[skey] = []
        self._session_turn_stack[skey].append({"user_msg_id": message_id, "bot_msg_ids": []})

    def _on_bot_message_sent(self, chat_id: str, message_id: int, thread_id: int | None) -> None:
        """Track each bot reply so it can be deleted on a successful undo."""
        skey = (
            f"telegram:{chat_id}:topic:{thread_id}"
            if thread_id is not None
            else f"telegram:{chat_id}"
        )
        stack = self._session_turn_stack.get(skey)
        if stack:
            stack[-1]["bot_msg_ids"].append(message_id)

    # -----------------------------------------------------------------------
    # Telegram-side: outbound handlers (special message types)
    # -----------------------------------------------------------------------

    async def _show_undo_confirmation(self, msg: OutboundMessage) -> None:
        """Display an inline-keyboard confirmation dialog for /undo."""
        from telegram import InlineKeyboardButton, InlineKeyboardMarkup

        assert self._channel is not None
        plan: dict = msg.metadata.get("_undo_plan", {})
        thread_id = msg.metadata.get("message_thread_id")
        skey = (
            f"telegram:{msg.chat_id}:topic:{thread_id}"
            if thread_id is not None
            else f"telegram:{msg.chat_id}"
        )
        thread_kwargs: dict = {}
        if thread_id is not None:
            thread_kwargs["message_thread_id"] = thread_id

        self._channel._stop_typing(msg.chat_id)

        if plan.get("nothing"):
            return

        user_count = plan.get("user_count", 0)
        assistant_count = plan.get("assistant_count", 0)
        reversible = plan.get("reversible_actions", [])
        non_reversible = plan.get("non_reversible", [])

        lines = ["↩️ Undo last turn?"]
        msg_parts = []
        if user_count:
            msg_parts.append(f"{user_count} user message{'s' if user_count != 1 else ''}")
        if assistant_count:
            msg_parts.append(f"{assistant_count} nanobot message{'s' if assistant_count != 1 else ''}")
        if msg_parts:
            lines.append(f"Removes: {', '.join(msg_parts)}")
        for action in reversible:
            lines.append(f"• {action}")
        for nr in non_reversible:
            lines.append(f"• {nr} (not reversible)")

        confirm_data = f"undo_confirm:{skey}"
        cancel_data = f"undo_cancel:{skey}"
        keyboard = InlineKeyboardMarkup([[
            InlineKeyboardButton("✅ Confirm", callback_data=confirm_data),
            InlineKeyboardButton("❌ Cancel", callback_data=cancel_data),
        ]])

        try:
            sent = await self._channel._app.bot.send_message(
                chat_id=int(msg.chat_id),
                text="\n".join(lines),
                reply_markup=keyboard,
                **thread_kwargs,
            )
        except Exception as e:
            logger.warning("Failed to send undo confirmation keyboard: {}", e)
            return

        stack = self._session_turn_stack.get(skey, [])
        top = stack[-1] if stack else {}
        session_key_override = skey if thread_id is not None else None

        self._pending_undo[skey] = {
            "chat_id": msg.chat_id,
            "session_key": session_key_override,
            "skey": skey,
            "turn_start_index": plan.get("turn_start_index"),
            "confirmation_msg_id": sent.message_id,
            "user_msg_id": top.get("user_msg_id"),
            "bot_msg_ids": list(top.get("bot_msg_ids", [])),
            "message_thread_id": thread_id,
            "inbound_metadata": {
                k: v for k, v in (msg.metadata or {}).items()
                if not k.startswith("_undo")
            },
            "sender_id": (
                f"{msg.metadata.get('user_id')}|{msg.metadata.get('username')}"
                if msg.metadata.get("username")
                else str(msg.metadata.get("user_id", ""))
            ),
        }

    async def _handle_undo_result(self, msg: OutboundMessage) -> None:
        """Process the structured undo result published by _handle_undo_apply().

        Deletes bubbles and pops the turn stack only on confirmed success.
        On expired/nothing/error, deletes the confirmation keyboard only.
        """
        assert self._channel is not None
        result: dict = msg.metadata.get("_undo_result", {})
        skey: str = msg.metadata.get("_pending_skey", f"telegram:{msg.chat_id}")
        status: str = result.get("status", "error")

        self._channel._stop_typing(msg.chat_id)

        pending = self._pending_undo.pop(skey, None)

        if pending is not None:
            confirmation_msg_id = pending.get("confirmation_msg_id")
            chat_id_int = int(pending["chat_id"])
            if confirmation_msg_id:
                try:
                    await self._channel._app.bot.delete_message(
                        chat_id=chat_id_int, message_id=confirmation_msg_id
                    )
                except Exception as e:
                    logger.debug("Could not delete undo confirmation message {}: {}", confirmation_msg_id, e)

        if status == "success" and pending is not None:
            chat_id_int = int(pending["chat_id"])
            user_msg_id = pending.get("user_msg_id")
            if user_msg_id is not None:
                try:
                    await self._channel._app.bot.delete_message(
                        chat_id=chat_id_int, message_id=user_msg_id
                    )
                except Exception as e:
                    logger.debug("Could not delete user message {}: {}", user_msg_id, e)
            for bot_msg_id in pending.get("bot_msg_ids", []):
                try:
                    await self._channel._app.bot.delete_message(
                        chat_id=chat_id_int, message_id=bot_msg_id
                    )
                except Exception as e:
                    logger.debug("Could not delete bot message {}: {}", bot_msg_id, e)

            stack = self._session_turn_stack.get(skey)
            if stack:
                stack.pop()
                if not stack:
                    del self._session_turn_stack[skey]

    # -----------------------------------------------------------------------
    # Telegram-side: command and callback handlers
    # -----------------------------------------------------------------------

    async def _on_undo_command(self, update, context) -> None:
        """Handle the /undo Telegram command."""
        if not update.message or not update.effective_user:
            return

        assert self._channel is not None
        message = update.message
        user = update.effective_user
        self._channel._remember_thread_context(message)
        chat_id = str(message.chat_id)
        session_key = self._channel._derive_topic_session_key(message)

        try:
            await message.delete()
        except Exception:
            pass

        meta = self._channel._build_message_metadata(message, user)
        meta["_undo_preview"] = True

        await self._channel._handle_message(
            sender_id=self._channel._sender_id(user),
            chat_id=chat_id,
            content="/undo",
            metadata=meta,
            session_key=session_key,
        )

    async def _on_undo_callback(self, update, context) -> None:
        """Handle inline keyboard button press for undo confirmation/cancellation."""
        query = update.callback_query
        if not query or not query.data or not update.effective_user:
            return

        assert self._channel is not None
        user = update.effective_user
        sender_id = self._channel._sender_id(user)

        if not self._channel.is_allowed(sender_id):
            await query.answer("Access denied.")
            return

        data: str = query.data
        if data.startswith("undo_confirm:"):
            skey = data[len("undo_confirm:"):]
            await self._handle_undo_confirm(query, skey)
        elif data.startswith("undo_cancel:"):
            skey = data[len("undo_cancel:"):]
            await self._handle_undo_cancel(query, skey)

    async def _handle_undo_confirm(self, query, skey: str) -> None:
        """Forward the undo-apply request to the backend after user confirmation."""
        assert self._channel is not None
        pending = self._pending_undo.get(skey)
        if pending is None:
            await query.answer("Undo already processed or expired.")
            return

        await query.answer()

        turn_start_index = pending.get("turn_start_index")
        chat_id = pending["chat_id"]
        session_key_override = pending.get("session_key")

        inbound_meta = dict(pending.get("inbound_metadata") or {})
        inbound_meta["_undo_apply_turn_start"] = turn_start_index
        inbound_meta["_pending_skey"] = skey

        await self._channel._handle_message(
            sender_id=pending.get("sender_id", ""),
            chat_id=chat_id,
            content="/undo",
            metadata=inbound_meta,
            session_key=session_key_override,
        )

    async def _handle_undo_cancel(self, query, skey: str) -> None:
        """Cancel the pending undo after the user clicked Cancel."""
        assert self._channel is not None
        pending = self._pending_undo.pop(skey, None)

        await query.answer("Undo cancelled")

        if pending is None:
            return
        confirmation_msg_id = pending["confirmation_msg_id"]
        chat_id = pending["chat_id"]
        try:
            await self._channel._app.bot.delete_message(
                chat_id=int(chat_id), message_id=confirmation_msg_id
            )
        except Exception as e:
            logger.debug("Could not delete undo confirmation message {}: {}", confirmation_msg_id, e)
