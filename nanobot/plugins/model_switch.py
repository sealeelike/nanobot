"""Model-switch plugin — /model hot-switching as a self-contained, loadable plugin module.

This plugin implements per-session model switching for both the CLI and Telegram
channels.  It is completely self-contained: it works with any nanobot build that
has the plugin hook infrastructure but does **not** require the built-in /model
handling to exist on ``AgentLoop`` or ``TelegramChannel``.

Usage — add to ``~/.nanobot/config.json``:

.. code-block:: json

    {
      "agents": {
        "defaults": {
          "plugins": ["nanobot.plugins.model_switch.ModelSwitchPlugin"]
        }
      }
    }

When the plugin is registered, it takes priority over the built-in /model
handling so it is safe to enable it alongside a fork that still contains the
original built-in implementation.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from loguru import logger

from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.plugins.base import NanobotPlugin

if TYPE_CHECKING:
    from nanobot.agent.loop import AgentLoop
    from nanobot.channels.telegram import TelegramChannel


class ModelSwitchPlugin(NanobotPlugin):
    """Plugin that implements /model hot-switching between configured candidate models."""

    name = "model_switch"

    def __init__(self) -> None:
        self._loop: AgentLoop | None = None
        self._channel: TelegramChannel | None = None
        # Per-session current model for Telegram inline-keyboard display.
        self._session_current_model: dict[str, str] = {}

    # -----------------------------------------------------------------------
    # Plugin lifecycle
    # -----------------------------------------------------------------------

    def setup_agent(self, loop: AgentLoop) -> None:
        """Register /model command handler with the AgentLoop."""
        self._loop = loop
        loop.register_command_handler(
            lambda msg: (
                msg.content.strip().lower() == "/model"
                or msg.content.strip().lower().startswith("/model ")
            ),
            self._handle_model_command,
        )

    def setup_telegram(self, channel: TelegramChannel, app) -> None:
        """Register Telegram command and callback handlers for /model."""
        from telegram.ext import CallbackQueryHandler, CommandHandler

        self._channel = channel
        app.add_handler(CommandHandler("model", self._on_model_command))
        app.add_handler(CallbackQueryHandler(self._on_model_callback, pattern="^model:"))

    # -----------------------------------------------------------------------
    # Agent-side command handler
    # -----------------------------------------------------------------------

    async def _handle_model_command(self, msg: InboundMessage) -> None:
        """Handle /model in the AgentLoop — list candidates or switch model."""
        assert self._loop is not None
        key = msg.session_key
        parts = msg.content.strip().split(None, 1)

        if len(parts) == 1:
            # List models
            current = self._loop.get_session_model(key)
            if not self._loop.candidate_models:
                config_hint = (
                    "{\n"
                    '  "agents": {\n'
                    '    "defaults": {\n'
                    f'      "model": "{current}",\n'
                    '      "candidateModels": [\n'
                    f'        "{current}",\n'
                    '        "openai/gpt-4o",\n'
                    '        "deepseek/deepseek-chat"\n'
                    "      ]\n"
                    "    }\n"
                    "  }\n"
                    "}"
                )
                response_content = (
                    f"Current model: `{current}`\n\n"
                    "No candidate models configured. "
                    "To enable `/model` hot-switching, add `candidateModels` to "
                    "`~/.nanobot/config.json`:\n"
                    f"```json\n{config_hint}\n```"
                )
            else:
                model_list = "\n".join(self._loop.candidate_models)
                response_content = (
                    f"Current model: `{current}`\n\nAvailable models:\n"
                    f"```\n{model_list}\n```\nUse `/model <name>` to switch."
                )
            await self._loop.bus.publish_outbound(OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id,
                content=response_content,
                metadata=dict(msg.metadata or {}),
            ))
        else:
            # Switch model
            new_model = parts[1].strip()
            self._loop.set_session_model(key, new_model)
            await self._loop.bus.publish_outbound(OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id,
                content=f"✅ Switched to: `{new_model}`",
                metadata={**dict(msg.metadata or {}), "_auto_delete": True},
            ))

    # -----------------------------------------------------------------------
    # Telegram-side: /model command and model selection callback
    # -----------------------------------------------------------------------

    async def _on_model_command(self, update, context) -> None:
        """Handle /model command: show inline keyboard or switch model directly."""
        if not update.message or not update.effective_user:
            return

        from telegram import InlineKeyboardButton, InlineKeyboardMarkup

        assert self._channel is not None
        message = update.message
        user = update.effective_user
        self._channel._remember_thread_context(message)
        sender_id = self._channel._sender_id(user)
        chat_id = str(message.chat_id)
        session_key = self._channel._derive_topic_session_key(message)

        try:
            await message.delete()
        except Exception:
            pass

        args = context.args or []
        if args:
            model_name = " ".join(args)
            skey = session_key or chat_id
            self._session_current_model[skey] = model_name

            await self._channel._handle_message(
                sender_id=sender_id,
                chat_id=chat_id,
                content=f"/model {model_name}",
                metadata={
                    **self._channel._build_message_metadata(message, user),
                    "_suppress_tg_response": True,
                },
                session_key=session_key,
            )
            try:
                sent = await self._channel._app.bot.send_message(
                    chat_id=int(chat_id),
                    text=f"✅ Switched to: {model_name}",
                )
                asyncio.create_task(self._delete_after(chat_id, sent.message_id))
            except Exception as e:
                logger.warning("Failed to send model switch confirmation: {}", e)
        else:
            candidate_models = self._channel.candidate_models
            if not candidate_models:
                # No candidates configured — fall through to agent to produce the config hint.
                if self._channel.is_allowed(sender_id):
                    await self._channel._handle_message(
                        sender_id=sender_id,
                        chat_id=chat_id,
                        content="/model",
                        metadata=self._channel._build_message_metadata(message, user),
                        session_key=session_key,
                    )
                return

            skey = session_key or chat_id
            current_model = self._session_current_model.get(skey, self._channel.default_model)
            keyboard_text = (
                f"Current: {current_model}\n\nSelect a model:"
                if current_model
                else "Select a model:"
            )
            keyboard = [
                [InlineKeyboardButton(model, callback_data=f"model:{model}")]
                for model in candidate_models
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            try:
                await self._channel._app.bot.send_message(
                    chat_id=int(chat_id),
                    text=keyboard_text,
                    reply_markup=reply_markup,
                )
            except Exception as e:
                logger.warning("Failed to send model keyboard: {}", e)

    async def _on_model_callback(self, update, context) -> None:
        """Handle inline keyboard button press for model selection."""
        query = update.callback_query
        if not query or not query.data or not update.effective_user:
            return

        assert self._channel is not None
        user = update.effective_user
        sender_id = self._channel._sender_id(user)

        if not self._channel.is_allowed(sender_id):
            await query.answer("Access denied.")
            return

        model_name = query.data[len("model:"):]
        await query.answer(f"✅ Switched to: {model_name}")

        try:
            await query.message.delete()
        except Exception:
            pass

        if not query.message:
            return

        chat_id = str(query.message.chat_id)
        session_key = self._channel._derive_topic_session_key(query.message)
        skey = session_key or chat_id
        self._session_current_model[skey] = model_name

        await self._channel._handle_message(
            sender_id=sender_id,
            chat_id=chat_id,
            content=f"/model {model_name}",
            metadata={"_suppress_tg_response": True},
            session_key=session_key,
        )

    async def _delete_after(self, chat_id: str, message_id: int, delay: float = 3.0) -> None:
        """Delete a message after a short delay."""
        try:
            await asyncio.sleep(delay)
            if self._channel and self._channel._app:
                await self._channel._app.bot.delete_message(
                    chat_id=int(chat_id), message_id=message_id
                )
        except Exception as e:
            logger.debug("Failed to delete message {}: {}", message_id, e)
