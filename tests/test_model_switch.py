"""Tests for /model command and per-session model switching."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.bus.queue import MessageBus
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.channels.telegram import TelegramChannel
from nanobot.config.schema import TelegramConfig
from nanobot.session.manager import Session
from telegram import InlineKeyboardMarkup


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_loop(candidate_models=None):
    """Create a minimal AgentLoop with mocked dependencies."""
    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "default-model"
    workspace = MagicMock()
    workspace.__truediv__ = MagicMock(return_value=MagicMock())

    with patch("nanobot.agent.loop.ContextBuilder"), \
         patch("nanobot.agent.loop.SessionManager"), \
         patch("nanobot.agent.loop.SubagentManager"):
        loop = AgentLoop(
            bus=bus,
            provider=provider,
            workspace=workspace,
            candidate_models=candidate_models or [],
        )
    return loop, bus


# ---------------------------------------------------------------------------
# AgentLoop /model command tests
# ---------------------------------------------------------------------------

class TestModelCommand:
    @pytest.mark.asyncio
    async def test_model_list_returns_code_block(self):
        """'/model' with no arg returns the candidate list in a code block."""
        loop, bus = _make_loop(candidate_models=["gpt-4o", "claude-3-5-sonnet"])
        msg = InboundMessage(
            channel="cli", sender_id="u1", chat_id="c1", content="/model"
        )
        response = await loop._process_message(msg)

        assert response is not None
        assert "gpt-4o" in response.content
        assert "claude-3-5-sonnet" in response.content
        assert "```" in response.content  # code block

    @pytest.mark.asyncio
    async def test_model_list_falls_back_to_default_model_when_no_candidates(self):
        """'/model' with no candidates lists the current default model."""
        loop, bus = _make_loop(candidate_models=[])
        msg = InboundMessage(
            channel="cli", sender_id="u1", chat_id="c1", content="/model"
        )
        response = await loop._process_message(msg)

        assert response is not None
        assert "default-model" in response.content

    @pytest.mark.asyncio
    async def test_model_switch_updates_session_model(self):
        """/model <name> updates the per-session model."""
        loop, bus = _make_loop(candidate_models=["gpt-4o", "claude-3"])
        msg = InboundMessage(
            channel="cli", sender_id="u1", chat_id="c1", content="/model gpt-4o"
        )
        response = await loop._process_message(msg)

        assert response is not None
        assert "gpt-4o" in response.content
        assert loop._session_models.get("cli:c1") == "gpt-4o"

    @pytest.mark.asyncio
    async def test_model_switch_not_saved_to_session_history(self):
        """/model <name> must NOT be persisted to the session message history."""
        loop, bus = _make_loop()
        # Manually install a real session so we can inspect it
        session = Session(key="cli:c1")
        loop.sessions = MagicMock()
        loop.sessions.get_or_create.return_value = session
        loop.sessions.save = MagicMock()

        msg = InboundMessage(
            channel="cli", sender_id="u1", chat_id="c1", content="/model new-model"
        )
        await loop._process_message(msg)

        # Session messages must remain empty — model switch not in context
        assert session.messages == []

    @pytest.mark.asyncio
    async def test_model_list_not_saved_to_session_history(self):
        """/model (list) must NOT be persisted to the session message history."""
        loop, bus = _make_loop(candidate_models=["a", "b"])
        session = Session(key="cli:c1")
        loop.sessions = MagicMock()
        loop.sessions.get_or_create.return_value = session
        loop.sessions.save = MagicMock()

        msg = InboundMessage(
            channel="cli", sender_id="u1", chat_id="c1", content="/model"
        )
        await loop._process_message(msg)

        assert session.messages == []

    @pytest.mark.asyncio
    async def test_model_switch_response_carries_auto_delete_flag(self):
        """/model <name> response has _auto_delete=True for channel-side cleanup."""
        loop, bus = _make_loop()
        msg = InboundMessage(
            channel="cli", sender_id="u1", chat_id="c1", content="/model fancy-model"
        )
        response = await loop._process_message(msg)

        assert response is not None
        assert response.metadata.get("_auto_delete") is True

    @pytest.mark.asyncio
    async def test_per_session_model_used_in_run_agent_loop(self):
        """/model switch is used when the agent calls the LLM."""
        loop, bus = _make_loop(candidate_models=["fast-model"])

        # Switch model for session
        switch_msg = InboundMessage(
            channel="cli", sender_id="u1", chat_id="c1", content="/model fast-model"
        )
        await loop._process_message(switch_msg)
        assert loop._session_models.get("cli:c1") == "fast-model"

        # Now set up for a real message: the provider should receive the overridden model
        called_with_model: list[str] = []

        async def _fake_chat(messages, tools, model, **kwargs):
            called_with_model.append(model)
            resp = MagicMock()
            resp.has_tool_calls = False
            resp.content = "hello"
            resp.finish_reason = "stop"
            resp.reasoning_content = None
            resp.thinking_blocks = None
            return resp

        loop.provider.chat = _fake_chat

        # Set up minimal context/session mocks
        loop.context = MagicMock()
        loop.context.build_messages.return_value = [{"role": "user", "content": "hi"}]
        loop.context.add_assistant_message.return_value = []
        loop.context._RUNTIME_CONTEXT_TAG = "__runtime__"
        loop.sessions = MagicMock()
        session = Session(key="cli:c1")
        loop._session_models["cli:c1"] = "fast-model"
        loop.sessions.get_or_create.return_value = session
        loop.sessions.save = MagicMock()

        msg = InboundMessage(
            channel="cli", sender_id="u1", chat_id="c1", content="hello"
        )
        await loop._process_message(msg)

        assert called_with_model == ["fast-model"]

    @pytest.mark.asyncio
    async def test_help_includes_model_command(self):
        """/help response must mention /model."""

        loop, bus = _make_loop()
        msg = InboundMessage(
            channel="cli", sender_id="u1", chat_id="c1", content="/help"
        )
        response = await loop._process_message(msg)

        assert response is not None
        assert "/model" in response.content


# ---------------------------------------------------------------------------
# TelegramChannel /model tests
# ---------------------------------------------------------------------------

class _FakeHTTPXRequest:
    def __init__(self, **kwargs) -> None:
        pass


class _FakeSentMessage:
    def __init__(self, message_id: int) -> None:
        self.message_id = message_id


class _FakeBot:
    def __init__(self) -> None:
        self.sent_messages: list[dict] = []
        self.deleted_messages: list[tuple] = []
        self.answered_queries: list[str] = []

    async def get_me(self):
        return SimpleNamespace(username="nanobot_test")

    async def set_my_commands(self, commands) -> None:
        self.commands = commands

    async def send_message(self, **kwargs):
        self.sent_messages.append(kwargs)
        return _FakeSentMessage(len(self.sent_messages))

    async def delete_message(self, chat_id, message_id) -> None:
        self.deleted_messages.append((chat_id, message_id))

    async def answer_callback_query(self, callback_query_id, text=None) -> None:
        self.answered_queries.append(text or "")


class _FakeApp:
    def __init__(self) -> None:
        self.bot = _FakeBot()
        self.handlers = []
        self.error_handlers = []

    def add_error_handler(self, handler) -> None:
        self.error_handlers.append(handler)

    def add_handler(self, handler) -> None:
        self.handlers.append(handler)

    async def initialize(self) -> None:
        pass

    async def start(self) -> None:
        pass


def _make_telegram_channel(candidate_models=None, default_model=""):
    config = TelegramConfig(enabled=True, token="123:abc", allow_from=["*"])
    channel = TelegramChannel(
        config,
        MessageBus(),
        candidate_models=candidate_models or [],
        default_model=default_model,
    )
    channel._app = _FakeApp()
    return channel


class TestTelegramModelCommand:
    @pytest.mark.asyncio
    async def test_model_command_shows_inline_keyboard(self):
        """'/model' with candidate models sends an InlineKeyboardMarkup."""
        channel = _make_telegram_channel(candidate_models=["gpt-4o", "claude-3"])

        # Build fake update
        fake_message = SimpleNamespace(
            message_id=10,
            chat_id=123,
            chat=SimpleNamespace(type="private"),
            text="/model",
            message_thread_id=None,
            delete=AsyncMock(),
        )
        fake_user = SimpleNamespace(id=1, username="alice", first_name="Alice")
        fake_context = SimpleNamespace(args=[])
        fake_update = SimpleNamespace(message=fake_message, effective_user=fake_user)

        await channel._on_model_command(fake_update, fake_context)

        # Bot should have sent a message with reply_markup (InlineKeyboardMarkup)
        sent = channel._app.bot.sent_messages
        assert len(sent) == 1
        assert isinstance(sent[0].get("reply_markup"), InlineKeyboardMarkup)
        buttons = sent[0]["reply_markup"].inline_keyboard
        model_labels = [btn.text for row in buttons for btn in row]
        assert "gpt-4o" in model_labels
        assert "claude-3" in model_labels

    @pytest.mark.asyncio
    async def test_model_command_with_arg_forwards_to_bus(self):
        """/model <name> in Telegram forwards the switch to the bus."""
        channel = _make_telegram_channel(candidate_models=["gpt-4o"])

        received: list = []

        async def _fake_handle_message(**kwargs):
            received.append(kwargs)

        channel._handle_message = _fake_handle_message

        fake_message = SimpleNamespace(
            message_id=10,
            chat_id=123,
            chat=SimpleNamespace(type="private"),
            text="/model gpt-4o",
            message_thread_id=None,
            delete=AsyncMock(),
        )
        fake_user = SimpleNamespace(id=1, username="alice", first_name="Alice")
        fake_context = SimpleNamespace(args=["gpt-4o"])
        fake_update = SimpleNamespace(message=fake_message, effective_user=fake_user)

        await channel._on_model_command(fake_update, fake_context)

        assert len(received) == 1
        assert received[0]["content"] == "/model gpt-4o"
        assert received[0]["metadata"].get("_suppress_tg_response") is True

    @pytest.mark.asyncio
    async def test_model_callback_answers_and_deletes_keyboard(self):
        """Clicking an inline model button answers callback and deletes the keyboard."""
        channel = _make_telegram_channel(candidate_models=["gpt-4o"])

        received: list = []

        async def _fake_handle_message(**kwargs):
            received.append(kwargs)

        channel._handle_message = _fake_handle_message

        fake_kb_message = SimpleNamespace(
            message_id=20,
            chat_id=123,
            chat=SimpleNamespace(type="private"),
            message_thread_id=None,
            delete=AsyncMock(),
        )
        fake_query = SimpleNamespace(
            data="model:gpt-4o",
            message=fake_kb_message,
            answer=AsyncMock(),
        )
        fake_user = SimpleNamespace(id=1, username="alice", first_name="Alice")
        fake_context = SimpleNamespace()
        fake_update = SimpleNamespace(callback_query=fake_query, effective_user=fake_user)

        await channel._on_model_callback(fake_update, fake_context)

        # Callback should be answered
        fake_query.answer.assert_called_once()
        # Keyboard message should be deleted
        fake_kb_message.delete.assert_called_once()
        # Model switch should be forwarded to bus
        assert len(received) == 1
        assert received[0]["content"] == "/model gpt-4o"
        assert received[0]["metadata"].get("_suppress_tg_response") is True

    @pytest.mark.asyncio
    async def test_send_suppresses_tg_response(self):
        """Messages with _suppress_tg_response are not forwarded to Telegram."""
        channel = _make_telegram_channel()

        await channel.send(
            OutboundMessage(
                channel="telegram",
                chat_id="123",
                content="should be suppressed",
                metadata={"_suppress_tg_response": True},
            )
        )

        assert channel._app.bot.sent_messages == []

    @pytest.mark.asyncio
    async def test_send_auto_delete_schedules_deletion(self):
        """Messages with _auto_delete=True get deleted after sending."""
        channel = _make_telegram_channel()

        deleted: list[tuple] = []

        async def _fake_delete(chat_id, message_id):
            deleted.append((chat_id, message_id))

        channel._app.bot.delete_message = _fake_delete

        await channel.send(
            OutboundMessage(
                channel="telegram",
                chat_id="123",
                content="✅ Switched to: gpt-4o",
                metadata={"_auto_delete": True},
            )
        )

        # Message was sent
        assert len(channel._app.bot.sent_messages) == 1

        # Wait for the auto-delete task (delay=3s default, use 0s for test)
        # Directly call _delete_after with 0 delay to test the mechanism
        await channel._delete_after("123", 1, delay=0)
        assert (123, 1) in deleted

    @pytest.mark.asyncio
    async def test_model_keyboard_shows_default_model_as_current(self):
        """/model keyboard header shows the configured default model when no switch has happened."""
        channel = _make_telegram_channel(
            candidate_models=["gpt-4o", "claude-3"],
            default_model="gpt-4o",
        )

        fake_message = SimpleNamespace(
            message_id=10,
            chat_id=123,
            chat=SimpleNamespace(type="private"),
            text="/model",
            message_thread_id=None,
            delete=AsyncMock(),
        )
        fake_user = SimpleNamespace(id=1, username="alice", first_name="Alice")
        fake_context = SimpleNamespace(args=[])
        fake_update = SimpleNamespace(message=fake_message, effective_user=fake_user)

        await channel._on_model_command(fake_update, fake_context)

        sent = channel._app.bot.sent_messages
        assert len(sent) == 1
        assert "gpt-4o" in sent[0]["text"]

    @pytest.mark.asyncio
    async def test_model_keyboard_shows_switched_model_as_current(self):
        """After switching via callback, /model keyboard header shows the new model."""
        channel = _make_telegram_channel(
            candidate_models=["gpt-4o", "claude-3"],
            default_model="gpt-4o",
        )

        async def _noop_handle(**kwargs):
            pass

        channel._handle_message = _noop_handle

        # Simulate a model switch via inline button
        fake_kb_message = SimpleNamespace(
            message_id=20,
            chat_id=123,
            chat=SimpleNamespace(type="private"),
            message_thread_id=None,
            delete=AsyncMock(),
        )
        fake_query = SimpleNamespace(
            data="model:claude-3",
            message=fake_kb_message,
            answer=AsyncMock(),
        )
        fake_user = SimpleNamespace(id=1, username="alice", first_name="Alice")
        fake_context = SimpleNamespace()
        fake_update = SimpleNamespace(callback_query=fake_query, effective_user=fake_user)

        await channel._on_model_callback(fake_update, fake_context)

        # Now send /model again — the keyboard should show "claude-3" as current
        fake_message = SimpleNamespace(
            message_id=30,
            chat_id=123,
            chat=SimpleNamespace(type="private"),
            text="/model",
            message_thread_id=None,
            delete=AsyncMock(),
        )
        fake_update2 = SimpleNamespace(message=fake_message, effective_user=fake_user)
        await channel._on_model_command(fake_update2, SimpleNamespace(args=[]))

        sent = channel._app.bot.sent_messages
        assert len(sent) == 1  # only the keyboard (callback deleted original)
        assert "claude-3" in sent[0]["text"]

    @pytest.mark.asyncio
    async def test_model_command_with_arg_records_current_model(self):
        """/model <name> records the model so the next keyboard shows it as current."""
        channel = _make_telegram_channel(
            candidate_models=["gpt-4o", "claude-3"],
            default_model="gpt-4o",
        )

        async def _noop_handle(**kwargs):
            pass

        channel._handle_message = _noop_handle

        fake_message = SimpleNamespace(
            message_id=10,
            chat_id=123,
            chat=SimpleNamespace(type="private"),
            text="/model claude-3",
            message_thread_id=None,
            delete=AsyncMock(),
        )
        fake_user = SimpleNamespace(id=1, username="alice", first_name="Alice")
        fake_update = SimpleNamespace(message=fake_message, effective_user=fake_user)

        await channel._on_model_command(fake_update, SimpleNamespace(args=["claude-3"]))

        # The channel should have recorded the new current model
        assert channel._session_current_model.get("123") == "claude-3"

        # Next /model keyboard should reflect it
        fake_message2 = SimpleNamespace(
            message_id=11,
            chat_id=123,
            chat=SimpleNamespace(type="private"),
            text="/model",
            message_thread_id=None,
            delete=AsyncMock(),
        )
        fake_update2 = SimpleNamespace(message=fake_message2, effective_user=fake_user)
        await channel._on_model_command(fake_update2, SimpleNamespace(args=[]))

        sent = channel._app.bot.sent_messages
        # Only the keyboard message is sent (confirmation auto-deletes, but that's a task)
        keyboard_msgs = [m for m in sent if "reply_markup" in m]
        assert len(keyboard_msgs) == 1
        assert "claude-3" in keyboard_msgs[0]["text"]
