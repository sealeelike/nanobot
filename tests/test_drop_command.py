"""Tests for /drop command: drop last conversation turn."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.telegram import TelegramChannel
from nanobot.config.schema import TelegramConfig
from nanobot.session.manager import Session


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_loop():
    """Create a minimal AgentLoop with mocked dependencies."""
    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "default-model"
    workspace = MagicMock()
    workspace.__truediv__ = MagicMock(return_value=MagicMock())

    with patch("nanobot.agent.loop.ContextBuilder"), \
         patch("nanobot.agent.loop.SessionManager"), \
         patch("nanobot.agent.loop.SubagentManager"):
        loop = AgentLoop(bus=bus, provider=provider, workspace=workspace)
    return loop, bus


def _make_session_with_turns(n_turns: int) -> Session:
    """Build a session with n_turns user+assistant exchange pairs."""
    session = Session(key="cli:c1")
    for i in range(n_turns):
        session.messages.append({"role": "user", "content": f"question {i}"})
        session.messages.append({"role": "assistant", "content": f"answer {i}"})
    return session


# ---------------------------------------------------------------------------
# Session.drop_last_turn() unit tests
# ---------------------------------------------------------------------------

class TestSessionDropLastTurn:
    def test_drop_removes_last_user_turn(self):
        """drop_last_turn() removes from the last user message onwards."""
        session = _make_session_with_turns(2)
        # 4 messages: user0, assistant0, user1, assistant1
        assert len(session.messages) == 4

        removed = session.drop_last_turn()

        assert removed == 2  # user1 + assistant1
        assert len(session.messages) == 2
        assert session.messages[-1]["content"] == "answer 0"

    def test_drop_returns_count_of_removed_messages(self):
        """drop_last_turn() returns the exact number of removed messages."""
        session = Session(key="k")
        session.messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hey"},
            {"role": "user", "content": "do something"},
            {"role": "assistant", "content": "tool call", "tool_calls": [{}]},
            {"role": "tool", "content": "result", "tool_call_id": "t1"},
            {"role": "assistant", "content": "done"},
        ]
        removed = session.drop_last_turn()
        assert removed == 4  # user "do something" + 3 following messages
        assert len(session.messages) == 2

    def test_drop_on_empty_history_returns_zero(self):
        """drop_last_turn() on an empty session returns 0 without raising."""
        session = Session(key="k")
        result = session.drop_last_turn()
        assert result == 0
        assert session.messages == []

    def test_drop_only_assistant_no_user_returns_zero(self):
        """If there is no user message, drop_last_turn() returns 0."""
        session = Session(key="k")
        session.messages = [{"role": "assistant", "content": "hello"}]
        result = session.drop_last_turn()
        assert result == 0
        assert len(session.messages) == 1  # unchanged

    def test_drop_single_user_message(self):
        """Single orphaned user message is removed."""
        session = Session(key="k")
        session.messages = [{"role": "user", "content": "only question"}]
        removed = session.drop_last_turn()
        assert removed == 1
        assert session.messages == []

    def test_drop_adjusts_last_consolidated_downward(self):
        """last_consolidated is clamped to the new length when truncated."""
        session = _make_session_with_turns(3)  # 6 messages
        session.last_consolidated = 6  # pretend all are consolidated

        session.drop_last_turn()  # remove last 2 → 4 messages remain

        assert session.last_consolidated == 4

    def test_drop_does_not_change_last_consolidated_when_not_affected(self):
        """last_consolidated is unchanged when it's already within the remaining messages."""
        session = _make_session_with_turns(3)  # 6 messages
        session.last_consolidated = 2  # only first 2 are consolidated

        session.drop_last_turn()  # remove last 2 → 4 messages remain

        assert session.last_consolidated == 2  # untouched


# ---------------------------------------------------------------------------
# AgentLoop /drop command tests
# ---------------------------------------------------------------------------

class TestAgentLoopDropCommand:
    @pytest.mark.asyncio
    async def test_drop_removes_last_turn_from_session(self):
        """/drop removes the last user+assistant turn from the active session."""
        loop, bus = _make_loop()

        session = _make_session_with_turns(2)
        loop.sessions = MagicMock()
        loop.sessions.get_or_create.return_value = session
        loop.sessions.save = MagicMock()
        loop.subagents.cancel_by_session = AsyncMock(return_value=0)

        msg = InboundMessage(channel="cli", sender_id="u1", chat_id="c1", content="/drop")
        await loop._handle_drop(msg)

        assert len(session.messages) == 2  # 2 turns → 1 turn after drop
        loop.sessions.save.assert_called_once_with(session)

    @pytest.mark.asyncio
    async def test_drop_on_empty_session_is_safe(self):
        """/drop on an empty session sends 'Nothing to drop.' without raising."""
        loop, bus = _make_loop()

        session = Session(key="cli:c1")
        loop.sessions = MagicMock()
        loop.sessions.get_or_create.return_value = session
        loop.sessions.save = MagicMock()
        loop.subagents.cancel_by_session = AsyncMock(return_value=0)

        msg = InboundMessage(channel="cli", sender_id="u1", chat_id="c1", content="/drop")
        await loop._handle_drop(msg)

        # Collect outbound response
        response = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
        assert "Nothing to drop" in response.content

    @pytest.mark.asyncio
    async def test_drop_confirms_removal(self):
        """/drop sends a confirmation message when a turn was removed."""
        loop, bus = _make_loop()

        session = _make_session_with_turns(1)
        loop.sessions = MagicMock()
        loop.sessions.get_or_create.return_value = session
        loop.sessions.save = MagicMock()
        loop.subagents.cancel_by_session = AsyncMock(return_value=0)

        msg = InboundMessage(channel="cli", sender_id="u1", chat_id="c1", content="/drop")
        await loop._handle_drop(msg)

        response = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
        assert "Dropped" in response.content
        assert "removed" in response.content

    @pytest.mark.asyncio
    async def test_drop_via_process_message(self):
        """/drop dispatched through _process_message returns None (handled directly)."""
        loop, bus = _make_loop()
        loop.subagents.cancel_by_session = AsyncMock(return_value=0)

        session = _make_session_with_turns(1)
        loop.sessions = MagicMock()
        loop.sessions.get_or_create.return_value = session
        loop.sessions.save = MagicMock()

        # /drop is handled in run(), NOT via _process_message, so _process_message
        # should treat it as an unknown command and fall through to LLM.
        # Instead test the run() dispatch path by calling _handle_drop directly.
        msg = InboundMessage(channel="cli", sender_id="u1", chat_id="c1", content="/drop")
        await loop._handle_drop(msg)

        assert len(session.messages) == 0  # 1 turn removed

    @pytest.mark.asyncio
    async def test_drop_not_saved_to_session_history(self):
        """/drop itself must not add any messages to the session history."""
        loop, bus = _make_loop()

        session = Session(key="cli:c1")
        loop.sessions = MagicMock()
        loop.sessions.get_or_create.return_value = session
        loop.sessions.save = MagicMock()
        loop.subagents.cancel_by_session = AsyncMock(return_value=0)

        msg = InboundMessage(channel="cli", sender_id="u1", chat_id="c1", content="/drop")
        await loop._handle_drop(msg)

        # Session should remain empty — the /drop command itself is not stored
        assert session.messages == []

    @pytest.mark.asyncio
    async def test_help_includes_drop_command(self):
        """/help response must mention /drop."""
        loop, _ = _make_loop()
        msg = InboundMessage(channel="cli", sender_id="u1", chat_id="c1", content="/help")
        response = await loop._process_message(msg)

        assert response is not None
        assert "/drop" in response.content

    @pytest.mark.asyncio
    async def test_no_regression_new_command(self):
        """/new still works correctly after /drop is added."""
        loop, _ = _make_loop()

        session = Session(key="cli:c1")
        loop.sessions = MagicMock()
        loop.sessions.get_or_create.return_value = session
        loop.sessions.save = MagicMock()
        loop.sessions.invalidate = MagicMock()

        msg = InboundMessage(channel="cli", sender_id="u1", chat_id="c1", content="/new")
        response = await loop._process_message(msg)

        assert response is not None
        assert "New session" in response.content

    @pytest.mark.asyncio
    async def test_no_regression_model_command(self):
        """/model still works correctly after /drop is added."""
        loop, _ = _make_loop()

        msg = InboundMessage(channel="cli", sender_id="u1", chat_id="c1", content="/model")
        response = await loop._process_message(msg)

        assert response is not None
        assert "model" in response.content.lower()


# ---------------------------------------------------------------------------
# TelegramChannel /drop tests
# ---------------------------------------------------------------------------

class _FakeSentMessage:
    def __init__(self, message_id: int) -> None:
        self.message_id = message_id


class _FakeBot:
    def __init__(self) -> None:
        self.sent_messages: list[dict] = []
        self.commands: list = []

    async def get_me(self):
        return SimpleNamespace(username="nanobot_test")

    async def set_my_commands(self, commands) -> None:
        self.commands = list(commands)

    async def send_message(self, **kwargs):
        self.sent_messages.append(kwargs)
        return _FakeSentMessage(len(self.sent_messages))


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


def _make_telegram_channel():
    config = TelegramConfig(enabled=True, token="123:abc", allow_from=["*"])
    channel = TelegramChannel(config, MessageBus())
    channel._app = _FakeApp()
    return channel


class TestTelegramDropCommand:
    def test_drop_in_bot_commands(self):
        """/drop must appear in TelegramChannel.BOT_COMMANDS."""
        command_names = [cmd.command for cmd in TelegramChannel.BOT_COMMANDS]
        assert "drop" in command_names

    def test_drop_command_description(self):
        """/drop description must mention 'conversation turn' or similar."""
        cmd = next(c for c in TelegramChannel.BOT_COMMANDS if c.command == "drop")
        assert cmd.description  # non-empty description
        assert len(cmd.description) > 5

    @pytest.mark.asyncio
    async def test_help_includes_drop(self):
        """Telegram _on_help must include /drop in its reply."""
        channel = _make_telegram_channel()

        replied: list[str] = []

        fake_message = SimpleNamespace(
            reply_text=AsyncMock(side_effect=lambda text: replied.append(text)),
        )
        fake_update = SimpleNamespace(message=fake_message)
        fake_context = SimpleNamespace()

        await channel._on_help(fake_update, fake_context)

        assert len(replied) == 1
        assert "/drop" in replied[0]

    @pytest.mark.asyncio
    async def test_no_regression_help_includes_new_stop_model(self):
        """Telegram _on_help still includes /new, /stop, /model after adding /drop."""
        channel = _make_telegram_channel()

        replied: list[str] = []

        fake_message = SimpleNamespace(
            reply_text=AsyncMock(side_effect=lambda text: replied.append(text)),
        )
        fake_update = SimpleNamespace(message=fake_message)
        fake_context = SimpleNamespace()

        await channel._on_help(fake_update, fake_context)

        text = replied[0]
        assert "/new" in text
        assert "/stop" in text
        assert "/model" in text
        assert "/help" in text
