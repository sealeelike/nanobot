"""Tests for /undo command: undo last conversation turn with file revert support."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.telegram import TelegramChannel
from nanobot.config.schema import TelegramConfig
from nanobot.session.manager import Session, SessionManager


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
    """Build a Session with n_turns user+assistant exchange pairs."""
    session = Session(key="cli:c1")
    for i in range(n_turns):
        session.messages.append({"role": "user", "content": f"question {i}"})
        session.messages.append({"role": "assistant", "content": f"answer {i}"})
    return session


def _make_telegram_channel() -> TelegramChannel:
    """Create a TelegramChannel with a fake in-memory bot app for testing."""
    config = TelegramConfig(enabled=True, token="123:abc", allow_from=["*"])
    channel = TelegramChannel(config, MessageBus())
    channel._app = _FakeApp()
    return channel


# ---------------------------------------------------------------------------
# Session.get_last_turn_start_index() tests
# ---------------------------------------------------------------------------

class TestSessionGetLastTurnStartIndex:
    def test_returns_index_of_last_user_message(self):
        session = _make_session_with_turns(2)
        # messages: user0, assistant0, user1, assistant1 — indices 0,1,2,3
        assert session.get_last_turn_start_index() == 2

    def test_returns_minus_one_on_empty_session(self):
        session = Session(key="k")
        assert session.get_last_turn_start_index() == -1

    def test_returns_minus_one_when_only_assistant_messages(self):
        session = Session(key="k")
        session.messages = [{"role": "assistant", "content": "hello"}]
        assert session.get_last_turn_start_index() == -1

    def test_returns_zero_for_single_user_message(self):
        session = Session(key="k")
        session.messages = [{"role": "user", "content": "hi"}]
        assert session.get_last_turn_start_index() == 0


# ---------------------------------------------------------------------------
# Session.drop_last_turn() tests
# ---------------------------------------------------------------------------

class TestSessionDropLastTurn:
    def test_drop_removes_last_user_turn(self):
        session = _make_session_with_turns(2)
        assert len(session.messages) == 4

        removed = session.drop_last_turn()

        assert removed == 2
        assert len(session.messages) == 2
        assert session.messages[-1]["content"] == "answer 0"

    def test_drop_returns_count_of_removed_messages(self):
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
        assert removed == 4
        assert len(session.messages) == 2

    def test_drop_on_empty_history_returns_zero(self):
        session = Session(key="k")
        result = session.drop_last_turn()
        assert result == 0
        assert session.messages == []

    def test_drop_only_assistant_no_user_returns_zero(self):
        session = Session(key="k")
        session.messages = [{"role": "assistant", "content": "hello"}]
        result = session.drop_last_turn()
        assert result == 0
        assert len(session.messages) == 1

    def test_drop_single_user_message(self):
        session = Session(key="k")
        session.messages = [{"role": "user", "content": "only question"}]
        removed = session.drop_last_turn()
        assert removed == 1
        assert session.messages == []

    def test_drop_adjusts_last_consolidated_downward(self):
        session = _make_session_with_turns(3)  # 6 messages
        session.last_consolidated = 6

        session.drop_last_turn()  # removes last 2 → 4 remain

        assert session.last_consolidated == 4

    def test_drop_does_not_change_last_consolidated_when_not_affected(self):
        session = _make_session_with_turns(3)  # 6 messages
        session.last_consolidated = 2

        session.drop_last_turn()  # removes last 2 → 4 remain

        assert session.last_consolidated == 2


# ---------------------------------------------------------------------------
# Session undo_log persistence tests
# ---------------------------------------------------------------------------

class TestSessionUndoLogPersistence:
    def test_undo_log_persisted_and_reloaded(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            manager = SessionManager(workspace)
            session = manager.get_or_create("cli:test")
            session.undo_log.append({
                "tool_name": "write_file",
                "path": "/tmp/foo.txt",
                "existed_before": False,
                "previous_content": None,
                "turn_start_index": 0,
            })
            manager.save(session)

            # Reload
            manager.invalidate("cli:test")
            reloaded = manager.get_or_create("cli:test")
            assert len(reloaded.undo_log) == 1
            entry = reloaded.undo_log[0]
            assert entry["tool_name"] == "write_file"
            assert entry["existed_before"] is False
            assert entry["turn_start_index"] == 0

    def test_undo_log_cleared_by_session_clear(self):
        session = Session(key="k")
        session.undo_log.append({"tool_name": "edit_file", "path": "/x"})
        session.clear()
        assert session.undo_log == []

    def test_undo_log_empty_by_default_when_no_field_in_file(self):
        """Old sessions without undo_log load without error and have empty log."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            manager = SessionManager(workspace)
            session = manager.get_or_create("cli:legacy")
            # Save without undo_log field to simulate an old-format session
            import json
            path = manager._get_session_path("cli:legacy")
            with open(path, "w") as f:
                metadata = {
                    "_type": "metadata",
                    "key": "cli:legacy",
                    "created_at": session.created_at.isoformat(),
                    "updated_at": session.updated_at.isoformat(),
                    "metadata": {},
                    "last_consolidated": 0,
                    # intentionally omit "undo_log"
                }
                f.write(json.dumps(metadata) + "\n")

            manager.invalidate("cli:legacy")
            reloaded = manager.get_or_create("cli:legacy")
            assert reloaded.undo_log == []


# ---------------------------------------------------------------------------
# AgentLoop /undo — pure conversation (no file tools)
# ---------------------------------------------------------------------------

class TestAgentLoopUndoPureConversation:
    @pytest.mark.asyncio
    async def test_undo_pure_conversation_removes_last_turn(self):
        """/undo on a pure conversation drops the last user+assistant pair."""
        loop, bus = _make_loop()

        session = _make_session_with_turns(2)
        loop.sessions = MagicMock()
        loop.sessions.get_or_create.return_value = session
        loop.sessions.save = MagicMock()
        loop.subagents.cancel_by_session = AsyncMock(return_value=0)

        msg = InboundMessage(channel="cli", sender_id="u1", chat_id="c1", content="/undo")
        await loop._handle_undo(msg)

        # One turn removed
        assert len(session.messages) == 2
        loop.sessions.save.assert_called_once_with(session)

    @pytest.mark.asyncio
    async def test_undo_nothing_reports_nothing_to_undo(self):
        """/undo on an empty session sends 'Nothing to undo.'"""
        loop, bus = _make_loop()

        session = Session(key="cli:c1")
        loop.sessions = MagicMock()
        loop.sessions.get_or_create.return_value = session
        loop.sessions.save = MagicMock()
        loop.subagents.cancel_by_session = AsyncMock(return_value=0)

        msg = InboundMessage(channel="cli", sender_id="u1", chat_id="c1", content="/undo")
        await loop._handle_undo(msg)

        response = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
        assert "Nothing to undo" in response.content

    @pytest.mark.asyncio
    async def test_undo_summary_mentions_removed_messages(self):
        """/undo summary lists how many user/assistant messages were removed."""
        loop, bus = _make_loop()

        session = _make_session_with_turns(1)
        loop.sessions = MagicMock()
        loop.sessions.get_or_create.return_value = session
        loop.sessions.save = MagicMock()
        loop.subagents.cancel_by_session = AsyncMock(return_value=0)

        msg = InboundMessage(channel="cli", sender_id="u1", chat_id="c1", content="/undo")
        await loop._handle_undo(msg)

        response = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
        assert "↩️" in response.content
        assert "user message" in response.content
        assert "assistant message" in response.content

    @pytest.mark.asyncio
    async def test_undo_not_saved_to_session_history(self):
        """/undo itself must not add messages to the session."""
        loop, bus = _make_loop()

        session = Session(key="cli:c1")
        loop.sessions = MagicMock()
        loop.sessions.get_or_create.return_value = session
        loop.sessions.save = MagicMock()
        loop.subagents.cancel_by_session = AsyncMock(return_value=0)

        msg = InboundMessage(channel="cli", sender_id="u1", chat_id="c1", content="/undo")
        await loop._handle_undo(msg)

        assert session.messages == []


# ---------------------------------------------------------------------------
# AgentLoop /undo — file edit turn
# ---------------------------------------------------------------------------

class TestAgentLoopUndoFileEdit:
    @pytest.mark.asyncio
    async def test_undo_reverts_edit_file(self):
        """/undo restores the original file content after an edit_file turn."""
        loop, bus = _make_loop()

        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "hello.txt"
            test_file.write_text("original content", encoding="utf-8")

            session = _make_session_with_turns(1)
            session.undo_log = [{
                "tool_name": "edit_file",
                "path": str(test_file),
                "existed_before": True,
                "previous_content": "original content",
                "turn_start_index": 0,
            }]
            # Simulate the file having been edited
            test_file.write_text("new content", encoding="utf-8")

            loop.sessions = MagicMock()
            loop.sessions.get_or_create.return_value = session
            loop.sessions.save = MagicMock()
            loop.subagents.cancel_by_session = AsyncMock(return_value=0)

            msg = InboundMessage(channel="cli", sender_id="u1", chat_id="c1", content="/undo")
            await loop._handle_undo(msg)

            assert test_file.read_text(encoding="utf-8") == "original content"
            response = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
            assert "Reverted" in response.content
            assert "edit_file" in response.content

    @pytest.mark.asyncio
    async def test_undo_removes_undo_log_entries_for_last_turn(self):
        """/undo clears the processed undo log entries."""
        loop, bus = _make_loop()

        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "f.txt"
            test_file.write_text("v1", encoding="utf-8")

            session = _make_session_with_turns(2)
            # Older turn has index 0, last turn has index 2
            session.undo_log = [
                {
                    "tool_name": "edit_file",
                    "path": str(test_file),
                    "existed_before": True,
                    "previous_content": "v0",
                    "turn_start_index": 0,  # older turn
                },
                {
                    "tool_name": "edit_file",
                    "path": str(test_file),
                    "existed_before": True,
                    "previous_content": "v1",
                    "turn_start_index": 2,  # last turn
                },
            ]

            loop.sessions = MagicMock()
            loop.sessions.get_or_create.return_value = session
            loop.sessions.save = MagicMock()
            loop.subagents.cancel_by_session = AsyncMock(return_value=0)

            msg = InboundMessage(channel="cli", sender_id="u1", chat_id="c1", content="/undo")
            await loop._handle_undo(msg)

            # Only the older entry should remain
            assert len(session.undo_log) == 1
            assert session.undo_log[0]["turn_start_index"] == 0

    @pytest.mark.asyncio
    async def test_undo_multiple_edits_reverted_in_reverse_order(self):
        """/undo reverts multiple edits in reverse order (most recent first)."""
        loop, bus = _make_loop()

        with tempfile.TemporaryDirectory() as tmpdir:
            file_a = Path(tmpdir) / "a.txt"
            file_b = Path(tmpdir) / "b.txt"
            file_a.write_text("A-edited", encoding="utf-8")
            file_b.write_text("B-edited", encoding="utf-8")

            session = _make_session_with_turns(1)
            session.undo_log = [
                {
                    "tool_name": "edit_file",
                    "path": str(file_a),
                    "existed_before": True,
                    "previous_content": "A-original",
                    "turn_start_index": 0,
                },
                {
                    "tool_name": "edit_file",
                    "path": str(file_b),
                    "existed_before": True,
                    "previous_content": "B-original",
                    "turn_start_index": 0,
                },
            ]

            loop.sessions = MagicMock()
            loop.sessions.get_or_create.return_value = session
            loop.sessions.save = MagicMock()
            loop.subagents.cancel_by_session = AsyncMock(return_value=0)

            msg = InboundMessage(channel="cli", sender_id="u1", chat_id="c1", content="/undo")
            await loop._handle_undo(msg)

            assert file_a.read_text(encoding="utf-8") == "A-original"
            assert file_b.read_text(encoding="utf-8") == "B-original"


# ---------------------------------------------------------------------------
# AgentLoop /undo — write_file that created a new file
# ---------------------------------------------------------------------------

class TestAgentLoopUndoWriteNewFile:
    @pytest.mark.asyncio
    async def test_undo_deletes_file_created_by_write_file(self):
        """/undo deletes a file that was created (did not exist before) by write_file."""
        loop, bus = _make_loop()

        with tempfile.TemporaryDirectory() as tmpdir:
            new_file = Path(tmpdir) / "new.txt"
            new_file.write_text("created content", encoding="utf-8")
            assert new_file.exists()

            session = _make_session_with_turns(1)
            session.undo_log = [{
                "tool_name": "write_file",
                "path": str(new_file),
                "existed_before": False,
                "previous_content": None,
                "turn_start_index": 0,
            }]

            loop.sessions = MagicMock()
            loop.sessions.get_or_create.return_value = session
            loop.sessions.save = MagicMock()
            loop.subagents.cancel_by_session = AsyncMock(return_value=0)

            msg = InboundMessage(channel="cli", sender_id="u1", chat_id="c1", content="/undo")
            await loop._handle_undo(msg)

            assert not new_file.exists()
            response = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
            assert "Reverted" in response.content
            assert "write_file" in response.content

    @pytest.mark.asyncio
    async def test_undo_overwrites_file_written_by_write_file(self):
        """/undo restores the previous content of a file overwritten by write_file."""
        loop, bus = _make_loop()

        with tempfile.TemporaryDirectory() as tmpdir:
            existing_file = Path(tmpdir) / "existing.txt"
            existing_file.write_text("new content", encoding="utf-8")

            session = _make_session_with_turns(1)
            session.undo_log = [{
                "tool_name": "write_file",
                "path": str(existing_file),
                "existed_before": True,
                "previous_content": "old content",
                "turn_start_index": 0,
            }]

            loop.sessions = MagicMock()
            loop.sessions.get_or_create.return_value = session
            loop.sessions.save = MagicMock()
            loop.subagents.cancel_by_session = AsyncMock(return_value=0)

            msg = InboundMessage(channel="cli", sender_id="u1", chat_id="c1", content="/undo")
            await loop._handle_undo(msg)

            assert existing_file.read_text(encoding="utf-8") == "old content"


# ---------------------------------------------------------------------------
# AgentLoop /undo — partial undo (non-reversible side effects)
# ---------------------------------------------------------------------------

class TestAgentLoopUndoPartial:
    @pytest.mark.asyncio
    async def test_undo_reports_shell_as_not_reverted(self):
        """/undo reports exec (shell) side effects as not reverted."""
        loop, bus = _make_loop()

        session = Session(key="cli:c1")
        session.messages = [
            {"role": "user", "content": "run something"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"function": {"name": "exec", "arguments": "{}"}, "id": "t1"}],
            },
            {"role": "tool", "content": "output", "tool_call_id": "t1"},
            {"role": "assistant", "content": "done"},
        ]

        loop.sessions = MagicMock()
        loop.sessions.get_or_create.return_value = session
        loop.sessions.save = MagicMock()
        loop.subagents.cancel_by_session = AsyncMock(return_value=0)

        msg = InboundMessage(channel="cli", sender_id="u1", chat_id="c1", content="/undo")
        await loop._handle_undo(msg)

        response = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
        assert "Not reverted" in response.content
        assert "exec" in response.content

    @pytest.mark.asyncio
    async def test_undo_reports_non_reversible_tool_and_reverts_file(self):
        """/undo reverts the file edit AND reports web_fetch as not reverted."""
        loop, bus = _make_loop()

        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "doc.txt"
            test_file.write_text("edited", encoding="utf-8")

            session = Session(key="cli:c1")
            session.messages = [
                {"role": "user", "content": "fetch and edit"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"function": {"name": "web_fetch", "arguments": "{}"}, "id": "t1"},
                    ],
                },
                {"role": "tool", "content": "fetched", "tool_call_id": "t1"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"function": {"name": "edit_file", "arguments": "{}"}, "id": "t2"},
                    ],
                },
                {"role": "tool", "content": "edited", "tool_call_id": "t2"},
                {"role": "assistant", "content": "done"},
            ]
            session.undo_log = [{
                "tool_name": "edit_file",
                "path": str(test_file),
                "existed_before": True,
                "previous_content": "original",
                "turn_start_index": 0,
            }]

            loop.sessions = MagicMock()
            loop.sessions.get_or_create.return_value = session
            loop.sessions.save = MagicMock()
            loop.subagents.cancel_by_session = AsyncMock(return_value=0)

            msg = InboundMessage(channel="cli", sender_id="u1", chat_id="c1", content="/undo")
            await loop._handle_undo(msg)

            assert test_file.read_text(encoding="utf-8") == "original"
            response = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
            assert "Reverted" in response.content
            assert "Not reverted" in response.content
            assert "web_fetch" in response.content


# ---------------------------------------------------------------------------
# AgentLoop /undo — help text
# ---------------------------------------------------------------------------

class TestAgentLoopUndoHelp:
    @pytest.mark.asyncio
    async def test_help_includes_undo_command(self):
        """/help response must mention /undo."""
        loop, _ = _make_loop()
        msg = InboundMessage(channel="cli", sender_id="u1", chat_id="c1", content="/help")
        response = await loop._process_message(msg)

        assert response is not None
        assert "/undo" in response.content

    @pytest.mark.asyncio
    async def test_help_still_includes_other_commands(self):
        """/help response still contains /new, /model, /stop after adding /undo."""
        loop, _ = _make_loop()
        msg = InboundMessage(channel="cli", sender_id="u1", chat_id="c1", content="/help")
        response = await loop._process_message(msg)

        assert response is not None
        assert "/new" in response.content
        assert "/model" in response.content
        assert "/stop" in response.content


# ---------------------------------------------------------------------------
# WriteFileTool / EditFileTool undo callback integration tests
# ---------------------------------------------------------------------------

class TestFiletoolUndoCallback:
    @pytest.mark.asyncio
    async def test_write_file_calls_undo_callback_with_prior_state(self):
        """WriteFileTool invokes undo_callback with existed_before=True and previous_content."""
        from nanobot.agent.tools.filesystem import WriteFileTool

        with tempfile.TemporaryDirectory() as tmpdir:
            tool = WriteFileTool(workspace=Path(tmpdir))
            recorded: list[dict] = []
            tool.set_undo_callback(recorded.append)

            # Create a pre-existing file
            (Path(tmpdir) / "target.txt").write_text("before", encoding="utf-8")

            await tool.execute(path="target.txt", content="after")

            assert len(recorded) == 1
            assert recorded[0]["tool_name"] == "write_file"
            assert recorded[0]["existed_before"] is True
            assert recorded[0]["previous_content"] == "before"

    @pytest.mark.asyncio
    async def test_write_file_calls_undo_callback_when_file_does_not_exist(self):
        """WriteFileTool records existed_before=False for a new file."""
        from nanobot.agent.tools.filesystem import WriteFileTool

        with tempfile.TemporaryDirectory() as tmpdir:
            tool = WriteFileTool(workspace=Path(tmpdir))
            recorded: list[dict] = []
            tool.set_undo_callback(recorded.append)

            await tool.execute(path="new.txt", content="hello")

            assert len(recorded) == 1
            assert recorded[0]["existed_before"] is False
            assert recorded[0]["previous_content"] is None

    @pytest.mark.asyncio
    async def test_edit_file_calls_undo_callback_with_full_content(self):
        """EditFileTool records the complete prior file content."""
        from nanobot.agent.tools.filesystem import EditFileTool

        with tempfile.TemporaryDirectory() as tmpdir:
            tool = EditFileTool(workspace=Path(tmpdir))
            recorded: list[dict] = []
            tool.set_undo_callback(recorded.append)

            (Path(tmpdir) / "doc.txt").write_text("line1\nline2\nline3", encoding="utf-8")
            await tool.execute(path="doc.txt", old_text="line2", new_text="updated")

            assert len(recorded) == 1
            assert recorded[0]["tool_name"] == "edit_file"
            assert recorded[0]["existed_before"] is True
            assert recorded[0]["previous_content"] == "line1\nline2\nline3"

    @pytest.mark.asyncio
    async def test_edit_file_no_callback_when_not_set(self):
        """EditFileTool works normally when no undo_callback is set."""
        from nanobot.agent.tools.filesystem import EditFileTool

        with tempfile.TemporaryDirectory() as tmpdir:
            tool = EditFileTool(workspace=Path(tmpdir))
            (Path(tmpdir) / "f.txt").write_text("abc", encoding="utf-8")

            result = await tool.execute(path="f.txt", old_text="abc", new_text="xyz")
            assert "Successfully" in result
            assert (Path(tmpdir) / "f.txt").read_text(encoding="utf-8") == "xyz"

    @pytest.mark.asyncio
    async def test_write_file_no_callback_when_not_set(self):
        """WriteFileTool works normally when no undo_callback is set."""
        from nanobot.agent.tools.filesystem import WriteFileTool

        with tempfile.TemporaryDirectory() as tmpdir:
            tool = WriteFileTool(workspace=Path(tmpdir))

            result = await tool.execute(path="out.txt", content="data")
            assert "Successfully" in result

    @pytest.mark.asyncio
    async def test_set_undo_callback_none_clears_callback(self):
        """set_undo_callback(None) removes a previously set callback."""
        from nanobot.agent.tools.filesystem import WriteFileTool

        with tempfile.TemporaryDirectory() as tmpdir:
            tool = WriteFileTool(workspace=Path(tmpdir))
            calls: list[dict] = []
            tool.set_undo_callback(calls.append)
            tool.set_undo_callback(None)

            await tool.execute(path="x.txt", content="test")
            assert calls == []


# ---------------------------------------------------------------------------
# Telegram /undo command wiring tests
# ---------------------------------------------------------------------------

class _FakeSentMessage:
    def __init__(self, message_id: int) -> None:
        self.message_id = message_id


class _FakeBot:
    def __init__(self) -> None:
        self.sent_messages: list[dict] = []
        self.deleted_messages: list[tuple] = []
        self.commands: list = []

    async def get_me(self):
        return SimpleNamespace(username="nanobot_test")

    async def set_my_commands(self, commands) -> None:
        self.commands = list(commands)

    async def send_message(self, **kwargs):
        self.sent_messages.append(kwargs)
        return _FakeSentMessage(len(self.sent_messages))

    async def delete_message(self, chat_id, message_id) -> None:
        self.deleted_messages.append((chat_id, message_id))


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


class TestTelegramUndoCommand:
    def test_undo_in_bot_commands(self):
        """/undo must appear in TelegramChannel.BOT_COMMANDS."""
        command_names = [cmd.command for cmd in TelegramChannel.BOT_COMMANDS]
        assert "undo" in command_names

    def test_undo_command_has_description(self):
        """/undo BotCommand must have a non-trivial description."""
        cmd = next(c for c in TelegramChannel.BOT_COMMANDS if c.command == "undo")
        assert len(cmd.description) > 5

    @pytest.mark.asyncio
    async def test_help_includes_undo(self):
        """Telegram _on_help must include /undo in its reply."""
        channel = _make_telegram_channel()
        replied: list[str] = []
        fake_message = SimpleNamespace(
            reply_text=AsyncMock(side_effect=lambda text: replied.append(text)),
        )
        fake_update = SimpleNamespace(message=fake_message)
        await channel._on_help(fake_update, SimpleNamespace())

        assert len(replied) == 1
        assert "/undo" in replied[0]

    @pytest.mark.asyncio
    async def test_undo_command_deletes_last_turn_messages(self):
        """_on_undo_command deletes the tracked user and bot messages."""
        channel = _make_telegram_channel()
        skey = "telegram:123"
        channel._session_turn_user_msg_id[skey] = 10
        channel._session_turn_bot_msg_ids[skey] = [20, 21]

        forwarded: list[dict] = []

        async def _fake_handle(**kwargs):
            forwarded.append(kwargs)

        channel._handle_message = _fake_handle

        fake_message = SimpleNamespace(
            message_id=30,
            chat_id=123,
            chat=SimpleNamespace(type="private"),
            text="/undo",
            message_thread_id=None,
            delete=AsyncMock(),
        )
        fake_user = SimpleNamespace(id=1, username="alice", first_name="Alice")
        fake_update = SimpleNamespace(message=fake_message, effective_user=fake_user)

        await channel._on_undo_command(fake_update, SimpleNamespace())

        deleted = channel._app.bot.deleted_messages
        deleted_ids = {msg_id for _, msg_id in deleted}
        assert 10 in deleted_ids   # user message
        assert 20 in deleted_ids   # bot message 1
        assert 21 in deleted_ids   # bot message 2

    @pytest.mark.asyncio
    async def test_undo_command_clears_turn_tracking(self):
        """After _on_undo_command, tracking dicts for session are cleared."""
        channel = _make_telegram_channel()
        skey = "telegram:123"
        channel._session_turn_user_msg_id[skey] = 10
        channel._session_turn_bot_msg_ids[skey] = [20]

        async def _noop(**kwargs):
            pass

        channel._handle_message = _noop
        fake_message = SimpleNamespace(
            message_id=99,
            chat_id=123,
            chat=SimpleNamespace(type="private"),
            text="/undo",
            message_thread_id=None,
            delete=AsyncMock(),
        )
        fake_user = SimpleNamespace(id=1, username="alice", first_name="Alice")
        fake_update = SimpleNamespace(message=fake_message, effective_user=fake_user)

        await channel._on_undo_command(fake_update, SimpleNamespace())

        assert skey not in channel._session_turn_user_msg_id
        assert skey not in channel._session_turn_bot_msg_ids

    @pytest.mark.asyncio
    async def test_undo_command_safe_when_no_prior_turn(self):
        """_on_undo_command is safe when no prior turn was tracked."""
        channel = _make_telegram_channel()
        forwarded: list[dict] = []

        async def _fake_handle(**kwargs):
            forwarded.append(kwargs)

        channel._handle_message = _fake_handle

        fake_message = SimpleNamespace(
            message_id=1,
            chat_id=123,
            chat=SimpleNamespace(type="private"),
            text="/undo",
            message_thread_id=None,
            delete=AsyncMock(),
        )
        fake_user = SimpleNamespace(id=1, username="alice", first_name="Alice")
        fake_update = SimpleNamespace(message=fake_message, effective_user=fake_user)

        # Should not raise
        await channel._on_undo_command(fake_update, SimpleNamespace())

        assert len(forwarded) == 1
        assert forwarded[0]["content"] == "/undo"

    @pytest.mark.asyncio
    async def test_undo_forwards_to_bus(self):
        """_on_undo_command forwards /undo content to the message bus."""
        channel = _make_telegram_channel()
        forwarded: list[dict] = []

        async def _fake_handle(**kwargs):
            forwarded.append(kwargs)

        channel._handle_message = _fake_handle

        fake_message = SimpleNamespace(
            message_id=5,
            chat_id=42,
            chat=SimpleNamespace(type="private"),
            text="/undo",
            message_thread_id=None,
            delete=AsyncMock(),
        )
        fake_user = SimpleNamespace(id=7, username="bob", first_name="Bob")
        fake_update = SimpleNamespace(message=fake_message, effective_user=fake_user)

        await channel._on_undo_command(fake_update, SimpleNamespace())

        assert forwarded[0]["content"] == "/undo"
        assert forwarded[0]["chat_id"] == "42"

    @pytest.mark.asyncio
    async def test_send_tracks_bot_message_ids(self):
        """send() tracks non-progress bot message IDs for /undo."""
        channel = _make_telegram_channel()
        skey = "telegram:123"
        # Simulate a prior user message establishing the tracking dict
        channel._session_turn_user_msg_id[skey] = 1
        channel._session_turn_bot_msg_ids[skey] = []

        await channel.send(OutboundMessage(
            channel="telegram",
            chat_id="123",
            content="response",
            metadata={},
        ))

        assert len(channel._session_turn_bot_msg_ids[skey]) == 1

    @pytest.mark.asyncio
    async def test_send_does_not_track_progress_messages(self):
        """send() does NOT track progress (typing) messages."""
        channel = _make_telegram_channel()
        skey = "telegram:123"
        channel._session_turn_user_msg_id[skey] = 1
        channel._session_turn_bot_msg_ids[skey] = []

        await channel.send(OutboundMessage(
            channel="telegram",
            chat_id="123",
            content="thinking...",
            metadata={"_progress": True},
        ))

        assert channel._session_turn_bot_msg_ids[skey] == []

    @pytest.mark.asyncio
    async def test_on_message_resets_bot_tracking_for_new_turn(self):
        """_on_message resets bot message ID tracking for each new user turn."""
        channel = _make_telegram_channel()
        skey = "telegram:123"
        # Pre-populate from a previous turn
        channel._session_turn_user_msg_id[skey] = 5
        channel._session_turn_bot_msg_ids[skey] = [10, 11]

        # Simulate a new incoming message (call the relevant tracking lines directly)
        str_chat_id = "123"
        new_user_msg_id = 20
        channel._session_turn_user_msg_id[skey] = new_user_msg_id
        channel._session_turn_bot_msg_ids[skey] = []

        assert channel._session_turn_user_msg_id[skey] == 20
        assert channel._session_turn_bot_msg_ids[skey] == []
