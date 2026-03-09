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

    @pytest.mark.asyncio
    async def test_write_file_records_reversible_false_when_read_fails(self):
        """WriteFileTool records reversible=False when prior content cannot be read."""
        import unittest.mock as _mock
        from nanobot.agent.tools.filesystem import WriteFileTool

        with tempfile.TemporaryDirectory() as tmpdir:
            tool = WriteFileTool(workspace=Path(tmpdir))
            recorded: list[dict] = []
            tool.set_undo_callback(recorded.append)

            target = Path(tmpdir) / "binary.bin"
            target.write_bytes(b"\xff\xfe")  # Invalid UTF-8

            # Patch read_text to raise to simulate an unreadable file
            with _mock.patch.object(Path, "read_text", side_effect=UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")):
                await tool.execute(path="binary.bin", content="safe text")

            assert len(recorded) == 1
            assert recorded[0]["existed_before"] is True
            assert recorded[0]["reversible"] is False
            assert recorded[0]["previous_content"] is None

    @pytest.mark.asyncio
    async def test_write_file_new_file_records_reversible_true(self):
        """WriteFileTool records reversible=True for a file that did not previously exist."""
        from nanobot.agent.tools.filesystem import WriteFileTool

        with tempfile.TemporaryDirectory() as tmpdir:
            tool = WriteFileTool(workspace=Path(tmpdir))
            recorded: list[dict] = []
            tool.set_undo_callback(recorded.append)

            await tool.execute(path="brand_new.txt", content="hello")

            assert len(recorded) == 1
            assert recorded[0]["existed_before"] is False
            assert recorded[0]["reversible"] is True
            assert recorded[0]["previous_content"] is None

    @pytest.mark.asyncio
    async def test_handle_undo_skips_non_reversible_entries(self):
        """_handle_undo skips entries with reversible=False and reports them as errors."""
        loop, bus = _make_loop()

        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "file.bin"
            target.write_bytes(b"\xff\xfe\x00\x00")

            session = _make_session_with_turns(1)
            session.undo_log = [{
                "tool_name": "write_file",
                "path": str(target),
                "existed_before": True,
                "previous_content": None,
                "reversible": False,
                "turn_start_index": 0,
            }]

            loop.sessions = MagicMock()
            loop.sessions.get_or_create.return_value = session
            loop.sessions.save = MagicMock()
            loop.subagents.cancel_by_session = AsyncMock(return_value=0)

            msg = InboundMessage(channel="cli", sender_id="u1", chat_id="c1", content="/undo")
            await loop._handle_undo(msg)

            response = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
            # The response should report an error for the non-reversible entry
            assert "Error reverting" in response.content or "prior content unavailable" in response.content
            # The file must NOT have been touched (not corrupted with empty string)
            assert target.read_bytes() == b"\xff\xfe\x00\x00"

    @pytest.mark.asyncio
    async def test_handle_undo_does_not_write_none_as_empty_string(self):
        """_handle_undo never restores a pre-existing file with None as empty string."""
        loop, bus = _make_loop()

        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "important.txt"
            target.write_text("real content", encoding="utf-8")

            session = _make_session_with_turns(1)
            # Simulate an entry where previous_content is None but existed_before=True
            # (this should never be reversible=True, but guard against it defensively)
            session.undo_log = [{
                "tool_name": "write_file",
                "path": str(target),
                "existed_before": True,
                "previous_content": None,
                "reversible": False,  # must be False when previous_content is None
                "turn_start_index": 0,
            }]

            loop.sessions = MagicMock()
            loop.sessions.get_or_create.return_value = session
            loop.sessions.save = MagicMock()
            loop.subagents.cancel_by_session = AsyncMock(return_value=0)

            msg = InboundMessage(channel="cli", sender_id="u1", chat_id="c1", content="/undo")
            await loop._handle_undo(msg)

            # File must remain unchanged — not overwritten with empty string
            assert target.read_text(encoding="utf-8") == "real content"

    @pytest.mark.asyncio
    async def test_handle_undo_defensive_guard_for_non_string_previous_content(self):
        """_handle_undo defensive guard: if previous_content is not a str, skip with error."""
        loop, bus = _make_loop()

        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "guard.txt"
            target.write_text("original", encoding="utf-8")

            session = _make_session_with_turns(1)
            # Malformed entry: reversible=True but previous_content is not a string
            session.undo_log = [{
                "tool_name": "write_file",
                "path": str(target),
                "existed_before": True,
                "previous_content": None,   # incorrect — should never happen in practice
                "reversible": True,          # misconfigured
                "turn_start_index": 0,
            }]

            loop.sessions = MagicMock()
            loop.sessions.get_or_create.return_value = session
            loop.sessions.save = MagicMock()
            loop.subagents.cancel_by_session = AsyncMock(return_value=0)

            msg = InboundMessage(channel="cli", sender_id="u1", chat_id="c1", content="/undo")
            await loop._handle_undo(msg)

            response = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
            # Reports as an error, not a successful revert
            assert "Error reverting" in response.content or "unexpected previous_content" in response.content
            # File must NOT have been corrupted
            assert target.read_text(encoding="utf-8") == "original"


# ---------------------------------------------------------------------------
# Bug 3: undo callback cleanup via try/finally
# ---------------------------------------------------------------------------

class TestUndoCallbackTryFinally:
    @pytest.mark.asyncio
    async def test_undo_callback_cleared_when_run_agent_loop_raises(self):
        """Undo callbacks are cleared even when _run_agent_loop raises an exception."""
        from nanobot.agent.tools.filesystem import WriteFileTool, EditFileTool

        loop, bus = _make_loop()

        # Use real file tools so we can check the callback state
        with tempfile.TemporaryDirectory() as tmpdir:
            write_tool = WriteFileTool(workspace=Path(tmpdir))
            edit_tool = EditFileTool(workspace=Path(tmpdir))
            loop.tools._tools = {
                "write_file": write_tool,
                "edit_file": edit_tool,
            }

            session = _make_session_with_turns(1)
            loop.sessions = MagicMock()
            loop.sessions.get_or_create.return_value = session
            loop.sessions.save = MagicMock()

            # Make _run_agent_loop raise
            async def _raising_loop(*args, **kwargs):
                raise RuntimeError("simulated loop failure")

            loop._run_agent_loop = _raising_loop
            loop.context = MagicMock()
            loop.context.build_messages.return_value = []
            loop.context._RUNTIME_CONTEXT_TAG = "__rt__"

            msg = InboundMessage(channel="cli", sender_id="u1", chat_id="c1", content="hello")
            try:
                await loop._process_message(msg)
            except Exception:
                pass

            # Callbacks must be cleared regardless of the exception
            assert write_tool._undo_callback is None
            assert edit_tool._undo_callback is None

    @pytest.mark.asyncio
    async def test_undo_callback_cleared_after_normal_completion(self):
        """Undo callbacks are cleared after a normal (non-raising) _run_agent_loop."""
        from nanobot.agent.tools.filesystem import WriteFileTool, EditFileTool

        loop, bus = _make_loop()

        with tempfile.TemporaryDirectory() as tmpdir:
            write_tool = WriteFileTool(workspace=Path(tmpdir))
            edit_tool = EditFileTool(workspace=Path(tmpdir))
            loop.tools._tools = {
                "write_file": write_tool,
                "edit_file": edit_tool,
            }

            session = _make_session_with_turns(0)
            loop.sessions = MagicMock()
            loop.sessions.get_or_create.return_value = session
            loop.sessions.save = MagicMock()

            async def _ok_loop(*args, **kwargs):
                return ("done", [], [{"role": "user", "content": "hi"}])

            loop._run_agent_loop = _ok_loop
            loop.context = MagicMock()
            loop.context.build_messages.return_value = [{"role": "user", "content": "hi"}]
            loop.context._RUNTIME_CONTEXT_TAG = "__rt__"
            loop.context.add_assistant_message.return_value = []

            msg = InboundMessage(channel="cli", sender_id="u1", chat_id="c1", content="hello")
            await loop._process_message(msg)

            assert write_tool._undo_callback is None
            assert edit_tool._undo_callback is None



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
    async def test_undo_command_does_not_delete_messages_before_confirmation(self):
        """_on_undo_command must NOT delete prior-turn messages; it requests a preview first."""
        channel = _make_telegram_channel()
        skey = "telegram:123"
        channel._session_turn_stack[skey] = [{"user_msg_id": 10, "bot_msg_ids": [20, 21]}]

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

        # Prior-turn messages must NOT be deleted yet — only the /undo command itself.
        deleted_ids = {msg_id for _, msg_id in channel._app.bot.deleted_messages}
        assert 10 not in deleted_ids   # user message still present
        assert 20 not in deleted_ids   # bot message still present
        assert 21 not in deleted_ids   # bot message still present

        # _on_undo_command forwards a preview request (not an immediate execute).
        assert len(forwarded) == 1
        assert forwarded[0]["metadata"].get("_undo_preview") is True
        # Candidate IDs are NOT in the forwarded metadata — they are captured later when
        # the confirmation keyboard is shown so they reflect the actual stack at confirm time.
        assert "_undo_candidate_user_msg_id" not in forwarded[0]["metadata"]
        assert "_undo_candidate_bot_msg_ids" not in forwarded[0]["metadata"]

    @pytest.mark.asyncio
    async def test_undo_command_tracking_still_present_after_forwarding(self):
        """Turn stack is NOT cleared by _on_undo_command — only by send() on confirmation."""
        channel = _make_telegram_channel()
        skey = "telegram:123"
        channel._session_turn_stack[skey] = [{"user_msg_id": 10, "bot_msg_ids": [20]}]

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

        # Turn stack is still present — it'll be popped by send() after confirmation.
        stack = channel._session_turn_stack.get(skey, [])
        assert len(stack) == 1
        assert stack[0]["user_msg_id"] == 10
        assert stack[0]["bot_msg_ids"] == [20]

    @pytest.mark.asyncio
    async def test_send_deletes_tracked_messages_on_undo_success(self):
        """send() deletes prior-turn messages when _undo_succeeded=True."""
        channel = _make_telegram_channel()
        skey = "telegram:123"
        channel._session_turn_stack[skey] = [{"user_msg_id": 10, "bot_msg_ids": [20, 21]}]

        await channel.send(OutboundMessage(
            channel="telegram",
            chat_id="123",
            content="↩️ Undid last turn",
            metadata={
                "_undo_succeeded": True,
                "_undo_candidate_user_msg_id": 10,
                "_undo_candidate_bot_msg_ids": [20, 21],
            },
        ))

        deleted_ids = {msg_id for _, msg_id in channel._app.bot.deleted_messages}
        assert 10 in deleted_ids   # user message deleted
        assert 20 in deleted_ids   # bot message 1 deleted
        assert 21 in deleted_ids   # bot message 2 deleted
        # Stack entry popped after successful undo; stack is now empty / removed
        assert not channel._session_turn_stack.get(skey)

    @pytest.mark.asyncio
    async def test_send_does_not_delete_on_nothing_to_undo(self):
        """send() must not delete prior-turn messages when undo found nothing to undo."""
        channel = _make_telegram_channel()
        skey = "telegram:123"
        channel._session_turn_stack[skey] = [{"user_msg_id": 10, "bot_msg_ids": [20]}]

        await channel.send(OutboundMessage(
            channel="telegram",
            chat_id="123",
            content="Nothing to undo.",
            metadata={
                # _undo_succeeded is intentionally absent / False
                "_undo_candidate_user_msg_id": 10,
                "_undo_candidate_bot_msg_ids": [20],
            },
        ))

        deleted_ids = {msg_id for _, msg_id in channel._app.bot.deleted_messages}
        assert 10 not in deleted_ids
        assert 20 not in deleted_ids
        # Stack must remain unchanged
        stack = channel._session_turn_stack.get(skey, [])
        assert len(stack) == 1
        assert stack[0]["user_msg_id"] == 10

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
        # Simulate a prior user message establishing the turn stack entry
        channel._session_turn_stack[skey] = [{"user_msg_id": 1, "bot_msg_ids": []}]

        await channel.send(OutboundMessage(
            channel="telegram",
            chat_id="123",
            content="response",
            metadata={},
        ))

        stack = channel._session_turn_stack.get(skey, [])
        assert len(stack) == 1
        assert len(stack[-1]["bot_msg_ids"]) == 1

    @pytest.mark.asyncio
    async def test_send_does_not_track_progress_messages(self):
        """send() does NOT track progress (typing) messages."""
        channel = _make_telegram_channel()
        skey = "telegram:123"
        channel._session_turn_stack[skey] = [{"user_msg_id": 1, "bot_msg_ids": []}]

        await channel.send(OutboundMessage(
            channel="telegram",
            chat_id="123",
            content="thinking...",
            metadata={"_progress": True},
        ))

        stack = channel._session_turn_stack.get(skey, [])
        assert stack and stack[-1]["bot_msg_ids"] == []

    @pytest.mark.asyncio
    async def test_on_message_pushes_new_turn_to_stack(self):
        """Each new user message pushes a fresh turn record onto the stack."""
        channel = _make_telegram_channel()
        skey = "telegram:123"
        # Pre-populate with a previous turn
        channel._session_turn_stack[skey] = [{"user_msg_id": 5, "bot_msg_ids": [10, 11]}]

        # Simulate the tracking lines that _on_message executes
        new_user_msg_id = 20
        if skey not in channel._session_turn_stack:
            channel._session_turn_stack[skey] = []
        channel._session_turn_stack[skey].append({"user_msg_id": new_user_msg_id, "bot_msg_ids": []})

        stack = channel._session_turn_stack[skey]
        assert len(stack) == 2
        assert stack[-1]["user_msg_id"] == 20
        assert stack[-1]["bot_msg_ids"] == []
        # Older turn is preserved
        assert stack[0]["user_msg_id"] == 5

    @pytest.mark.asyncio
    async def test_single_undo_removes_one_turns_bubbles(self):
        """A single /undo removes only the latest turn's message bubbles."""
        channel = _make_telegram_channel()
        skey = "telegram:123"
        # Two turns in the stack
        channel._session_turn_stack[skey] = [
            {"user_msg_id": 1, "bot_msg_ids": [2, 3]},    # turn 1 (older)
            {"user_msg_id": 10, "bot_msg_ids": [20, 21]},  # turn 2 (latest)
        ]

        # Simulate send() receiving _undo_succeeded for turn 2
        await channel.send(OutboundMessage(
            channel="telegram",
            chat_id="123",
            content="↩️ Undid last turn",
            metadata={
                "_undo_succeeded": True,
                "_undo_candidate_user_msg_id": 10,
                "_undo_candidate_bot_msg_ids": [20, 21],
            },
        ))

        deleted_ids = {msg_id for _, msg_id in channel._app.bot.deleted_messages}
        # Turn 2 bubbles deleted
        assert 10 in deleted_ids
        assert 20 in deleted_ids
        assert 21 in deleted_ids
        # Turn 1 bubbles NOT deleted
        assert 1 not in deleted_ids
        assert 2 not in deleted_ids
        assert 3 not in deleted_ids

        # Stack still has turn 1
        stack = channel._session_turn_stack.get(skey, [])
        assert len(stack) == 1
        assert stack[0]["user_msg_id"] == 1

    @pytest.mark.asyncio
    async def test_two_consecutive_undos_remove_two_turns_bubbles(self):
        """Two consecutive /undos each remove the correct turn's message bubbles."""
        channel = _make_telegram_channel()
        skey = "telegram:123"
        channel._session_turn_stack[skey] = [
            {"user_msg_id": 1, "bot_msg_ids": [2, 3]},    # turn 1 (older)
            {"user_msg_id": 10, "bot_msg_ids": [20, 21]},  # turn 2 (latest)
        ]

        # First undo — removes turn 2
        await channel.send(OutboundMessage(
            channel="telegram",
            chat_id="123",
            content="↩️ Undid last turn",
            metadata={
                "_undo_succeeded": True,
                "_undo_candidate_user_msg_id": 10,
                "_undo_candidate_bot_msg_ids": [20, 21],
            },
        ))

        deleted_after_first = {msg_id for _, msg_id in channel._app.bot.deleted_messages}
        assert 10 in deleted_after_first
        assert 20 in deleted_after_first
        assert 21 in deleted_after_first
        assert 1 not in deleted_after_first  # turn 1 still present

        # Stack now has only turn 1
        assert len(channel._session_turn_stack.get(skey, [])) == 1

        # Second undo — removes turn 1
        await channel.send(OutboundMessage(
            channel="telegram",
            chat_id="123",
            content="↩️ Undid last turn",
            metadata={
                "_undo_succeeded": True,
                "_undo_candidate_user_msg_id": 1,
                "_undo_candidate_bot_msg_ids": [2, 3],
            },
        ))

        deleted_after_second = {msg_id for _, msg_id in channel._app.bot.deleted_messages}
        assert 1 in deleted_after_second
        assert 2 in deleted_after_second
        assert 3 in deleted_after_second

        # Stack is now empty
        assert not channel._session_turn_stack.get(skey)

    @pytest.mark.asyncio
    async def test_nothing_to_undo_does_not_pop_stack(self):
        """When backend returns 'Nothing to undo.', the turn stack is left unchanged."""
        channel = _make_telegram_channel()
        skey = "telegram:123"
        channel._session_turn_stack[skey] = [
            {"user_msg_id": 5, "bot_msg_ids": [50]},
        ]

        await channel.send(OutboundMessage(
            channel="telegram",
            chat_id="123",
            content="Nothing to undo.",
            metadata={
                "_undo_candidate_user_msg_id": 5,
                "_undo_candidate_bot_msg_ids": [50],
                # No _undo_succeeded
            },
        ))

        deleted_ids = {msg_id for _, msg_id in channel._app.bot.deleted_messages}
        assert 5 not in deleted_ids
        assert 50 not in deleted_ids

        # Stack must be unchanged
        stack = channel._session_turn_stack.get(skey, [])
        assert len(stack) == 1
        assert stack[0]["user_msg_id"] == 5


# ---------------------------------------------------------------------------
# Undo confirmation UI — plan, keyboard, confirm, cancel
# ---------------------------------------------------------------------------

class TestTelegramUndoConfirmationUI:
    """Tests for the /undo confirmation keyboard UX introduced in the confirmation-step overhaul."""

    # ---- _show_undo_confirmation ----

    @pytest.mark.asyncio
    async def test_show_undo_confirmation_sends_keyboard_with_plan(self):
        """_show_undo_confirmation sends a message with an InlineKeyboardMarkup when there is something to undo."""
        from telegram import InlineKeyboardMarkup

        channel = _make_telegram_channel()
        skey = "telegram:456"
        channel._session_turn_stack[skey] = [{"user_msg_id": 10, "bot_msg_ids": [20, 21]}]

        plan = {
            "nothing": False,
            "user_count": 1,
            "assistant_count": 2,
            "reversible_actions": ["edit_file memory/MEMORY.md"],
            "non_reversible": [],
        }
        await channel._show_undo_confirmation(OutboundMessage(
            channel="telegram", chat_id="456", content="",
            metadata={"_undo_plan": plan},
        ))

        sent = channel._app.bot.sent_messages
        assert len(sent) == 1
        assert isinstance(sent[0].get("reply_markup"), InlineKeyboardMarkup)
        buttons = sent[0]["reply_markup"].inline_keyboard[0]
        labels = [b.text for b in buttons]
        assert "✅ Confirm" in labels
        assert "❌ Cancel" in labels

    @pytest.mark.asyncio
    async def test_show_undo_confirmation_text_includes_plan_details(self):
        """Confirmation message text includes user/bot count and reversible actions."""
        channel = _make_telegram_channel()
        skey = "telegram:456"
        channel._session_turn_stack[skey] = [{"user_msg_id": 10, "bot_msg_ids": [20]}]

        plan = {
            "nothing": False,
            "user_count": 1,
            "assistant_count": 1,
            "reversible_actions": ["edit_file memory/MEMORY.md", "write_file workspace/USER.md"],
            "non_reversible": ["exec"],
        }
        await channel._show_undo_confirmation(OutboundMessage(
            channel="telegram", chat_id="456", content="",
            metadata={"_undo_plan": plan},
        ))

        text = channel._app.bot.sent_messages[0]["text"]
        assert "• edit_file memory/MEMORY.md" in text
        assert "• write_file workspace/USER.md" in text
        assert "exec" in text  # non-reversible tool mentioned with "(not reversible)"
        assert "(not reversible)" in text
        assert "1 user message" in text
        assert "1 nanobot message" in text

    @pytest.mark.asyncio
    async def test_show_undo_confirmation_nothing_sends_self_deleting_notice(self):
        """When nothing can be undone, a self-deleting notice is sent and no keyboard is shown."""
        from telegram import InlineKeyboardMarkup

        channel = _make_telegram_channel()
        plan = {"nothing": True}

        await channel._show_undo_confirmation(OutboundMessage(
            channel="telegram", chat_id="789", content="",
            metadata={"_undo_plan": plan},
        ))

        sent = channel._app.bot.sent_messages
        assert len(sent) == 1
        assert "Nothing to undo" in sent[0]["text"]
        assert not isinstance(sent[0].get("reply_markup"), InlineKeyboardMarkup)

    @pytest.mark.asyncio
    async def test_show_undo_confirmation_stores_pending_state(self):
        """_show_undo_confirmation stores the pending undo state keyed by skey."""
        channel = _make_telegram_channel()
        skey = "telegram:456"
        channel._session_turn_stack[skey] = [{"user_msg_id": 10, "bot_msg_ids": [20, 21]}]

        plan = {
            "nothing": False,
            "user_count": 1,
            "assistant_count": 1,
            "reversible_actions": [],
            "non_reversible": [],
        }
        await channel._show_undo_confirmation(OutboundMessage(
            channel="telegram", chat_id="456", content="",
            metadata={"_undo_plan": plan, "user_id": 7, "username": "alice"},
        ))

        assert skey in channel._pending_undo
        pending = channel._pending_undo[skey]
        assert pending["user_msg_id"] == 10
        assert pending["bot_msg_ids"] == [20, 21]
        assert pending["chat_id"] == "456"
        assert pending["confirmation_msg_id"] == 1  # first sent message

    # ---- send() routing for _undo_plan ----

    @pytest.mark.asyncio
    async def test_send_routes_undo_plan_to_confirmation(self):
        """send() with _undo_plan in metadata shows confirmation keyboard, not a chat bubble."""
        from telegram import InlineKeyboardMarkup

        channel = _make_telegram_channel()
        skey = "telegram:123"
        channel._session_turn_stack[skey] = [{"user_msg_id": 5, "bot_msg_ids": [50]}]

        plan = {
            "nothing": False,
            "user_count": 1,
            "assistant_count": 1,
            "reversible_actions": [],
            "non_reversible": [],
        }
        await channel.send(OutboundMessage(
            channel="telegram", chat_id="123", content="",
            metadata={"_undo_plan": plan},
        ))

        sent = channel._app.bot.sent_messages
        assert len(sent) == 1
        assert isinstance(sent[0].get("reply_markup"), InlineKeyboardMarkup)

    @pytest.mark.asyncio
    async def test_send_routes_nothing_to_undo_to_self_deleting_notice(self):
        """send() with _undo_plan={'nothing': True} sends a self-deleting notice."""
        channel = _make_telegram_channel()

        await channel.send(OutboundMessage(
            channel="telegram", chat_id="123", content="",
            metadata={"_undo_plan": {"nothing": True}},
        ))

        sent = channel._app.bot.sent_messages
        assert len(sent) == 1
        assert "Nothing to undo" in sent[0]["text"]

    # ---- _handle_undo_confirm ----

    @pytest.mark.asyncio
    async def test_confirm_deletes_tracked_bubbles_and_confirmation_msg(self):
        """Clicking Confirm deletes the confirmation message and all tracked turn bubbles."""
        channel = _make_telegram_channel()
        skey = "telegram:123"
        channel._session_turn_stack[skey] = [{"user_msg_id": 10, "bot_msg_ids": [20, 21]}]
        channel._pending_undo[skey] = {
            "chat_id": "123",
            "session_key": None,
            "skey": skey,
            "confirmation_msg_id": 99,
            "user_msg_id": 10,
            "bot_msg_ids": [20, 21],
            "message_thread_id": None,
            "inbound_metadata": {},
            "sender_id": "1|alice",
        }

        forwarded: list[dict] = []

        async def _fake_handle(**kwargs):
            forwarded.append(kwargs)

        channel._handle_message = _fake_handle

        fake_query = SimpleNamespace(
            data=f"undo_confirm:{skey}",
            answer=AsyncMock(),
        )
        await channel._handle_undo_confirm(fake_query, skey)

        deleted_ids = {msg_id for _, msg_id in channel._app.bot.deleted_messages}
        assert 99 in deleted_ids   # confirmation message deleted
        assert 10 in deleted_ids   # user bubble deleted
        assert 20 in deleted_ids   # bot bubble deleted
        assert 21 in deleted_ids   # bot bubble deleted

    @pytest.mark.asyncio
    async def test_confirm_pops_turn_stack(self):
        """Clicking Confirm pops the last entry from the turn stack."""
        channel = _make_telegram_channel()
        skey = "telegram:123"
        channel._session_turn_stack[skey] = [
            {"user_msg_id": 1, "bot_msg_ids": [2]},
            {"user_msg_id": 10, "bot_msg_ids": [20]},
        ]
        channel._pending_undo[skey] = {
            "chat_id": "123",
            "session_key": None,
            "skey": skey,
            "confirmation_msg_id": 99,
            "user_msg_id": 10,
            "bot_msg_ids": [20],
            "message_thread_id": None,
            "inbound_metadata": {},
            "sender_id": "1|alice",
        }

        async def _noop(**kwargs):
            pass

        channel._handle_message = _noop
        fake_query = SimpleNamespace(data=f"undo_confirm:{skey}", answer=AsyncMock())
        await channel._handle_undo_confirm(fake_query, skey)

        stack = channel._session_turn_stack.get(skey, [])
        assert len(stack) == 1
        assert stack[0]["user_msg_id"] == 1

    @pytest.mark.asyncio
    async def test_confirm_uses_callback_answer_not_chat_bubble(self):
        """Confirm uses answerCallbackQuery (no extra bot.send_message) for success feedback."""
        channel = _make_telegram_channel()
        skey = "telegram:123"
        channel._session_turn_stack[skey] = [{"user_msg_id": 5, "bot_msg_ids": [50]}]
        channel._pending_undo[skey] = {
            "chat_id": "123",
            "session_key": None,
            "skey": skey,
            "confirmation_msg_id": 99,
            "user_msg_id": 5,
            "bot_msg_ids": [50],
            "message_thread_id": None,
            "inbound_metadata": {},
            "sender_id": "1|alice",
        }

        async def _noop(**kwargs):
            pass

        channel._handle_message = _noop
        fake_query = SimpleNamespace(data=f"undo_confirm:{skey}", answer=AsyncMock())
        await channel._handle_undo_confirm(fake_query, skey)

        # answer() must have been called with the success message
        fake_query.answer.assert_called_once()
        call_args = fake_query.answer.call_args[0]
        assert "Undid" in call_args[0] or "↩️" in call_args[0]

        # No new chat bubble sent
        assert len(channel._app.bot.sent_messages) == 0

    @pytest.mark.asyncio
    async def test_confirm_forwards_undo_apply_to_bus_with_suppress(self):
        """Confirm forwards the actual /undo to the bus with _suppress_tg_response=True."""
        channel = _make_telegram_channel()
        skey = "telegram:123"
        channel._session_turn_stack[skey] = [{"user_msg_id": 5, "bot_msg_ids": []}]
        channel._pending_undo[skey] = {
            "chat_id": "123",
            "session_key": None,
            "skey": skey,
            "confirmation_msg_id": 99,
            "user_msg_id": 5,
            "bot_msg_ids": [],
            "message_thread_id": None,
            "inbound_metadata": {"user_id": 1},
            "sender_id": "1|alice",
        }

        forwarded: list[dict] = []

        async def _fake_handle(**kwargs):
            forwarded.append(kwargs)

        channel._handle_message = _fake_handle
        fake_query = SimpleNamespace(data=f"undo_confirm:{skey}", answer=AsyncMock())
        await channel._handle_undo_confirm(fake_query, skey)

        assert len(forwarded) == 1
        assert forwarded[0]["content"] == "/undo"
        assert forwarded[0]["metadata"].get("_suppress_tg_response") is True

    @pytest.mark.asyncio
    async def test_confirm_removes_pending_state(self):
        """After confirm, the pending state is removed from _pending_undo."""
        channel = _make_telegram_channel()
        skey = "telegram:123"
        channel._session_turn_stack[skey] = [{"user_msg_id": 5, "bot_msg_ids": []}]
        channel._pending_undo[skey] = {
            "chat_id": "123",
            "session_key": None,
            "skey": skey,
            "confirmation_msg_id": 99,
            "user_msg_id": 5,
            "bot_msg_ids": [],
            "message_thread_id": None,
            "inbound_metadata": {},
            "sender_id": "",
        }

        async def _noop(**kwargs):
            pass

        channel._handle_message = _noop
        fake_query = SimpleNamespace(data=f"undo_confirm:{skey}", answer=AsyncMock())
        await channel._handle_undo_confirm(fake_query, skey)

        assert skey not in channel._pending_undo

    @pytest.mark.asyncio
    async def test_confirm_handles_already_processed(self):
        """Confirm on an already-processed skey returns a graceful answer."""
        channel = _make_telegram_channel()
        skey = "telegram:123"
        # No pending state — already processed
        fake_query = SimpleNamespace(data=f"undo_confirm:{skey}", answer=AsyncMock())
        await channel._handle_undo_confirm(fake_query, skey)

        fake_query.answer.assert_called_once()
        call_text = fake_query.answer.call_args[0][0]
        assert "expired" in call_text.lower() or "processed" in call_text.lower()

    # ---- _handle_undo_cancel ----

    @pytest.mark.asyncio
    async def test_cancel_deletes_confirmation_message_only(self):
        """Clicking Cancel deletes only the confirmation keyboard message."""
        channel = _make_telegram_channel()
        skey = "telegram:123"
        channel._session_turn_stack[skey] = [{"user_msg_id": 10, "bot_msg_ids": [20]}]
        channel._pending_undo[skey] = {
            "chat_id": "123",
            "session_key": None,
            "skey": skey,
            "confirmation_msg_id": 99,
            "user_msg_id": 10,
            "bot_msg_ids": [20],
            "message_thread_id": None,
            "inbound_metadata": {},
            "sender_id": "",
        }

        fake_query = SimpleNamespace(data=f"undo_cancel:{skey}", answer=AsyncMock())
        await channel._handle_undo_cancel(fake_query, skey)

        deleted_ids = {msg_id for _, msg_id in channel._app.bot.deleted_messages}
        assert 99 in deleted_ids   # confirmation message deleted
        assert 10 not in deleted_ids   # user bubble preserved
        assert 20 not in deleted_ids   # bot bubble preserved

    @pytest.mark.asyncio
    async def test_cancel_does_not_change_turn_stack(self):
        """Clicking Cancel leaves the turn stack unchanged."""
        channel = _make_telegram_channel()
        skey = "telegram:123"
        channel._session_turn_stack[skey] = [{"user_msg_id": 10, "bot_msg_ids": [20]}]
        channel._pending_undo[skey] = {
            "chat_id": "123",
            "session_key": None,
            "skey": skey,
            "confirmation_msg_id": 99,
            "user_msg_id": 10,
            "bot_msg_ids": [20],
            "message_thread_id": None,
            "inbound_metadata": {},
            "sender_id": "",
        }

        fake_query = SimpleNamespace(data=f"undo_cancel:{skey}", answer=AsyncMock())
        await channel._handle_undo_cancel(fake_query, skey)

        stack = channel._session_turn_stack.get(skey, [])
        assert len(stack) == 1
        assert stack[0]["user_msg_id"] == 10

    @pytest.mark.asyncio
    async def test_cancel_uses_callback_answer(self):
        """Cancel uses answerCallbackQuery for feedback, not a chat bubble."""
        channel = _make_telegram_channel()
        skey = "telegram:123"
        channel._pending_undo[skey] = {
            "chat_id": "123",
            "confirmation_msg_id": 99,
        }

        fake_query = SimpleNamespace(data=f"undo_cancel:{skey}", answer=AsyncMock())
        await channel._handle_undo_cancel(fake_query, skey)

        fake_query.answer.assert_called_once()
        call_text = fake_query.answer.call_args[0][0]
        assert "cancel" in call_text.lower()
        assert len(channel._app.bot.sent_messages) == 0

    @pytest.mark.asyncio
    async def test_cancel_removes_pending_state(self):
        """After cancel, the pending state is removed from _pending_undo."""
        channel = _make_telegram_channel()
        skey = "telegram:123"
        channel._pending_undo[skey] = {
            "chat_id": "123",
            "confirmation_msg_id": 99,
        }

        fake_query = SimpleNamespace(data=f"undo_cancel:{skey}", answer=AsyncMock())
        await channel._handle_undo_cancel(fake_query, skey)

        assert skey not in channel._pending_undo

    # ---- No persistent undo-result bubble ----

    @pytest.mark.asyncio
    async def test_no_persistent_undo_bubble_after_confirm(self):
        """After confirm + suppressed undo response, no extra chat bubble is left in chat."""
        channel = _make_telegram_channel()
        skey = "telegram:123"
        channel._session_turn_stack[skey] = [{"user_msg_id": 5, "bot_msg_ids": [50]}]
        channel._pending_undo[skey] = {
            "chat_id": "123",
            "session_key": None,
            "skey": skey,
            "confirmation_msg_id": 99,
            "user_msg_id": 5,
            "bot_msg_ids": [50],
            "message_thread_id": None,
            "inbound_metadata": {},
            "sender_id": "",
        }

        async def _noop(**kwargs):
            pass

        channel._handle_message = _noop
        fake_query = SimpleNamespace(data=f"undo_confirm:{skey}", answer=AsyncMock())
        await channel._handle_undo_confirm(fake_query, skey)

        # send() is now called with the suppressed undo response from AgentLoop
        await channel.send(OutboundMessage(
            channel="telegram", chat_id="123", content="↩️ Undid last turn",
            metadata={"_suppress_tg_response": True},
        ))

        # No new chat messages sent (neither confirmation nor result bubble)
        assert len(channel._app.bot.sent_messages) == 0

    # ---- Consecutive undos remain UI-consistent ----

    @pytest.mark.asyncio
    async def test_consecutive_undos_remain_ui_consistent(self):
        """Two consecutive /undos each show their own confirmation, operate on correct turns."""
        channel = _make_telegram_channel()
        skey = "telegram:123"
        # Two turns in the stack
        channel._session_turn_stack[skey] = [
            {"user_msg_id": 1, "bot_msg_ids": [2, 3]},
            {"user_msg_id": 10, "bot_msg_ids": [20, 21]},
        ]

        forwarded: list[dict] = []

        async def _fake_handle(**kwargs):
            forwarded.append(kwargs)

        channel._handle_message = _fake_handle

        # --- First undo: confirm turn 2 ---
        channel._pending_undo[skey] = {
            "chat_id": "123",
            "session_key": None,
            "skey": skey,
            "confirmation_msg_id": 99,
            "user_msg_id": 10,
            "bot_msg_ids": [20, 21],
            "message_thread_id": None,
            "inbound_metadata": {},
            "sender_id": "",
        }
        fake_query1 = SimpleNamespace(data=f"undo_confirm:{skey}", answer=AsyncMock())
        await channel._handle_undo_confirm(fake_query1, skey)

        deleted_after_first = {mid for _, mid in channel._app.bot.deleted_messages}
        assert 10 in deleted_after_first   # turn-2 user bubble
        assert 20 in deleted_after_first
        assert 21 in deleted_after_first
        assert 1 not in deleted_after_first   # turn-1 bubble untouched

        # Stack now has only turn 1
        assert len(channel._session_turn_stack.get(skey, [])) == 1

        # --- Second undo: confirm turn 1 ---
        channel._pending_undo[skey] = {
            "chat_id": "123",
            "session_key": None,
            "skey": skey,
            "confirmation_msg_id": 100,
            "user_msg_id": 1,
            "bot_msg_ids": [2, 3],
            "message_thread_id": None,
            "inbound_metadata": {},
            "sender_id": "",
        }
        fake_query2 = SimpleNamespace(data=f"undo_confirm:{skey}", answer=AsyncMock())
        await channel._handle_undo_confirm(fake_query2, skey)

        deleted_after_second = {mid for _, mid in channel._app.bot.deleted_messages}
        assert 1 in deleted_after_second
        assert 2 in deleted_after_second
        assert 3 in deleted_after_second

        # Stack is now empty
        assert not channel._session_turn_stack.get(skey)

    # ---- AgentLoop _plan_undo / _handle_undo_preview ----

    def test_plan_undo_returns_nothing_for_empty_session(self):
        """_plan_undo returns nothing:True when session has no messages."""
        loop, _ = _make_loop()
        session = Session(key="cli:c1")
        loop.sessions = MagicMock()
        loop.sessions.get_or_create.return_value = session

        plan = loop._plan_undo("cli:c1")
        assert plan["nothing"] is True

    def test_plan_undo_returns_correct_counts(self):
        """_plan_undo returns correct user/assistant counts from session messages."""
        loop, _ = _make_loop()
        session = _make_session_with_turns(2)
        loop.sessions = MagicMock()
        loop.sessions.get_or_create.return_value = session

        plan = loop._plan_undo("cli:c1")
        assert plan["nothing"] is False
        assert plan["user_count"] == 1
        assert plan["assistant_count"] == 1

    def test_plan_undo_includes_reversible_actions(self):
        """_plan_undo lists reversible actions from the undo_log."""
        loop, _ = _make_loop()
        session = _make_session_with_turns(1)
        session.undo_log = [{
            "tool_name": "edit_file",
            "path": "memory/MEMORY.md",
            "existed_before": True,
            "previous_content": "old",
            "turn_start_index": 0,
            "reversible": True,
        }]
        loop.sessions = MagicMock()
        loop.sessions.get_or_create.return_value = session

        plan = loop._plan_undo("cli:c1")
        assert "edit_file memory/MEMORY.md" in plan["reversible_actions"]

    @pytest.mark.asyncio
    async def test_handle_undo_preview_publishes_plan_not_execution(self):
        """_handle_undo_preview publishes an _undo_plan without changing session history."""
        loop, bus = _make_loop()
        session = _make_session_with_turns(1)
        loop.sessions = MagicMock()
        loop.sessions.get_or_create.return_value = session
        loop.subagents = MagicMock()
        loop.subagents.cancel_by_session = AsyncMock(return_value=0)

        msg = InboundMessage(
            channel="telegram", sender_id="u1", chat_id="c1",
            content="/undo",
            metadata={"_undo_preview": True},
        )
        await loop._handle_undo_preview(msg)

        out: OutboundMessage = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
        assert "_undo_plan" in out.metadata
        plan = out.metadata["_undo_plan"]
        assert plan["nothing"] is False
        assert plan["user_count"] == 1

        # Session history must NOT have been modified
        assert len(session.messages) == 2


