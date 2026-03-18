"""Tests for the nanobot plugin system.

Covers:
- Plugin loading (nanobot.plugins.load_plugins)
- UndoPlugin agent-side via AgentLoop.register_command_handler()
- ModelSwitchPlugin agent-side via AgentLoop.register_command_handler()
- AgentLoop.run() dispatches to plugin command handlers first
- AgentLoop public API helpers: set_session_model, get_session_model, cancel_session_tasks
- TelegramChannel plugin hooks: register_outbound_handler, register_inbound_hook, register_sent_hook
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.telegram import TelegramChannel
from nanobot.config.schema import TelegramConfig
from nanobot.plugins import load_plugins
from nanobot.plugins.base import NanobotPlugin
from nanobot.plugins.model_switch import ModelSwitchPlugin
from nanobot.plugins.undo import UndoPlugin
from nanobot.session.manager import Session, SessionManager


# ---------------------------------------------------------------------------
# Helpers shared with existing test suites
# ---------------------------------------------------------------------------

def _make_loop(candidate_models=None, plugins=None):
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
            plugins=plugins,
        )
    return loop, bus


def _make_session_with_turns(n_turns: int) -> Session:
    session = Session(key="cli:c1")
    for i in range(n_turns):
        session.messages.append({"role": "user", "content": f"question {i}"})
        session.messages.append({"role": "assistant", "content": f"answer {i}"})
    return session


def _make_telegram_channel(plugins=None) -> TelegramChannel:
    config = TelegramConfig(enabled=True, token="123:abc", allow_from=["*"])
    channel = TelegramChannel(config, MessageBus(), plugins=plugins)
    channel._app = _FakeApp()
    return channel


class _FakeApp:
    """Minimal fake for channel._app used in Telegram tests."""
    def __init__(self):
        self.bot = _FakeBot()


class _FakeBot:
    async def send_message(self, **kwargs):
        m = MagicMock()
        m.message_id = 999
        return m

    async def delete_message(self, **kwargs):
        pass


# ---------------------------------------------------------------------------
# load_plugins() tests
# ---------------------------------------------------------------------------

class TestLoadPlugins:
    def test_load_empty_list(self):
        plugins = load_plugins([])
        assert plugins == []

    def test_load_undo_plugin(self):
        plugins = load_plugins(["nanobot.plugins.undo.UndoPlugin"])
        assert len(plugins) == 1
        assert isinstance(plugins[0], UndoPlugin)
        assert plugins[0].name == "undo"

    def test_load_model_switch_plugin(self):
        plugins = load_plugins(["nanobot.plugins.model_switch.ModelSwitchPlugin"])
        assert len(plugins) == 1
        assert isinstance(plugins[0], ModelSwitchPlugin)
        assert plugins[0].name == "model_switch"

    def test_load_both_plugins(self):
        plugins = load_plugins([
            "nanobot.plugins.undo.UndoPlugin",
            "nanobot.plugins.model_switch.ModelSwitchPlugin",
        ])
        assert len(plugins) == 2

    def test_load_bad_path_logs_and_continues(self):
        """A bad plugin path should not raise — just log and skip."""
        plugins = load_plugins([
            "nanobot.plugins.nonexistent.FakePlugin",
            "nanobot.plugins.undo.UndoPlugin",
        ])
        assert len(plugins) == 1
        assert isinstance(plugins[0], UndoPlugin)


# ---------------------------------------------------------------------------
# NanobotPlugin base class
# ---------------------------------------------------------------------------

class TestNanobotPluginBase:
    def test_base_class_has_name(self):
        plugin = NanobotPlugin()
        assert hasattr(plugin, "name")

    def test_setup_agent_is_no_op(self):
        plugin = NanobotPlugin()
        loop, _ = _make_loop()
        # Should not raise
        plugin.setup_agent(loop)

    def test_setup_telegram_is_no_op(self):
        plugin = NanobotPlugin()
        # Should not raise with dummy args
        plugin.setup_telegram(None, None)


# ---------------------------------------------------------------------------
# AgentLoop.register_command_handler() and public API
# ---------------------------------------------------------------------------

class TestAgentLoopPluginAPI:
    def test_register_command_handler_appends(self):
        loop, _ = _make_loop()
        assert loop._command_handlers == []
        loop.register_command_handler(lambda m: True, AsyncMock())
        assert len(loop._command_handlers) == 1

    def test_set_and_get_session_model(self):
        loop, _ = _make_loop()
        key = "cli:test"
        assert loop.get_session_model(key) == "default-model"  # falls back to default
        loop.set_session_model(key, "gpt-4o")
        assert loop.get_session_model(key) == "gpt-4o"

    @pytest.mark.asyncio
    async def test_cancel_session_tasks(self):
        loop, _ = _make_loop()
        loop.subagents.cancel_by_session = AsyncMock(return_value=0)
        key = "cli:x"
        task = asyncio.create_task(asyncio.sleep(60))
        loop._active_tasks[key] = [task]
        await loop.cancel_session_tasks(key)
        assert task.cancelled()
        assert key not in loop._active_tasks


# ---------------------------------------------------------------------------
# AgentLoop.run() dispatches plugin handlers FIRST
# ---------------------------------------------------------------------------

class TestAgentLoopRunDispatch:
    @pytest.mark.asyncio
    async def test_plugin_handler_takes_priority_over_builtin_undo(self):
        """A plugin command handler matching /undo should be called instead of _handle_undo."""
        loop, bus = _make_loop()

        handled_by_plugin = []

        async def fake_handler(msg: InboundMessage):
            handled_by_plugin.append(msg.content)

        loop.register_command_handler(
            lambda m: m.content.strip().lower() == "/undo",
            fake_handler,
        )
        loop._handle_undo = AsyncMock()

        # Publish the /undo message and run one iteration of the loop.
        await bus.publish_inbound(InboundMessage(
            channel="cli", sender_id="u", chat_id="c", content="/undo"
        ))

        loop._running = True
        # Run a single iteration via a task that we cancel after one message.
        async def _run_one():
            try:
                await asyncio.wait_for(loop.run(), timeout=0.5)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass

        await _run_one()

        assert "/undo" in handled_by_plugin
        loop._handle_undo.assert_not_called()

    @pytest.mark.asyncio
    async def test_builtin_undo_fires_when_no_plugin_registered(self):
        """Without plugin handlers, built-in /undo is called."""
        loop, bus = _make_loop()
        loop._handle_undo = AsyncMock()

        await bus.publish_inbound(InboundMessage(
            channel="cli", sender_id="u", chat_id="c", content="/undo"
        ))

        loop._running = True
        async def _run_one():
            try:
                await asyncio.wait_for(loop.run(), timeout=0.5)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass

        await _run_one()
        loop._handle_undo.assert_called_once()


# ---------------------------------------------------------------------------
# UndoPlugin agent-side
# ---------------------------------------------------------------------------

class TestUndoPluginAgentSide:
    def _make_loop_with_session(self, session: Session):
        loop, bus = _make_loop()
        sm = MagicMock()
        sm.get_or_create.return_value = session
        sm.save = MagicMock()
        loop.sessions = sm
        loop.subagents.cancel_by_session = AsyncMock(return_value=0)
        return loop, bus

    @pytest.mark.asyncio
    async def test_undo_plugin_setup_agent_registers_handlers(self):
        loop, _ = _make_loop()
        plugin = UndoPlugin()
        plugin.setup_agent(loop)
        assert len(loop._command_handlers) == 3  # preview, apply, simple

    @pytest.mark.asyncio
    async def test_undo_nothing_to_undo(self):
        session = Session(key="cli:c1")
        loop, bus = self._make_loop_with_session(session)

        plugin = UndoPlugin()
        plugin.setup_agent(loop)

        msg = InboundMessage(channel="cli", sender_id="u", chat_id="c1", content="/undo")
        await plugin._handle_undo(msg)

        outbound = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
        assert "Nothing to undo" in outbound.content

    @pytest.mark.asyncio
    async def test_undo_removes_last_turn(self):
        session = _make_session_with_turns(2)
        loop, bus = self._make_loop_with_session(session)

        plugin = UndoPlugin()
        plugin.setup_agent(loop)

        msg = InboundMessage(channel="cli", sender_id="u", chat_id="c1", content="/undo")
        await plugin._handle_undo(msg)

        assert len(session.messages) == 2  # only the first turn remains
        outbound = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
        assert "Undid last turn" in outbound.content

    @pytest.mark.asyncio
    async def test_undo_preview_returns_plan(self):
        session = _make_session_with_turns(1)
        loop, bus = self._make_loop_with_session(session)

        plugin = UndoPlugin()
        plugin.setup_agent(loop)

        msg = InboundMessage(
            channel="cli", sender_id="u", chat_id="c1", content="/undo",
            metadata={"_undo_preview": True},
        )
        await plugin._handle_undo_preview(msg)

        outbound = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
        assert "_undo_plan" in outbound.metadata
        plan = outbound.metadata["_undo_plan"]
        assert plan.get("nothing") is False
        assert plan["user_count"] == 1

    @pytest.mark.asyncio
    async def test_undo_apply_expired_when_turn_shifted(self):
        session = _make_session_with_turns(2)
        loop, bus = self._make_loop_with_session(session)

        plugin = UndoPlugin()
        plugin.setup_agent(loop)

        # Request turn 0, but session currently ends at turn 2 (index 2).
        msg = InboundMessage(
            channel="cli", sender_id="u", chat_id="c1", content="/undo",
            metadata={"_undo_apply_turn_start": 0, "_pending_skey": "telegram:c1"},
        )
        await plugin._handle_undo_apply(msg)

        outbound = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
        result = outbound.metadata["_undo_result"]
        assert result["status"] == "expired"

    @pytest.mark.asyncio
    async def test_undo_apply_success_reverts_file(self):
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / "test.txt"
            f.write_text("old content")

            session = _make_session_with_turns(1)
            turn_start = session.get_last_turn_start_index()
            session.undo_log = [{
                "tool_name": "write_file",
                "path": str(f),
                "existed_before": True,
                "previous_content": "original content",
                "reversible": True,
                "turn_start_index": turn_start,
            }]

            loop, bus = self._make_loop_with_session(session)
            plugin = UndoPlugin()
            plugin.setup_agent(loop)

            msg = InboundMessage(
                channel="cli", sender_id="u", chat_id="c1", content="/undo",
                metadata={"_undo_apply_turn_start": turn_start, "_pending_skey": "telegram:c1"},
            )
            await plugin._handle_undo_apply(msg)

            outbound = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
            result = outbound.metadata["_undo_result"]
            assert result["status"] == "success"
            assert f.read_text() == "original content"


# ---------------------------------------------------------------------------
# ModelSwitchPlugin agent-side
# ---------------------------------------------------------------------------

class TestModelSwitchPluginAgentSide:
    @pytest.mark.asyncio
    async def test_model_list_with_candidates(self):
        loop, bus = _make_loop(candidate_models=["gpt-4o", "claude"])
        plugin = ModelSwitchPlugin()
        plugin.setup_agent(loop)

        msg = InboundMessage(channel="cli", sender_id="u", chat_id="c1", content="/model")
        await plugin._handle_model_command(msg)

        out = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
        assert "gpt-4o" in out.content
        assert "claude" in out.content

    @pytest.mark.asyncio
    async def test_model_switch_updates_loop(self):
        loop, bus = _make_loop(candidate_models=["gpt-4o"])
        plugin = ModelSwitchPlugin()
        plugin.setup_agent(loop)

        msg = InboundMessage(channel="cli", sender_id="u", chat_id="c1", content="/model gpt-4o")
        await plugin._handle_model_command(msg)

        assert loop.get_session_model("cli:c1") == "gpt-4o"
        out = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
        assert "gpt-4o" in out.content

    @pytest.mark.asyncio
    async def test_model_list_no_candidates_shows_config_hint(self):
        loop, bus = _make_loop(candidate_models=[])
        plugin = ModelSwitchPlugin()
        plugin.setup_agent(loop)

        msg = InboundMessage(channel="cli", sender_id="u", chat_id="c1", content="/model")
        await plugin._handle_model_command(msg)

        out = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
        assert "candidateModels" in out.content


# ---------------------------------------------------------------------------
# TelegramChannel plugin hooks
# ---------------------------------------------------------------------------

class TestTelegramChannelPluginHooks:
    def test_register_outbound_handler(self):
        channel = _make_telegram_channel()
        assert channel._outbound_handlers == []
        channel.register_outbound_handler(lambda m: True, AsyncMock())
        assert len(channel._outbound_handlers) == 1

    def test_register_inbound_hook(self):
        channel = _make_telegram_channel()
        assert channel._inbound_hooks == []
        channel.register_inbound_hook(lambda *a: None)
        assert len(channel._inbound_hooks) == 1

    def test_register_sent_hook(self):
        channel = _make_telegram_channel()
        assert channel._sent_hooks == []
        channel.register_sent_hook(lambda *a: None)
        assert len(channel._sent_hooks) == 1

    @pytest.mark.asyncio
    async def test_outbound_handler_intercepts_matching_message(self):
        """A registered outbound handler that matches should prevent built-in processing."""
        channel = _make_telegram_channel()
        captured = []

        async def my_handler(msg: OutboundMessage):
            captured.append(msg)

        channel.register_outbound_handler(
            lambda m: (m.metadata or {}).get("_custom_flag"),
            my_handler,
        )

        msg = OutboundMessage(
            channel="telegram", chat_id="123",
            content="hello",
            metadata={"_custom_flag": True},
        )
        # send() should call my_handler and return without sending the Telegram message.
        await channel.send(msg)
        assert len(captured) == 1

    @pytest.mark.asyncio
    async def test_outbound_handler_skips_non_matching(self):
        """A registered outbound handler that does NOT match falls through to normal send."""
        channel = _make_telegram_channel()
        captured = []

        async def my_handler(msg: OutboundMessage):
            captured.append(msg)

        channel.register_outbound_handler(
            lambda m: False,  # never matches
            my_handler,
        )

        # Suppress to avoid real Telegram send
        msg = OutboundMessage(
            channel="telegram", chat_id="123",
            content="",
            metadata={"_suppress_tg_response": True},
        )
        await channel.send(msg)
        assert captured == []  # handler never called


# ---------------------------------------------------------------------------
# Integration: UndoPlugin loaded via AgentLoop plugins= param
# ---------------------------------------------------------------------------

class TestUndoPluginIntegration:
    @pytest.mark.asyncio
    async def test_undo_plugin_works_end_to_end_via_run(self):
        """UndoPlugin handles /undo via the plugin dispatch path inside run()."""
        session = _make_session_with_turns(1)
        undo_plugin = UndoPlugin()
        loop, bus = _make_loop(plugins=[undo_plugin])

        sm = MagicMock()
        sm.get_or_create.return_value = session
        sm.save = MagicMock()
        loop.sessions = sm
        loop.subagents.cancel_by_session = AsyncMock(return_value=0)

        await bus.publish_inbound(InboundMessage(
            channel="cli", sender_id="u", chat_id="c1", content="/undo"
        ))

        loop._running = True
        async def _run_one():
            try:
                await asyncio.wait_for(loop.run(), timeout=0.5)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass

        await _run_one()

        # Session should have been modified (turn removed)
        assert len(session.messages) == 0

        out = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
        assert "Undid last turn" in out.content


# ---------------------------------------------------------------------------
# Integration: ModelSwitchPlugin loaded via AgentLoop plugins= param
# ---------------------------------------------------------------------------

class TestModelSwitchPluginIntegration:
    @pytest.mark.asyncio
    async def test_model_switch_plugin_works_end_to_end_via_run(self):
        """ModelSwitchPlugin handles /model gpt-4o via the plugin dispatch path inside run()."""
        model_plugin = ModelSwitchPlugin()
        loop, bus = _make_loop(candidate_models=["gpt-4o"], plugins=[model_plugin])

        await bus.publish_inbound(InboundMessage(
            channel="cli", sender_id="u", chat_id="c1", content="/model gpt-4o"
        ))

        loop._running = True
        async def _run_one():
            try:
                await asyncio.wait_for(loop.run(), timeout=0.5)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass

        await _run_one()

        assert loop.get_session_model("cli:c1") == "gpt-4o"
        out = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
        assert "gpt-4o" in out.content
