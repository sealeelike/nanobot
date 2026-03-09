"""Tests for per-subagent model selection via SpawnTool."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.agent.subagent import SubagentManager
from nanobot.agent.tools.spawn import SpawnTool
from nanobot.bus.queue import MessageBus


def _make_manager(default_model: str = "default-model") -> SubagentManager:
    provider = MagicMock()
    provider.get_default_model.return_value = default_model
    bus = MessageBus()
    return SubagentManager(provider=provider, workspace=MagicMock(), bus=bus)


class TestSpawnToolSchema:
    def test_model_parameter_in_schema(self):
        """SpawnTool schema must include an optional 'model' property."""
        mgr = _make_manager()
        tool = SpawnTool(mgr)
        props = tool.parameters["properties"]
        assert "model" in props
        assert props["model"]["type"] == "string"
        # model is not required
        assert "model" not in tool.parameters.get("required", [])

    def test_model_description_mentions_default(self):
        """SpawnTool 'model' description must mention defaulting to current model."""
        mgr = _make_manager()
        tool = SpawnTool(mgr)
        desc = tool.parameters["properties"]["model"]["description"].lower()
        assert "default" in desc


class TestSpawnToolExecute:
    @pytest.mark.asyncio
    async def test_execute_without_model_forwards_none(self):
        """execute() without model passes model=None to manager.spawn()."""
        mgr = _make_manager()
        mgr.spawn = AsyncMock(return_value="started")
        tool = SpawnTool(mgr)

        await tool.execute(task="do something")

        mgr.spawn.assert_called_once()
        _, kwargs = mgr.spawn.call_args
        assert kwargs.get("model") is None

    @pytest.mark.asyncio
    async def test_execute_with_model_forwards_model(self):
        """execute() with model= passes the model to manager.spawn()."""
        mgr = _make_manager()
        mgr.spawn = AsyncMock(return_value="started")
        tool = SpawnTool(mgr)

        await tool.execute(task="do something", model="gpt-4o")

        mgr.spawn.assert_called_once()
        _, kwargs = mgr.spawn.call_args
        assert kwargs.get("model") == "gpt-4o"


class TestSubagentManagerSpawn:
    @pytest.mark.asyncio
    async def test_spawn_without_model_uses_default(self):
        """spawn() without model runs the subagent with the manager's default model."""
        called_models: list[str] = []

        async def _fake_run(task_id, task, label, origin, model=None):
            called_models.append(model or "DEFAULT")

        mgr = _make_manager(default_model="default-model")
        mgr._run_subagent = _fake_run  # type: ignore[assignment]

        await mgr.spawn(task="test task")
        await asyncio.sleep(0)

        assert called_models == ["DEFAULT"]

    @pytest.mark.asyncio
    async def test_spawn_with_model_forwards_model(self):
        """spawn() with model= passes that model to _run_subagent."""
        called_models: list[str | None] = []

        async def _fake_run(task_id, task, label, origin, model=None):
            called_models.append(model)

        mgr = _make_manager(default_model="default-model")
        mgr._run_subagent = _fake_run  # type: ignore[assignment]

        await mgr.spawn(task="test task", model="special-model")
        await asyncio.sleep(0)

        assert called_models == ["special-model"]


class TestRunSubagentModelSelection:
    @pytest.mark.asyncio
    async def test_run_subagent_without_model_uses_global(self):
        """_run_subagent without model uses self.model."""
        used_models: list[str] = []

        async def _fake_chat(messages, tools, model, **kwargs):
            used_models.append(model)
            resp = MagicMock()
            resp.has_tool_calls = False
            resp.content = "done"
            return resp

        mgr = _make_manager(default_model="global-model")
        mgr.provider.chat = _fake_chat

        with patch.object(mgr, "_announce_result", new=AsyncMock()), \
             patch.object(mgr, "_build_subagent_prompt", return_value="prompt"):
            await mgr._run_subagent("id1", "task", "label", {"channel": "cli", "chat_id": "direct"})

        assert used_models == ["global-model"]

    @pytest.mark.asyncio
    async def test_run_subagent_with_model_uses_specified_model(self):
        """_run_subagent with model= uses that model instead of self.model."""
        used_models: list[str] = []

        async def _fake_chat(messages, tools, model, **kwargs):
            used_models.append(model)
            resp = MagicMock()
            resp.has_tool_calls = False
            resp.content = "done"
            return resp

        mgr = _make_manager(default_model="global-model")
        mgr.provider.chat = _fake_chat

        with patch.object(mgr, "_announce_result", new=AsyncMock()), \
             patch.object(mgr, "_build_subagent_prompt", return_value="prompt"):
            await mgr._run_subagent(
                "id2", "task", "label", {"channel": "cli", "chat_id": "direct"},
                model="override-model",
            )

        assert used_models == ["override-model"]
