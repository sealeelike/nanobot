"""Plugin loader and public re-exports for nanobot.plugins."""

from __future__ import annotations

import importlib

from loguru import logger

from nanobot.plugins.base import NanobotPlugin


def load_plugins(plugin_paths: list[str]) -> list[NanobotPlugin]:
    """Load and instantiate plugins from dotted class paths.

    Each entry must be the fully-qualified class path, e.g.
    ``"nanobot.plugins.undo.UndoPlugin"``.  Import errors are logged and
    skipped so that a misconfigured plugin does not crash the whole bot.
    """
    plugins: list[NanobotPlugin] = []
    for path in plugin_paths:
        try:
            module_path, class_name = path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
            plugin: NanobotPlugin = cls()
            plugins.append(plugin)
            logger.info("Loaded plugin: {} ({})", getattr(cls, "name", class_name) or class_name, path)
        except Exception as e:
            logger.error("Failed to load plugin {}: {}", path, e)
    return plugins


__all__ = ["NanobotPlugin", "load_plugins"]
