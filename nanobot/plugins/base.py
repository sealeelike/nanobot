"""Base class for nanobot plugins."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nanobot.agent.loop import AgentLoop
    from nanobot.channels.telegram import TelegramChannel


class NanobotPlugin:
    """Base class for nanobot plugins.

    Subclass this to create a plugin that adds new commands or Telegram UI features.
    Register your plugin class path in ``agents.defaults.plugins`` in config, e.g.:

    .. code-block:: json

        {
          "agents": {
            "defaults": {
              "plugins": ["nanobot.plugins.undo.UndoPlugin"]
            }
          }
        }

    Plugins are loaded at startup and receive references to the ``AgentLoop`` and,
    if running with Telegram, the ``TelegramChannel`` and its Telegram ``Application``.
    """

    #: Short identifier for the plugin (used in log messages).
    name: str = ""

    def setup_agent(self, loop: AgentLoop) -> None:
        """Called once with the AgentLoop after it is created.

        Override to register command handlers on the loop via
        ``loop.register_command_handler(predicate, handler)``.
        """

    def setup_telegram(self, channel: TelegramChannel, app) -> None:
        """Called once with the TelegramChannel and its Telegram Application.

        Called during ``TelegramChannel.start()`` before polling begins.
        Override to register Telegram ``CommandHandler`` / ``CallbackQueryHandler``
        objects and outbound/inbound hooks on the channel.
        """
