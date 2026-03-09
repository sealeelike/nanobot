"""Configuration loading utilities."""

import json
from pathlib import Path

from nanobot.config.schema import Config


# Global variable to store current config path (for multi-instance support)
_current_config_path: Path | None = None


def set_config_path(path: Path) -> None:
    """Set the current config path (used to derive data directory)."""
    global _current_config_path
    _current_config_path = path


def get_config_path() -> Path:
    """Get the configuration file path."""
    if _current_config_path:
        return _current_config_path
    return Path.home() / ".nanobot" / "config.json"


def load_config(config_path: Path | None = None) -> Config:
    """
    Load configuration from file or create default.

    Args:
        config_path: Optional path to config file. Uses default if not provided.

    Returns:
        Loaded configuration object.
    """
    path = config_path or get_config_path()

    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            data = _migrate_config(data)
            return Config.model_validate(data)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Warning: Failed to load config from {path}: {e}")
            print("Using default configuration.")

    return Config()


def save_config(config: Config, config_path: Path | None = None) -> None:
    """
    Save configuration to file.

    Args:
        config: Configuration to save.
        config_path: Optional path to save to. Uses default if not provided.
    """
    path = config_path or get_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    data = config.model_dump(by_alias=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _migrate_config(data: dict) -> dict:
    """Migrate old config formats to current."""
    # Move tools.exec.restrictToWorkspace → tools.restrictToWorkspace
    tools = data.get("tools", {})
    exec_cfg = tools.get("exec", {})
    if "restrictToWorkspace" in exec_cfg and "restrictToWorkspace" not in tools:
        tools["restrictToWorkspace"] = exec_cfg.pop("restrictToWorkspace")

    # Migrate top-level default_provider → agents.defaults.provider
    if "default_provider" in data:
        agents = data.setdefault("agents", {})
        defaults = agents.setdefault("defaults", {})
        if "provider" not in defaults:
            defaults["provider"] = data.pop("default_provider")
        else:
            data.pop("default_provider")

    # Migrate old flat telegram config → channels.telegram
    # Old format: {"telegram": {"bot_token": "...", "allowed_users": [...]}}
    if "telegram" in data and isinstance(data["telegram"], dict):
        old_tg = data.pop("telegram")
        channels = data.setdefault("channels", {})
        tg = channels.setdefault("telegram", {})
        if "bot_token" in old_tg and "token" not in tg:
            tg["token"] = old_tg["bot_token"]
        if "allowed_users" in old_tg and "allowFrom" not in tg and "allow_from" not in tg:
            tg["allowFrom"] = old_tg["allowed_users"]
        # Copy any remaining keys that don't conflict
        for k, v in old_tg.items():
            if k not in ("bot_token", "allowed_users") and k not in tg:
                tg[k] = v

    # Migrate base_url → api_base / apiBase in all provider configs
    # Handles both snake_case ("base_url") and camelCase ("baseUrl") old keys.
    # Always removes the old key to avoid unrecognised-field warnings.
    providers = data.get("providers", {})
    if isinstance(providers, dict):
        for _provider_name, pcfg in providers.items():
            if not isinstance(pcfg, dict):
                continue
            # snake_case variant
            if "base_url" in pcfg:
                if "api_base" not in pcfg and "apiBase" not in pcfg:
                    pcfg["apiBase"] = pcfg.pop("base_url")
                else:
                    pcfg.pop("base_url")
            # camelCase variant (baseUrl)
            if "baseUrl" in pcfg:
                if "api_base" not in pcfg and "apiBase" not in pcfg:
                    pcfg["apiBase"] = pcfg.pop("baseUrl")
                else:
                    pcfg.pop("baseUrl")

    return data
