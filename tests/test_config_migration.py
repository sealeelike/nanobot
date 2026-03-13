"""Tests for config migration logic in loader._migrate_config."""

import pytest

from nanobot.config.loader import _migrate_config, load_config
from nanobot.config.schema import Config


# ---------------------------------------------------------------------------
# base_url / baseUrl → apiBase migration
# ---------------------------------------------------------------------------


def test_migrate_base_url_snake_case():
    """base_url (snake_case) in a provider config is promoted to apiBase."""
    data = {
        "providers": {
            "openai": {"apiKey": "sk-test", "base_url": "https://my.proxy/v1"}
        }
    }
    result = _migrate_config(data)
    prov = result["providers"]["openai"]
    assert "base_url" not in prov
    assert prov["apiBase"] == "https://my.proxy/v1"


def test_migrate_base_url_camel_case():
    """baseUrl (camelCase) in a provider config is promoted to apiBase."""
    data = {
        "providers": {
            "custom": {"apiKey": "sk-test", "baseUrl": "https://local.server/v1"}
        }
    }
    result = _migrate_config(data)
    prov = result["providers"]["custom"]
    assert "baseUrl" not in prov
    assert prov["apiBase"] == "https://local.server/v1"


def test_migrate_base_url_does_not_overwrite_existing_api_base():
    """If apiBase is already set, base_url is dropped without overwriting apiBase."""
    data = {
        "providers": {
            "openai": {
                "apiKey": "sk-test",
                "apiBase": "https://keep.this/v1",
                "base_url": "https://do.not.use/v1",
            }
        }
    }
    result = _migrate_config(data)
    prov = result["providers"]["openai"]
    assert prov["apiBase"] == "https://keep.this/v1"
    assert "base_url" not in prov


# ---------------------------------------------------------------------------
# Old Telegram field migration
# ---------------------------------------------------------------------------


def test_migrate_telegram_bot_token():
    """Old flat telegram.bot_token is moved to channels.telegram.token."""
    data = {
        "telegram": {"bot_token": "123:ABC", "allowed_users": ["user1", "user2"]}
    }
    result = _migrate_config(data)
    assert "telegram" not in result
    tg = result["channels"]["telegram"]
    assert tg["token"] == "123:ABC"
    assert tg["allowFrom"] == ["user1", "user2"]


def test_migrate_telegram_does_not_overwrite_existing_channels():
    """If channels.telegram.token already exists, the old value is not overwritten."""
    data = {
        "telegram": {"bot_token": "OLD_TOKEN"},
        "channels": {"telegram": {"token": "EXISTING_TOKEN"}},
    }
    result = _migrate_config(data)
    # Old telegram key is removed from root
    assert "telegram" not in result
    # The existing channels.telegram.token is preserved
    assert result["channels"]["telegram"]["token"] == "EXISTING_TOKEN"


# ---------------------------------------------------------------------------
# Old default_provider migration
# ---------------------------------------------------------------------------


def test_migrate_default_provider():
    """Top-level default_provider is moved to agents.defaults.provider."""
    data = {"default_provider": "openrouter"}
    result = _migrate_config(data)
    assert "default_provider" not in result
    assert result["agents"]["defaults"]["provider"] == "openrouter"


def test_migrate_default_provider_does_not_overwrite():
    """If agents.defaults.provider is already set, default_provider is dropped."""
    data = {
        "default_provider": "openrouter",
        "agents": {"defaults": {"provider": "anthropic"}},
    }
    result = _migrate_config(data)
    assert "default_provider" not in result
    assert result["agents"]["defaults"]["provider"] == "anthropic"


# ---------------------------------------------------------------------------
# Round-trip: migrated config validates against schema
# ---------------------------------------------------------------------------


def test_migrated_old_telegram_loads_into_schema(tmp_path):
    """Full round-trip: old telegram format survives migration → schema validation."""
    import json

    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(
        json.dumps(
            {
                "telegram": {
                    "bot_token": "999:XYZ",
                    "allowed_users": ["42"],
                },
                "providers": {
                    "openai": {"apiKey": "sk-test", "base_url": "https://proxy/v1"}
                },
                "agents": {"defaults": {"model": "gpt-4o"}},
            }
        )
    )

    config: Config = load_config(cfg_file)
    assert config.channels.telegram.token == "999:XYZ"
    assert config.channels.telegram.allow_from == ["42"]
    assert config.providers.openai.api_base == "https://proxy/v1"
    assert config.agents.defaults.model == "gpt-4o"


def test_migrated_default_provider_loads_into_schema(tmp_path):
    """Full round-trip: old default_provider survives migration → schema validation."""
    import json

    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(
        json.dumps(
            {
                "default_provider": "openrouter",
                "providers": {"openrouter": {"apiKey": "sk-or-v1-test"}},
                "agents": {"defaults": {"model": "anthropic/claude-opus-4-5"}},
            }
        )
    )

    config: Config = load_config(cfg_file)
    assert config.agents.defaults.provider == "openrouter"


# ---------------------------------------------------------------------------
# candidateModels default
# ---------------------------------------------------------------------------


def test_default_config_has_non_empty_candidate_models():
    """Fresh Config() must include example candidate models, not an empty list."""
    config = Config()
    assert config.agents.defaults.candidate_models, (
        "AgentDefaults.candidate_models must not be empty — "
        "onboard-generated configs should surface the /model hot-switching feature"
    )
    assert "anthropic/claude-opus-4-5" in config.agents.defaults.candidate_models


def test_candidate_models_preserved_when_user_sets_empty_list(tmp_path):
    """If a user explicitly sets candidateModels: [] in their config, that choice is preserved."""
    import json

    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps({"agents": {"defaults": {"candidateModels": []}}}))

    config: Config = load_config(cfg_file)
    assert config.agents.defaults.candidate_models == []
