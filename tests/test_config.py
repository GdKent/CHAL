#!/usr/bin/env python
"""
test_config.py

Quick test script to verify configuration system works.
Run this after installing pyyaml.

Usage:
    python -m pytest tests/test_config.py
    # or
    python tests/test_config.py
"""

import sys
import textwrap
from pathlib import Path
from chal.config import load_config

def test_default_config():
    """Test loading default configuration."""
    print("Testing default configuration...")
    config = load_config('default')

    assert config.name == "Default Debate", f"Expected 'Default Debate', got '{config.name}'"
    assert config.topic == "Does free will exist?", f"Wrong topic: {config.topic}"
    assert len(config.agents) == 2, f"Expected 2 agents, got {len(config.agents)}"
    assert config.max_rounds == 1, f"Expected 1 round, got {config.max_rounds}"
    assert config.agents[0].name == "Agent-Empiricist"
    assert config.agents[1].name == "Agent-Supernaturalist"
    assert config.adjudication.logic_weight == 1.0
    assert config.outputs.save_synthesis == True
    assert config.scribe.enabled == True

    print(f"✓ Config loaded successfully: {config.name}")
    print(f"  Topic: {config.topic}")
    print(f"  Agents: {', '.join(a.name for a in config.agents)}")
    print(f"  Max rounds: {config.max_rounds}")
    print(f"  Storage: {config.outputs.storage_dir}")

def test_quick_test_config():
    """Test loading quick_test configuration."""
    print("\nTesting quick_test configuration...")
    config = load_config('quick_test')

    assert config.name == "Quick Test"
    assert len(config.agents) == 2
    assert config.outputs.save_synthesis == False  # Disabled in quick_test
    assert config.scribe.enabled == False  # Disabled in quick_test

    print(f"✓ Config loaded successfully: {config.name}")
    print(f"  Topic: {config.topic}")
    print(f"  Scribe enabled: {config.scribe.enabled}")
    print(f"  Save synthesis: {config.outputs.save_synthesis}")

def test_storage_dir_creation():
    """Test that storage directory creation works."""
    print("\nTesting storage directory creation...")
    config = load_config('default')
    config.outputs.ensure_storage_dir()

    assert config.outputs.storage_dir.exists(), "Storage directory was not created"
    print(f"✓ Storage directory exists: {config.outputs.storage_dir}")

# ==============================================
# Provider Field Tests (Phase 7 — Multi-Provider)
# ==============================================

def test_agent_provider_defaults_to_openai():
    """All agents in default.yaml report provider='openai'."""
    config = load_config('default')
    for agent in config.agents:
        assert agent.provider == "openai", (
            f"Agent '{agent.name}' expected provider='openai', got '{agent.provider}'"
        )


def test_adjudication_provider_defaults_to_openai():
    """Adjudicator in default.yaml reports provider='openai'."""
    config = load_config('default')
    assert config.adjudication.provider == "openai"


def test_quick_test_agent_providers():
    """All agents in quick_test.yaml report provider='openai'."""
    config = load_config('quick_test')
    for agent in config.agents:
        assert agent.provider == "openai"
    assert config.adjudication.provider == "openai"


def test_agent_provider_explicit_anthropic(tmp_path):
    """provider: 'anthropic' on an agent is parsed correctly from YAML."""
    yaml_content = textwrap.dedent("""\
        metadata:
          name: "Anthropic Test"
          version: "1.0"
        debate:
          topic: "Test topic"
          max_rounds: 1
        agents:
          - name: "Agent-Claude"
            persona: "EMPIRICIST"
            model: "claude-sonnet-4-6"
            temperature: 0.7
            provider: "anthropic"
        adjudication:
          model: "claude-opus-4-6"
          provider: "anthropic"
          logic_weight: 1.0
          ethics_weight: 0.0
          logic_system: "Classical logic"
          ethics_system: "None"
    """)
    config_file = tmp_path / "anthropic_test.yaml"
    config_file.write_text(yaml_content)

    from chal.config import DebateConfig
    config = DebateConfig.from_yaml(config_file)

    assert config.agents[0].provider == "anthropic"
    assert config.agents[0].model == "claude-sonnet-4-6"
    assert config.adjudication.provider == "anthropic"


def test_agent_provider_explicit_google(tmp_path):
    """provider: 'google' on an agent is parsed correctly from YAML."""
    yaml_content = textwrap.dedent("""\
        metadata:
          name: "Google Test"
          version: "1.0"
        debate:
          topic: "Test topic"
          max_rounds: 1
        agents:
          - name: "Agent-Gemini"
            persona: "RATIONALIST"
            model: "gemini-2.0-flash"
            temperature: 0.7
            provider: "google"
        adjudication:
          model: "gemini-2.0-flash"
          provider: "google"
          logic_weight: 1.0
          ethics_weight: 0.0
          logic_system: "Classical logic"
          ethics_system: "None"
    """)
    config_file = tmp_path / "google_test.yaml"
    config_file.write_text(yaml_content)

    from chal.config import DebateConfig
    config = DebateConfig.from_yaml(config_file)

    assert config.agents[0].provider == "google"
    assert config.adjudication.provider == "google"


def test_backward_compat_no_provider_field(tmp_path):
    """Legacy YAML without a provider field loads without error and defaults to 'openai'."""
    yaml_content = textwrap.dedent("""\
        metadata:
          name: "Legacy Config"
          version: "1.0"
        debate:
          topic: "Legacy topic"
          max_rounds: 1
        agents:
          - name: "Agent-A"
            persona: "EMPIRICIST"
            model: "gpt-4o"
            temperature: 0.7
        adjudication:
          model: "gpt-4o"
          logic_weight: 1.0
          ethics_weight: 0.0
          logic_system: "Classical logic"
          ethics_system: "None"
    """)
    config_file = tmp_path / "legacy.yaml"
    config_file.write_text(yaml_content)

    from chal.config import DebateConfig
    config = DebateConfig.from_yaml(config_file)

    assert config.agents[0].provider == "openai"
    assert config.adjudication.provider == "openai"


def test_mixed_provider_config(tmp_path):
    """A config with different providers per agent parses all providers correctly."""
    yaml_content = textwrap.dedent("""\
        metadata:
          name: "Mixed Provider Test"
          version: "1.0"
        debate:
          topic: "Mixed provider debate"
          max_rounds: 1
        agents:
          - name: "Agent-OpenAI"
            persona: "EMPIRICIST"
            model: "gpt-4o"
            temperature: 0.7
            provider: "openai"
          - name: "Agent-Anthropic"
            persona: "RATIONALIST"
            model: "claude-sonnet-4-6"
            temperature: 0.7
            provider: "anthropic"
          - name: "Agent-Google"
            persona: "SKEPTIC"
            model: "gemini-2.0-flash"
            temperature: 0.7
            provider: "google"
        adjudication:
          model: "o1-mini"
          provider: "openai"
          logic_weight: 1.0
          ethics_weight: 0.0
          logic_system: "Classical logic"
          ethics_system: "None"
    """)
    config_file = tmp_path / "mixed.yaml"
    config_file.write_text(yaml_content)

    from chal.config import DebateConfig
    config = DebateConfig.from_yaml(config_file)

    providers = {a.name: a.provider for a in config.agents}
    assert providers["Agent-OpenAI"] == "openai"
    assert providers["Agent-Anthropic"] == "anthropic"
    assert providers["Agent-Google"] == "google"
    assert config.adjudication.provider == "openai"


if __name__ == "__main__":
    try:
        test_default_config()
        test_quick_test_config()
        test_storage_dir_creation()
        print("\n" + "=" * 60)
        print("✅ All configuration tests passed!")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
