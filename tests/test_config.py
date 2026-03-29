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


def test_agent_provider_explicit_ollama(tmp_path):
    """provider: 'ollama' on an agent is parsed and stored correctly from YAML."""
    yaml_content = textwrap.dedent("""\
        metadata:
          name: "Ollama Test"
          version: "1.0"
        debate:
          topic: "Test topic"
          max_rounds: 1
        agents:
          - name: "Agent-Local"
            persona: "EMPIRICIST"
            model: "deepseek-r1:14b"
            temperature: 0.7
            provider: "ollama"
        adjudication:
          model: "deepseek-r1:14b"
          provider: "ollama"
          logic_weight: 1.0
          ethics_weight: 0.0
          logic_system: "Classical logic"
          ethics_system: "None"
    """)
    config_file = tmp_path / "ollama_inline_test.yaml"
    config_file.write_text(yaml_content)

    from chal.config import DebateConfig
    config = DebateConfig.from_yaml(config_file)

    assert config.agents[0].provider == "ollama"
    assert config.agents[0].model == "deepseek-r1:14b"
    assert config.adjudication.provider == "ollama"


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


# ==============================================
# Config Serialization Tests (to_dict / to_yaml)
# ==============================================

def test_to_dict_returns_dict():
    """to_dict() returns a dict with expected top-level keys."""
    config = load_config('default')
    d = config.to_dict()

    assert isinstance(d, dict)
    expected_keys = {
        "metadata", "debate", "agents", "adjudication",
        "stages", "outputs", "scribe", "collaborative",
        "bloodsport", "moderator", "parallel",
    }
    assert set(d.keys()) == expected_keys


def test_to_dict_storage_dir_is_string():
    """storage_dir in to_dict() output is a string, not a Path."""
    config = load_config('default')
    d = config.to_dict()

    assert isinstance(d["outputs"]["storage_dir"], str)
    assert "\\" not in d["outputs"]["storage_dir"]  # forward slashes only


def test_to_dict_round_trip():
    """Load YAML -> to_dict -> write YAML -> reload -> compare."""
    from chal.config import DebateConfig

    original = load_config('default')
    d = original.to_dict()

    import tempfile, yaml
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8') as f:
        yaml.dump(d, f, default_flow_style=False, sort_keys=False)
        tmp_path = Path(f.name)

    try:
        reloaded = DebateConfig.from_yaml(tmp_path)

        assert reloaded.name == original.name
        assert reloaded.topic == original.topic
        assert reloaded.max_rounds == original.max_rounds
        assert reloaded.stage2_mode == original.stage2_mode
        assert reloaded.stage3_mode == original.stage3_mode
        assert len(reloaded.agents) == len(original.agents)
        for a_orig, a_new in zip(original.agents, reloaded.agents):
            assert a_orig.name == a_new.name
            assert a_orig.persona == a_new.persona
            assert a_orig.model == a_new.model
            assert a_orig.provider == a_new.provider
        assert reloaded.adjudication.model == original.adjudication.model
        assert reloaded.adjudication.logic_weight == original.adjudication.logic_weight
        assert reloaded.scribe.enabled == original.scribe.enabled
    finally:
        tmp_path.unlink()


def test_to_yaml_writes_loadable_file(tmp_path):
    """to_yaml() writes a file that from_yaml() can load."""
    from chal.config import DebateConfig, AgentConfig, OutputConfig

    config = DebateConfig(
        name="Wizard Test",
        topic="Is consciousness real?",
        max_rounds=2,
        stage2_mode="moderated",
        stage3_mode="rebuttal",
        agents=[
            AgentConfig(name="A1", persona="EMPIRICIST", model="gpt-4o", provider="openai"),
            AgentConfig(name="A2", persona="SKEPTIC", model="o1-mini", provider="openai"),
        ],
        outputs=OutputConfig(storage_dir=tmp_path),
    )

    yaml_path = tmp_path / "test_output.yaml"
    config.to_yaml(yaml_path)

    assert yaml_path.exists()

    reloaded = DebateConfig.from_yaml(yaml_path)
    assert reloaded.name == "Wizard Test"
    assert reloaded.topic == "Is consciousness real?"
    assert reloaded.max_rounds == 2
    assert reloaded.stage2_mode == "moderated"
    assert len(reloaded.agents) == 2
    assert reloaded.agents[0].persona == "EMPIRICIST"
    assert reloaded.agents[1].persona == "SKEPTIC"


# ==============================================
# ParallelConfig Tests (Phase 3E)
# ==============================================

def test_parallel_config_defaults():
    """Default ParallelConfig has enabled=False, max_workers=5."""
    from chal.config import ParallelConfig

    pc = ParallelConfig()
    assert pc.enabled is False
    assert pc.max_workers == 5


def test_parallel_config_in_debate_config():
    """DebateConfig includes ParallelConfig from default.yaml."""
    config = load_config('default')
    assert hasattr(config, "parallel")
    assert config.parallel.enabled is True
    assert config.parallel.max_workers == 5


def test_parallel_config_from_yaml(tmp_path):
    """YAML with parallel: section parses correctly."""
    yaml_content = textwrap.dedent("""\
        metadata:
          name: "Parallel Test"
          version: "1.0"
        debate:
          topic: "Test topic"
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
        parallel:
          enabled: true
          max_workers: 8
    """)
    config_file = tmp_path / "parallel_test.yaml"
    config_file.write_text(yaml_content)

    from chal.config import DebateConfig
    config = DebateConfig.from_yaml(config_file)

    assert config.parallel.enabled is True
    assert config.parallel.max_workers == 8


def test_parallel_config_missing_section(tmp_path):
    """Missing parallel: section in YAML → defaults (enabled=False)."""
    yaml_content = textwrap.dedent("""\
        metadata:
          name: "No Parallel Section"
          version: "1.0"
        debate:
          topic: "Test topic"
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
    config_file = tmp_path / "no_parallel.yaml"
    config_file.write_text(yaml_content)

    from chal.config import DebateConfig
    config = DebateConfig.from_yaml(config_file)

    assert config.parallel.enabled is False
    assert config.parallel.max_workers == 5


def test_parallel_config_to_yaml_roundtrip(tmp_path):
    """to_dict() → from_yaml() preserves parallel settings."""
    from chal.config import DebateConfig, AgentConfig, OutputConfig, ParallelConfig

    config = DebateConfig(
        name="Roundtrip Test",
        topic="Test topic",
        max_rounds=1,
        agents=[
            AgentConfig(name="A1", persona="EMPIRICIST", model="gpt-4o"),
        ],
        outputs=OutputConfig(storage_dir=tmp_path),
        parallel=ParallelConfig(enabled=True, max_workers=6),
    )

    yaml_path = tmp_path / "roundtrip.yaml"
    config.to_yaml(yaml_path)

    reloaded = DebateConfig.from_yaml(yaml_path)
    assert reloaded.parallel.enabled is True
    assert reloaded.parallel.max_workers == 6


def test_to_dict_includes_parallel():
    """to_dict() includes parallel section."""
    config = load_config('default')
    d = config.to_dict()

    assert "parallel" in d
    assert "enabled" in d["parallel"]
    assert "max_workers" in d["parallel"]


if __name__ == "__main__":
    try:
        test_default_config()
        test_storage_dir_creation()
        print("\n" + "=" * 60)
        print("All configuration tests passed!")
        print("=" * 60)
    except AssertionError as e:
        print(f"\nTest failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
