"""
Unit tests for BloodSport configuration and YAML parsing.

Tests cover:
- BloodSportConfig dataclass defaults
- Intensity validation (mild | moderate | extreme)
- YAML parsing with bloodsport section
- stage3_mode: bloodsport acceptance
- OutputConfig new fields for analysis/training data
"""

import pytest
import textwrap
from pathlib import Path
from chal.config import (
    BloodSportConfig,
    DebateConfig,
    OutputConfig,
    load_config,
)


# ============================================================
# 1. BloodSportConfig Dataclass
# ============================================================

class TestBloodSportConfigDefaults:
    """Tests for BloodSportConfig dataclass defaults."""

    @pytest.mark.unit
    def test_default_intensity(self):
        cfg = BloodSportConfig()
        assert cfg.intensity == "moderate"

    @pytest.mark.unit
    def test_default_max_exchanges(self):
        cfg = BloodSportConfig()
        assert cfg.max_exchanges == 5

    @pytest.mark.unit
    def test_custom_intensity(self):
        for level in ("mild", "moderate", "extreme"):
            cfg = BloodSportConfig(intensity=level)
            assert cfg.intensity == level

    @pytest.mark.unit
    def test_custom_max_exchanges(self):
        cfg = BloodSportConfig(max_exchanges=10)
        assert cfg.max_exchanges == 10


# ============================================================
# 2. DebateConfig BloodSport Field
# ============================================================

class TestDebateConfigBloodSportField:
    """Tests for bloodsport field on DebateConfig."""

    @pytest.mark.unit
    def test_debate_config_has_bloodsport_field(self):
        config = DebateConfig()
        assert hasattr(config, "bloodsport")
        assert isinstance(config.bloodsport, BloodSportConfig)

    @pytest.mark.unit
    def test_debate_config_bloodsport_defaults(self):
        config = DebateConfig()
        assert config.bloodsport.intensity == "moderate"
        assert config.bloodsport.max_exchanges == 5

    @pytest.mark.unit
    def test_stage3_mode_bloodsport(self):
        config = DebateConfig(stage3_mode="bloodsport")
        assert config.stage3_mode == "bloodsport"


# ============================================================
# 3. YAML Parsing
# ============================================================

class TestBloodSportYAMLParsing:
    """Tests for loading bloodsport config from YAML."""

    @pytest.mark.unit
    def test_load_bloodsport_config(self, tmp_path):
        """Inline bloodsport YAML loads with stage3_mode='bloodsport'."""
        yaml_content = textwrap.dedent("""\
            metadata:
              name: "Blood Sport Debate"
              version: "1.0"
            debate:
              topic: "Does free will exist?"
              max_rounds: 1
              stage3_mode: "bloodsport"
            agents:
              - name: "Agent-A"
                persona: "EMPIRICIST"
                model: "gpt-4o"
                temperature: 0.7
              - name: "Agent-B"
                persona: "RATIONALIST"
                model: "gpt-4o"
                temperature: 0.7
            adjudication:
              model: "gpt-4o"
              logic_weight: 1.0
              ethics_weight: 0.0
              logic_system: "Classical logic"
              ethics_system: "None"
            bloodsport:
              intensity: "moderate"
              max_exchanges: 5
        """)
        config_file = tmp_path / "bs_test.yaml"
        config_file.write_text(yaml_content)
        config = DebateConfig.from_yaml(config_file)
        assert config.name == "Blood Sport Debate"
        assert config.stage3_mode == "bloodsport"
        assert config.topic == "Does free will exist?"
        assert len(config.agents) == 2

    @pytest.mark.unit
    def test_bloodsport_section_parsed(self, tmp_path):
        """Bloodsport settings are correctly parsed from YAML."""
        yaml_content = textwrap.dedent("""\
            metadata:
              name: "BS Parsed"
              version: "1.0"
            debate:
              topic: "Test"
              max_rounds: 1
              stage3_mode: "bloodsport"
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
            bloodsport:
              intensity: "moderate"
              max_exchanges: 5
        """)
        config_file = tmp_path / "bs_parsed.yaml"
        config_file.write_text(yaml_content)
        config = DebateConfig.from_yaml(config_file)
        assert config.bloodsport.intensity == "moderate"
        assert config.bloodsport.max_exchanges == 5

    @pytest.mark.unit
    def test_output_config_training_data_fields(self, tmp_path):
        """OutputConfig new fields are parsed from inline bloodsport YAML."""
        yaml_content = textwrap.dedent("""\
            metadata:
              name: "BS Output Test"
              version: "1.0"
            debate:
              topic: "Test"
              max_rounds: 1
              stage3_mode: "bloodsport"
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
            bloodsport:
              intensity: "moderate"
              max_exchanges: 5
            outputs:
              save_analysis_report: true
              save_training_data: true
              analysis_report_file: "debate_analysis_report.md"
              training_data_file: "debate_training_data.jsonl"
              belief_pairs_file: "debate_belief_pairs.jsonl"
        """)
        config_file = tmp_path / "bs_output.yaml"
        config_file.write_text(yaml_content)
        config = DebateConfig.from_yaml(config_file)
        assert config.outputs.save_analysis_report is True
        assert config.outputs.save_training_data is True
        assert config.outputs.analysis_report_file == "debate_analysis_report.md"
        assert config.outputs.training_data_file == "debate_training_data.jsonl"
        assert config.outputs.belief_pairs_file == "debate_belief_pairs.jsonl"

    @pytest.mark.unit
    def test_custom_bloodsport_yaml(self, tmp_path):
        """Custom YAML with bloodsport settings parses correctly."""
        yaml_content = textwrap.dedent("""\
            metadata:
              name: "Custom Bloodsport"
              version: "1.0"
            debate:
              topic: "Is AI conscious?"
              max_rounds: 3
              stage3_mode: "bloodsport"
            agents:
              - name: "Agent-X"
                persona: "EMPIRICIST"
                model: "gpt-4o"
                temperature: 0.5
            adjudication:
              model: "gpt-4o"
              logic_weight: 1.0
              ethics_weight: 0.0
              logic_system: "Classical logic"
              ethics_system: "None"
            bloodsport:
              intensity: "extreme"
              max_exchanges: 8
        """)
        config_file = tmp_path / "custom_bs.yaml"
        config_file.write_text(yaml_content)

        config = DebateConfig.from_yaml(config_file)

        assert config.stage3_mode == "bloodsport"
        assert config.bloodsport.intensity == "extreme"
        assert config.bloodsport.max_exchanges == 8

    @pytest.mark.unit
    def test_yaml_without_bloodsport_section_uses_defaults(self, tmp_path):
        """YAML without bloodsport section uses BloodSportConfig defaults."""
        yaml_content = textwrap.dedent("""\
            metadata:
              name: "No BS Section"
              version: "1.0"
            debate:
              topic: "Test"
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
              logic_system: "Classical"
              ethics_system: "None"
        """)
        config_file = tmp_path / "no_bs.yaml"
        config_file.write_text(yaml_content)

        config = DebateConfig.from_yaml(config_file)
        assert config.bloodsport.intensity == "moderate"
        assert config.bloodsport.max_exchanges == 5

    @pytest.mark.unit
    def test_default_config_still_loads_correctly(self):
        """default.yaml still loads with stage3_mode='rebuttal' (no regression)."""
        config = load_config("default")
        assert config.stage3_mode == "rebuttal"

    @pytest.mark.unit
    def test_collaborative_config_still_loads_correctly(self, tmp_path):
        """Inline collaborative YAML loads with stage3_mode='collaborative' (no regression)."""
        yaml_content = textwrap.dedent("""\
            metadata:
              name: "Collaborative Test"
              version: "1.0"
            debate:
              topic: "Test"
              max_rounds: 1
              stage3_mode: "collaborative"
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
        config_file = tmp_path / "collab.yaml"
        config_file.write_text(yaml_content)
        config = DebateConfig.from_yaml(config_file)
        assert config.stage3_mode == "collaborative"


# ============================================================
# 4. OutputConfig New Fields
# ============================================================

class TestOutputConfigNewFields:
    """Tests for new OutputConfig fields (analysis report, training data)."""

    @pytest.mark.unit
    def test_defaults_disabled(self):
        """New output fields default to False/disabled."""
        config = DebateConfig()
        assert config.outputs.save_analysis_report is False
        assert config.outputs.save_training_data is False

    @pytest.mark.unit
    def test_default_file_names(self):
        """New output fields have sensible default file names."""
        config = DebateConfig()
        assert config.outputs.analysis_report_file == "debate_analysis_report.md"
        assert config.outputs.training_data_file == "debate_training_data.jsonl"
        assert config.outputs.belief_pairs_file == "debate_belief_pairs.jsonl"

    @pytest.mark.unit
    def test_custom_output_yaml(self, tmp_path):
        """Custom output fields in YAML are parsed correctly."""
        yaml_content = textwrap.dedent("""\
            metadata:
              name: "Output Test"
              version: "1.0"
            debate:
              topic: "Test"
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
              logic_system: "Classical"
              ethics_system: "None"
            outputs:
              storage_dir: "src/chal/storage"
              save_analysis_report: true
              analysis_report_file: "custom_report.md"
              save_training_data: true
              training_data_file: "custom_training.jsonl"
              belief_pairs_file: "custom_pairs.jsonl"
        """)
        config_file = tmp_path / "output_test.yaml"
        config_file.write_text(yaml_content)

        config = DebateConfig.from_yaml(config_file)

        assert config.outputs.save_analysis_report is True
        assert config.outputs.analysis_report_file == "custom_report.md"
        assert config.outputs.save_training_data is True
        assert config.outputs.training_data_file == "custom_training.jsonl"
        assert config.outputs.belief_pairs_file == "custom_pairs.jsonl"
