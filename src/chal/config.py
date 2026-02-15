"""
config.py

Central configuration management for CHAL debates.
Handles YAML loading, path resolution, and runtime settings.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import yaml
from dataclasses import dataclass, field

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent  # CHAL/
CHAL_PACKAGE_DIR = Path(__file__).parent  # src/chal/
CONFIG_DIR = CHAL_PACKAGE_DIR / "configurations"
DEFAULT_STORAGE_DIR = PROJECT_ROOT / "src" / "chal" / "storage"


@dataclass
class AgentConfig:
    """Configuration for a single debate agent."""
    name: str
    persona: str
    model: str = "gpt-4o"
    temperature: float = 0.7


@dataclass
class AdjudicationConfig:
    """Configuration for the adjudicator agent."""
    model: str = "gpt-4o"
    logic_weight: float = 1.0
    ethics_weight: float = 0.0
    logic_system: str = ""
    ethics_system: str = ""


@dataclass
class StageConfig:
    """Configuration for debate stage parameters."""
    # Stage 2: Cross-Examination
    max_questions_per_cross_exam: int = 5
    max_question_length_chars: int = 500

    # Stage 3: Rebuttals
    max_rebuttals_per_response: int = 5
    max_rebuttal_length_chars: int = 500

    # Generation settings
    generation_temperature: float = 0.2

    # Text length limits
    short_note_max_chars: int = 140


@dataclass
class OutputConfig:
    """Configuration for debate outputs and file storage."""
    storage_dir: Path

    # Text outputs
    save_synthesis: bool = True
    synthesis_file: str = "debate_synthesis.txt"

    save_transcript: bool = True
    transcript_file: str = "debate_transcript.txt"

    save_initial_beliefs: bool = True
    initial_beliefs_file: str = "initial_beliefs.txt"

    save_final_beliefs: bool = True
    final_beliefs_file: str = "final_beliefs.txt"

    # Analysis outputs
    generate_embeddings: bool = True
    embeddings_file: str = "embeddings.npz"

    plot_trajectories: bool = True
    trajectory_plot_file: str = "belief_trajectories.png"

    save_agent_stats: bool = True
    stats_file: str = "agent_stats.json"

    # Graph visualization
    generate_graph_visualization: bool = True
    graph_file: str = "belief_graph.html"

    # Logging outputs
    save_debug_log: bool = True
    debug_log_file: str = "log.txt"

    def ensure_storage_dir(self):
        """Create storage directory if it doesn't exist."""
        self.storage_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class ScribeConfig:
    """Configuration for the debate scribe/narrator."""
    enabled: bool = True
    model: str = "gpt-4o"
    max_chars_per_chunk: int = 15000
    overlap_chars: int = 1000
    scribe_temperature: float = 0.3
    style_hint: str = "formal, expository, research-paper tone"


@dataclass
class DebateConfig:
    """Main configuration container for a CHAL debate."""

    # Metadata
    name: str = "Unnamed Debate"
    description: str = ""
    version: str = "1.0"

    # Core settings
    topic: str = ""
    max_rounds: int = 1

    # Component configs
    agents: List[AgentConfig] = field(default_factory=list)
    adjudication: AdjudicationConfig = field(default_factory=AdjudicationConfig)
    stages: StageConfig = field(default_factory=StageConfig)
    outputs: OutputConfig = field(default_factory=lambda: OutputConfig(storage_dir=DEFAULT_STORAGE_DIR))
    scribe: ScribeConfig = field(default_factory=ScribeConfig)

    @classmethod
    def from_yaml(cls, config_path: Path) -> 'DebateConfig':
        """Load configuration from YAML file."""
        with open(config_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        # Parse metadata
        meta = data.get('metadata', {})

        # Parse agents
        agents = [
            AgentConfig(
                name=a['name'],
                persona=a['persona'],
                model=a.get('model', 'gpt-4o'),
                temperature=a.get('temperature', 0.7)
            )
            for a in data.get('agents', [])
        ]

        # Parse adjudication
        adj_data = data.get('adjudication', {})
        adjudication = AdjudicationConfig(
            model=adj_data.get('model', 'gpt-4o'),
            logic_weight=adj_data.get('logic_weight', 1.0),
            ethics_weight=adj_data.get('ethics_weight', 0.0),
            logic_system=adj_data.get('logic_system', ''),
            ethics_system=adj_data.get('ethics_system', '')
        )

        # Parse stages
        stage_data = data.get('stages', {})
        stages = StageConfig(
            max_questions_per_cross_exam=stage_data.get('max_questions_per_cross_exam', 5),
            max_question_length_chars=stage_data.get('max_question_length_chars', 500),
            max_rebuttals_per_response=stage_data.get('max_rebuttals_per_response', 5),
            max_rebuttal_length_chars=stage_data.get('max_rebuttal_length_chars', 500),
            generation_temperature=stage_data.get('generation_temperature', 0.2),
            short_note_max_chars=stage_data.get('short_note_max_chars', 140)
        )

        # Parse outputs
        out_data = data.get('outputs', {})
        storage_path = PROJECT_ROOT / out_data.get('storage_dir', 'src/chal/storage')
        outputs = OutputConfig(
            storage_dir=storage_path,
            save_synthesis=out_data.get('save_synthesis', True),
            synthesis_file=out_data.get('synthesis_file', 'debate_synthesis.txt'),
            save_transcript=out_data.get('save_transcript', True),
            transcript_file=out_data.get('transcript_file', 'debate_transcript.txt'),
            save_initial_beliefs=out_data.get('save_initial_beliefs', True),
            initial_beliefs_file=out_data.get('initial_beliefs_file', 'initial_beliefs.txt'),
            save_final_beliefs=out_data.get('save_final_beliefs', True),
            final_beliefs_file=out_data.get('final_beliefs_file', 'final_beliefs.txt'),
            generate_embeddings=out_data.get('generate_embeddings', True),
            embeddings_file=out_data.get('embeddings_file', 'embeddings.npz'),
            plot_trajectories=out_data.get('plot_trajectories', True),
            trajectory_plot_file=out_data.get('trajectory_plot_file', 'belief_trajectories.png'),
            save_agent_stats=out_data.get('save_agent_stats', True),
            stats_file=out_data.get('stats_file', 'agent_stats.json'),
            generate_graph_visualization=out_data.get('generate_graph_visualization', True),
            graph_file=out_data.get('graph_file', 'belief_graph.html'),
            save_debug_log=out_data.get('save_debug_log', True),
            debug_log_file=out_data.get('debug_log_file', 'log.txt')
        )

        # Parse scribe
        scribe_data = data.get('scribe', {})
        scribe = ScribeConfig(
            enabled=scribe_data.get('enabled', True),
            model=scribe_data.get('model', 'gpt-4o'),
            max_chars_per_chunk=scribe_data.get('max_chars_per_chunk', 15000),
            overlap_chars=scribe_data.get('overlap_chars', 1000),
            scribe_temperature=scribe_data.get('scribe_temperature', 0.3),
            style_hint=scribe_data.get('style_hint', 'formal, expository, research-paper tone')
        )

        # Parse debate settings
        debate_data = data.get('debate', {})

        return cls(
            name=meta.get('name', 'Unnamed Debate'),
            description=meta.get('description', ''),
            version=meta.get('version', '1.0'),
            topic=debate_data.get('topic', ''),
            max_rounds=debate_data.get('max_rounds', 1),
            agents=agents,
            adjudication=adjudication,
            stages=stages,
            outputs=outputs,
            scribe=scribe
        )

    @classmethod
    def from_name(cls, config_name: str) -> 'DebateConfig':
        """Load config by name from configurations/ directory."""
        config_path = CONFIG_DIR / f"{config_name}.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration '{config_name}' not found at {config_path}")
        return cls.from_yaml(config_path)


def load_config(config_path_or_name: str) -> DebateConfig:
    """
    Load a debate configuration.

    Args:
        config_path_or_name: Either a path to a YAML file or a config name (e.g., 'default')

    Returns:
        DebateConfig instance
    """
    path = Path(config_path_or_name)

    # If it's an existing file path, load it directly
    if path.exists() and path.suffix in ['.yaml', '.yml']:
        return DebateConfig.from_yaml(path)

    # Otherwise, treat it as a config name
    return DebateConfig.from_name(config_path_or_name)
