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
    provider: str = "openai"  # "openai" | "anthropic" | "google" | "ollama" | "xai" | "perplexity"
    belief_file: Optional[str] = None  # Path to a pre-defined CBS belief JSON file (skips Stage 1)


_VALID_WEIGHT_COMBOS = {(1.0, 0.0), (0.5, 0.5), (0.0, 1.0)}


@dataclass
class AdjudicationConfig:
    """Configuration for the adjudicator agent."""
    model: str = "o4-mini"
    logic_weight: float = 1.0
    ethics_weight: float = 0.0
    logic_system: str = "CLASSICAL_INFORMAL_BAYESIAN"
    ethics_system: str = "NONE"
    provider: str = "openai"  # "openai" | "anthropic" | "google" | "ollama" | "xai" | "perplexity"
    threshold: float = 0.15

    def __post_init__(self):
        combo = (self.logic_weight, self.ethics_weight)
        if combo not in _VALID_WEIGHT_COMBOS:
            raise ValueError(
                f"Invalid weight combination (logic={self.logic_weight}, "
                f"ethics={self.ethics_weight}). Must be one of: "
                f"(1.0, 0.0), (0.5, 0.5), or (0.0, 1.0)."
            )


@dataclass
class StageConfig:
    """Configuration for debate stage parameters."""
    # Stage 2: Cross-Examination
    max_questions_per_cross_exam: int = 5

    # Stage 3: Rebuttals
    max_rebuttals_per_response: int = 5
    max_rebuttal_length_chars: int = 500

    # Generation settings
    generation_temperature: float = 0.2

    # Text length limits
    short_note_max_chars: int = 140

    # Retry settings
    parse_retries: int = 3  # Max retries when LLM output fails to parse


@dataclass
class OutputConfig:
    """Configuration for debate outputs and file storage."""
    storage_dir: Path

    # Text outputs
    save_transcript: bool = True
    transcript_file: str = "debate_transcript.txt"

    save_initial_beliefs: bool = True
    initial_beliefs_dir: str = "initial_beliefs"

    save_final_beliefs: bool = True
    final_beliefs_dir: str = "final_beliefs"

    # Best-agent belief outputs (highest-performance_score agent only)
    best_beliefs_json_file: str = "best_initial_final_beliefs.json"
    best_beliefs_text_file: str = "best_initial_final_beliefs.txt"

    # Analysis outputs
    generate_embeddings: bool = True
    embeddings_file: str = "embeddings.npz"

    plot_trajectories: bool = True
    trajectory_plot_file: str = "belief_trajectories.png"
    pca_plot_file: str = "belief_trajectories_pca.png"

    save_agent_stats: bool = True
    stats_file: str = "agent_stats.json"

    # Graph visualization
    generate_graph_visualization: bool = True
    graph_file: str = "belief_graph.html"

    # Logging outputs
    save_debug_log: bool = True
    debug_log_file: str = "log.txt"

    # Training data & analysis (mode-agnostic)
    save_analysis_report: bool = False
    analysis_report_file: str = "debate_analysis_report.md"
    save_training_data: bool = False
    training_data_file: str = "debate_training_data.jsonl"
    belief_pairs_file: str = "debate_belief_pairs.jsonl"

    def ensure_storage_dir(self):
        """Create storage directory if it doesn't exist."""
        self.storage_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class ParallelConfig:
    """Configuration for parallel API call dispatch."""
    enabled: bool = False               # Master switch — off by default
    max_workers: int = 5                # ThreadPoolExecutor max_workers


@dataclass
class DefenseBoostConfig:
    """Configuration for the mechanical defense boost system.

    When a node survives a challenge (REBUTTAL_VALID verdict), the system
    automatically applies a flat strength increase.

    Boost per defense: flat_boost (constant, regardless of streak length)
    Ceiling: min(current + flat_boost, original_strength + max_cumulative_boost, 1.0)
    """
    enabled: bool = True
    flat_boost: float = 0.02               # Constant boost per successful defense
    max_cumulative_boost: float = 0.15     # Max total boost above original_strength


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
    stage3_mode: str = "rebuttal"  # Only supported mode: "rebuttal"

    # Component configs
    agents: List[AgentConfig] = field(default_factory=list)
    adjudication: AdjudicationConfig = field(default_factory=AdjudicationConfig)
    stages: StageConfig = field(default_factory=StageConfig)
    outputs: OutputConfig = field(default_factory=lambda: OutputConfig(storage_dir=DEFAULT_STORAGE_DIR))
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    defense_boost: DefenseBoostConfig = field(default_factory=DefenseBoostConfig)

    @classmethod
    def from_yaml(cls, config_path: Path) -> 'DebateConfig':
        """Load configuration from YAML file."""
        with open(config_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        # Parse metadata
        meta = data.get('metadata', {})

        # Parse agents
        config_dir = Path(config_path).parent
        agents = []
        for a in data.get('agents', []):
            # Resolve belief_file path relative to the YAML file's directory
            belief_file_raw = a.get('belief_file')
            belief_file: Optional[str] = None
            if belief_file_raw:
                bf_path = Path(belief_file_raw)
                if not bf_path.is_absolute():
                    bf_path = config_dir / bf_path
                belief_file = str(bf_path)
            agents.append(AgentConfig(
                name=a['name'],
                persona=a['persona'],
                model=a.get('model', 'gpt-4o'),
                temperature=a.get('temperature', 0.7),
                provider=a.get('provider', 'openai'),
                belief_file=belief_file,
            ))

        # Parse adjudication
        adj_data = data.get('adjudication', {})
        adjudication = AdjudicationConfig(
            model=adj_data.get('model', 'gpt-4o'),
            logic_weight=adj_data.get('logic_weight', 1.0),
            ethics_weight=adj_data.get('ethics_weight', 0.0),
            logic_system=adj_data.get('logic_system', 'CLASSICAL_INFORMAL_BAYESIAN'),
            ethics_system=adj_data.get('ethics_system', 'NONE'),
            provider=adj_data.get('provider', 'openai')
        )

        # Parse stages
        stage_data = data.get('stages', {})
        stages = StageConfig(
            max_questions_per_cross_exam=stage_data.get('max_questions_per_cross_exam', 5),
            max_rebuttals_per_response=stage_data.get('max_rebuttals_per_response', 5),
            max_rebuttal_length_chars=stage_data.get('max_rebuttal_length_chars', 500),
            generation_temperature=stage_data.get('generation_temperature', 0.2),
            short_note_max_chars=stage_data.get('short_note_max_chars', 140),
            parse_retries=stage_data.get('parse_retries', 3),
        )

        # Parse outputs
        out_data = data.get('outputs', {})
        storage_path = PROJECT_ROOT / out_data.get('storage_dir', 'src/chal/storage')
        outputs = OutputConfig(
            storage_dir=storage_path,
            save_transcript=out_data.get('save_transcript', True),
            transcript_file=out_data.get('transcript_file', 'debate_transcript.txt'),
            save_initial_beliefs=out_data.get('save_initial_beliefs', True),
            initial_beliefs_dir=out_data.get('initial_beliefs_dir', 'initial_beliefs'),
            save_final_beliefs=out_data.get('save_final_beliefs', True),
            final_beliefs_dir=out_data.get('final_beliefs_dir', 'final_beliefs'),
            best_beliefs_json_file=out_data.get('best_beliefs_json_file', 'best_initial_final_beliefs.json'),
            best_beliefs_text_file=out_data.get('best_beliefs_text_file', 'best_initial_final_beliefs.txt'),
            generate_embeddings=out_data.get('generate_embeddings', True),
            embeddings_file=out_data.get('embeddings_file', 'embeddings.npz'),
            plot_trajectories=out_data.get('plot_trajectories', True),
            trajectory_plot_file=out_data.get('trajectory_plot_file', 'belief_trajectories.png'),
            pca_plot_file=out_data.get('pca_plot_file', 'belief_trajectories_pca.png'),
            save_agent_stats=out_data.get('save_agent_stats', True),
            stats_file=out_data.get('stats_file', 'agent_stats.json'),
            generate_graph_visualization=out_data.get('generate_graph_visualization', True),
            graph_file=out_data.get('graph_file', 'belief_graph.html'),
            save_debug_log=out_data.get('save_debug_log', True),
            debug_log_file=out_data.get('debug_log_file', 'log.txt'),
            save_analysis_report=out_data.get('save_analysis_report', False),
            analysis_report_file=out_data.get('analysis_report_file', 'debate_analysis_report.md'),
            save_training_data=out_data.get('save_training_data', False),
            training_data_file=out_data.get('training_data_file', 'debate_training_data.jsonl'),
            belief_pairs_file=out_data.get('belief_pairs_file', 'debate_belief_pairs.jsonl')
        )

        # Parse parallel config
        par_data = data.get('parallel', {})
        parallel = ParallelConfig(
            enabled=par_data.get('enabled', False),
            max_workers=par_data.get('max_workers', 5),
        )

        # Parse defense boost config
        db_data = data.get('defense_boost', {})
        defense_boost = DefenseBoostConfig(
            enabled=db_data.get('enabled', True),
            flat_boost=db_data.get('flat_boost', 0.02),
            max_cumulative_boost=db_data.get('max_cumulative_boost', 0.15),
        )

        # Parse debate settings
        debate_data = data.get('debate', {})

        return cls(
            name=meta.get('name', 'Unnamed Debate'),
            description=meta.get('description', ''),
            version=meta.get('version', '1.0'),
            topic=debate_data.get('topic', ''),
            max_rounds=debate_data.get('max_rounds', 1),
            stage3_mode=debate_data.get('stage3_mode', 'rebuttal'),
            agents=agents,
            adjudication=adjudication,
            stages=stages,
            outputs=outputs,
            parallel=parallel,
            defense_boost=defense_boost,
        )

    @classmethod
    def from_name(cls, config_name: str) -> 'DebateConfig':
        """Load config by name from configurations/ directory."""
        config_path = CONFIG_DIR / f"{config_name}.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration '{config_name}' not found at {config_path}")
        return cls.from_yaml(config_path)

    def to_dict(self) -> dict:
        """Convert config to a dict matching the YAML schema that from_yaml() reads."""
        # Convert storage_dir Path to relative string with forward slashes
        try:
            storage_str = str(self.outputs.storage_dir.relative_to(PROJECT_ROOT))
        except ValueError:
            storage_str = str(self.outputs.storage_dir)
        storage_str = storage_str.replace("\\", "/")

        return {
            "metadata": {
                "name": self.name,
                "description": self.description,
                "version": self.version,
            },
            "debate": {
                "topic": self.topic,
                "max_rounds": self.max_rounds,
                "stage3_mode": self.stage3_mode,
            },
            "agents": [
                {
                    k: v for k, v in {
                        "name": a.name,
                        "persona": a.persona,
                        "model": a.model,
                        "temperature": a.temperature,
                        "provider": a.provider,
                        "belief_file": a.belief_file,
                    }.items() if v is not None
                }
                for a in self.agents
            ],
            "adjudication": {
                "model": self.adjudication.model,
                "provider": self.adjudication.provider,
                "logic_weight": self.adjudication.logic_weight,
                "ethics_weight": self.adjudication.ethics_weight,
                "logic_system": self.adjudication.logic_system,
                "ethics_system": self.adjudication.ethics_system,
            },
            "stages": {
                "max_questions_per_cross_exam": self.stages.max_questions_per_cross_exam,
                "max_rebuttals_per_response": self.stages.max_rebuttals_per_response,
                "max_rebuttal_length_chars": self.stages.max_rebuttal_length_chars,
                "generation_temperature": self.stages.generation_temperature,
                "short_note_max_chars": self.stages.short_note_max_chars,
                "parse_retries": self.stages.parse_retries,
            },
            "outputs": {
                "storage_dir": storage_str,
                "save_transcript": self.outputs.save_transcript,
                "transcript_file": self.outputs.transcript_file,
                "save_initial_beliefs": self.outputs.save_initial_beliefs,
                "initial_beliefs_dir": self.outputs.initial_beliefs_dir,
                "save_final_beliefs": self.outputs.save_final_beliefs,
                "final_beliefs_dir": self.outputs.final_beliefs_dir,
                "best_beliefs_json_file": self.outputs.best_beliefs_json_file,
                "best_beliefs_text_file": self.outputs.best_beliefs_text_file,
                "generate_embeddings": self.outputs.generate_embeddings,
                "embeddings_file": self.outputs.embeddings_file,
                "plot_trajectories": self.outputs.plot_trajectories,
                "trajectory_plot_file": self.outputs.trajectory_plot_file,
                "pca_plot_file": self.outputs.pca_plot_file,
                "save_agent_stats": self.outputs.save_agent_stats,
                "stats_file": self.outputs.stats_file,
                "generate_graph_visualization": self.outputs.generate_graph_visualization,
                "graph_file": self.outputs.graph_file,
                "save_debug_log": self.outputs.save_debug_log,
                "debug_log_file": self.outputs.debug_log_file,
                "save_analysis_report": self.outputs.save_analysis_report,
                "analysis_report_file": self.outputs.analysis_report_file,
                "save_training_data": self.outputs.save_training_data,
                "training_data_file": self.outputs.training_data_file,
                "belief_pairs_file": self.outputs.belief_pairs_file,
            },
            "parallel": {
                "enabled": self.parallel.enabled,
                "max_workers": self.parallel.max_workers,
            },
            "defense_boost": {
                "enabled": self.defense_boost.enabled,
                "flat_boost": self.defense_boost.flat_boost,
                "max_cumulative_boost": self.defense_boost.max_cumulative_boost,
            },
        }

    def to_yaml(self, path: Path) -> None:
        """Write configuration to a YAML file.

        Args:
            path: Destination file path.
        """
        data = self.to_dict()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


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
