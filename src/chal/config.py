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
class ScribeConfig:
    """Configuration for the debate scribe/narrator."""
    enabled: bool = True
    model: str = "gpt-4o"
    max_chars_per_chunk: int = 15000
    overlap_chars: int = 1000
    scribe_temperature: float = 0.3
    style_hint: str = "formal, expository, research-paper tone"


@dataclass
class ParallelConfig:
    """Configuration for parallel API call dispatch."""
    enabled: bool = False               # Master switch — off by default
    max_workers: int = 5                # ThreadPoolExecutor max_workers


@dataclass
class DefenseBoostConfig:
    """Configuration for the mechanical defense boost system.

    When a node survives a challenge (REBUTTAL_VALID verdict), the system
    automatically applies a formula-driven strength increase. All parameters
    are tunable here.

    Formula: boost(n) = min(base_boost + boost_increment * n, max_boost_per_defense)
    Ceiling: min(current + boost, original_strength + max_cumulative_boost, 1.0)

    Where n = consecutive successful defenses (1-indexed).
    """
    enabled: bool = True                    # Master switch for defense boosts
    base_boost: float = 0.02               # Starting constant in the boost formula
    boost_increment: float = 0.01          # Added per consecutive defense
    max_boost_per_defense: float = 0.05    # Per-defense ceiling
    max_cumulative_boost: float = 0.20     # Max total boost above original_strength


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
    scribe: ScribeConfig = field(default_factory=ScribeConfig)
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
        agents = [
            AgentConfig(
                name=a['name'],
                persona=a['persona'],
                model=a.get('model', 'gpt-4o'),
                temperature=a.get('temperature', 0.7),
                provider=a.get('provider', 'openai')
            )
            for a in data.get('agents', [])
        ]

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
            debug_log_file=out_data.get('debug_log_file', 'log.txt'),
            save_analysis_report=out_data.get('save_analysis_report', False),
            analysis_report_file=out_data.get('analysis_report_file', 'debate_analysis_report.md'),
            save_training_data=out_data.get('save_training_data', False),
            training_data_file=out_data.get('training_data_file', 'debate_training_data.jsonl'),
            belief_pairs_file=out_data.get('belief_pairs_file', 'debate_belief_pairs.jsonl')
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
            base_boost=db_data.get('base_boost', 0.02),
            boost_increment=db_data.get('boost_increment', 0.01),
            max_boost_per_defense=db_data.get('max_boost_per_defense', 0.05),
            max_cumulative_boost=db_data.get('max_cumulative_boost', 0.20),
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
            scribe=scribe,
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
                    "name": a.name,
                    "persona": a.persona,
                    "model": a.model,
                    "temperature": a.temperature,
                    "provider": a.provider,
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
                "save_synthesis": self.outputs.save_synthesis,
                "synthesis_file": self.outputs.synthesis_file,
                "save_transcript": self.outputs.save_transcript,
                "transcript_file": self.outputs.transcript_file,
                "save_initial_beliefs": self.outputs.save_initial_beliefs,
                "initial_beliefs_file": self.outputs.initial_beliefs_file,
                "save_final_beliefs": self.outputs.save_final_beliefs,
                "final_beliefs_file": self.outputs.final_beliefs_file,
                "generate_embeddings": self.outputs.generate_embeddings,
                "embeddings_file": self.outputs.embeddings_file,
                "plot_trajectories": self.outputs.plot_trajectories,
                "trajectory_plot_file": self.outputs.trajectory_plot_file,
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
            "scribe": {
                "enabled": self.scribe.enabled,
                "model": self.scribe.model,
                "max_chars_per_chunk": self.scribe.max_chars_per_chunk,
                "overlap_chars": self.scribe.overlap_chars,
                "scribe_temperature": self.scribe.scribe_temperature,
                "style_hint": self.scribe.style_hint,
            },
            "parallel": {
                "enabled": self.parallel.enabled,
                "max_workers": self.parallel.max_workers,
            },
            "defense_boost": {
                "enabled": self.defense_boost.enabled,
                "base_boost": self.defense_boost.base_boost,
                "boost_increment": self.defense_boost.boost_increment,
                "max_boost_per_defense": self.defense_boost.max_boost_per_defense,
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
