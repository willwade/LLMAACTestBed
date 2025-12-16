"""
Configuration Management

Utilities for loading and managing configuration.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ExperimentConfig:
    """
    Configuration for experiments.
    """

    # Data configuration
    data_path: str = ""
    profile_path: str = ""
    corpus_ratio: float = 0.67
    sample_size: int | None = None

    # Model configuration
    provider: str = "gemini"
    model_name: str = "gemini-2.0-flash-exp"
    temperature: float = 0.2
    judge_model: str = "gemini-2.0-flash-exp"

    # Context configuration
    context_filters: list = field(default_factory=lambda: ["none", "time", "geo", "time_geo"])
    conversation_windows: list = field(default_factory=lambda: [0, 1, 3, 5])
    time_window_hours: float = 2.0
    geo_radius_km: float = 0.5

    # Evaluation configuration
    metrics: list = field(
        default_factory=lambda: [
            "embedding_similarity",
            "llm_judge_score",
            "character_accuracy",
            "word_accuracy",
        ]
    )
    partial_methods: list = field(default_factory=lambda: ["prefix_3", "keyword_2", "random"])
    generation_methods: list = field(default_factory=lambda: ["lexical", "tfidf", "embedding"])

    # Output configuration
    output_dir: str = "results"
    save_plots: bool = True
    plot_formats: list = field(default_factory=lambda: ["png"])
    plot_dpi: int = 300

    # Advanced options
    cache_embeddings: bool = True
    verbose: bool = True

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "ExperimentConfig":
        """Create from dictionary."""
        return cls(**config_dict)

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "ExperimentConfig":
        """Load from YAML file."""
        with open(yaml_path) as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "data_path": self.data_path,
            "profile_path": self.profile_path,
            "corpus_ratio": self.corpus_ratio,
            "sample_size": self.sample_size,
            "provider": self.provider,
            "model_name": self.model_name,
            "temperature": self.temperature,
            "judge_model": self.judge_model,
            "context_filters": self.context_filters,
            "conversation_windows": self.conversation_windows,
            "time_window_hours": self.time_window_hours,
            "geo_radius_km": self.geo_radius_km,
            "metrics": self.metrics,
            "partial_methods": self.partial_methods,
            "generation_methods": self.generation_methods,
            "output_dir": self.output_dir,
            "save_plots": self.save_plots,
            "plot_formats": self.plot_formats,
            "plot_dpi": self.plot_dpi,
            "cache_embeddings": self.cache_embeddings,
            "verbose": self.verbose,
        }

    def save(self, path: str | Path):
        """Save configuration to file."""
        path = Path(path)
        config_dict = self.to_dict()

        if path.suffix.lower() == ".json":
            with open(path, "w") as f:
                json.dump(config_dict, f, indent=2)
        elif path.suffix.lower() in [".yaml", ".yml"]:
            with open(path, "w") as f:
                yaml.safe_dump(config_dict, f, default_flow_style=False, indent=2)
        else:
            raise ValueError(f"Unsupported config file format: {path.suffix}")


class Config:
    """
    Global configuration manager.
    """

    def __init__(self, config_path: str | Path | None = None):
        """
        Initialize configuration manager.

        Args:
            config_path: Path to configuration file
        """
        self._config = {}
        self.config_path = config_path

        if config_path:
            self.load(config_path)

    def load(self, config_path: str | Path):
        """
        Load configuration from file.

        Args:
            config_path: Path to configuration file
        """
        path = Path(config_path)

        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        if path.suffix.lower() == ".json":
            with open(path) as f:
                self._config = json.load(f)
        elif path.suffix.lower() in [".yaml", ".yml"]:
            with open(path) as f:
                self._config = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")

        # Apply environment variable overrides
        self._apply_env_overrides()

    def _apply_env_overrides(self):
        """Apply environment variable overrides."""
        env_mappings = {
            "LLM_GEMINI_KEY": "gemini_api_key",
            "DATA_PATH": "data_path",
            "MODEL_NAME": "model_name",
            "OUTPUT_DIR": "output_dir",
        }

        for env_var, config_key in env_mappings.items():
            value = os.getenv(env_var)
            if value:
                self.set(config_key, value)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.

        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split(".")
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any):
        """
        Set configuration value.

        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split(".")
        config = self._config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def update(self, updates: dict[str, Any]):
        """
        Update multiple configuration values.

        Args:
            updates: Dictionary of updates
        """
        for key, value in updates.items():
            self.set(key, value)

    def save(self, path: str | Path | None = None):
        """
        Save configuration to file.

        Args:
            path: Path to save (uses loaded path if not provided)
        """
        if path is None:
            path = self.config_path

        if path is None:
            raise ValueError("No path specified for saving configuration")

        path = Path(path)

        if path.suffix.lower() == ".json":
            with open(path, "w") as f:
                json.dump(self._config, f, indent=2)
        elif path.suffix.lower() in [".yaml", ".yml"]:
            with open(path, "w") as f:
                yaml.safe_dump(self._config, f, default_flow_style=False, indent=2)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")

    def to_experiment_config(self) -> ExperimentConfig:
        """
        Convert to ExperimentConfig.

        Returns:
            ExperimentConfig instance
        """
        return ExperimentConfig.from_dict(self._config)


def load_config(config_path: str | Path) -> Config:
    """
    Load configuration from file.

    Args:
        config_path: Path to configuration file

    Returns:
        Config instance
    """
    return Config(config_path)
