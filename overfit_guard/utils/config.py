"""Configuration management utilities."""

from typing import Dict, Any, Optional, List
import json
from pathlib import Path


class Config:
    """Configuration manager for overfit-guard."""

    DEFAULT_CONFIG = {
        # Monitor settings
        'auto_correct': False,
        'min_severity_for_correction': 'MODERATE',
        'correction_cooldown': 5,
        'log_level': 'INFO',

        # Detector settings
        'detectors': {
            'gap_detector': {
                'enabled': True,
                'gap_threshold_mild': 0.05,
                'gap_threshold_moderate': 0.10,
                'gap_threshold_severe': 0.20,
                'metric_name': 'loss',
                'use_relative_gap': True,
                'window_size': 1
            },
            'curve_analyzer': {
                'enabled': True,
                'lookback_window': 10,
                'metric_name': 'loss',
                'divergence_threshold': 0.05,
                'trend_threshold': 0.01,
                'min_epochs': 5
            },
            'cv_detector': {
                'enabled': False,  # Disabled by default
                'variance_threshold_mild': 0.05,
                'variance_threshold_moderate': 0.10,
                'variance_threshold_severe': 0.20,
                'metric_name': 'loss',
                'min_folds': 3
            },
            'statistical': {
                'enabled': False,  # Disabled by default
                'significance_level': 0.05,
                'metric_name': 'loss',
                'min_samples': 10,
                'test_type': 'both'
            }
        },

        # Corrector settings
        'correctors': {
            'regularization': {
                'enabled': True,
                'enable_weight_decay': True,
                'enable_dropout': True,
                'enable_early_stopping': True,
                'weight_decay_step': 0.001,
                'dropout_step': 0.1,
                'early_stop_patience': 10
            },
            'augmentation': {
                'enabled': True,
                'augmentation_strategies': ['rotation', 'flip', 'crop', 'noise'],
                'initial_strength': 0.1,
                'strength_increment': 0.1,
                'max_strength': 0.8
            },
            'architecture': {
                'enabled': False,  # Disabled by default (requires manual intervention)
                'enable_pruning': True,
                'enable_dimension_reduction': True,
                'enable_normalization': True,
                'dimension_reduction_factor': 0.8,
                'min_layer_size': 32
            },
            'hyperparameter': {
                'enabled': True,
                'enable_lr_adjustment': True,
                'enable_batch_size_adjustment': True,
                'lr_reduction_factor': 0.5,
                'batch_size_increase_factor': 2,
                'min_lr': 1e-6,
                'max_batch_size': 512
            }
        }
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize configuration.

        Args:
            config: Custom configuration dictionary
        """
        self._config = self._merge_configs(self.DEFAULT_CONFIG.copy(), config or {})

    @staticmethod
    def _merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge configuration dictionaries."""
        result = base.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = Config._merge_configs(result[key], value)
            else:
                result[key] = value

        return result

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        keys = key.split('.')
        config = self._config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self._config.copy()

    @classmethod
    def from_file(cls, filepath: str) -> 'Config':
        """Load configuration from JSON file."""
        path = Path(filepath)

        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")

        with open(path, 'r') as f:
            config_dict = json.load(f)

        return cls(config_dict)

    def save(self, filepath: str) -> None:
        """Save configuration to JSON file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            json.dump(self._config, f, indent=2)

    def get_detector_config(self, detector_name: str) -> Dict[str, Any]:
        """Get configuration for a specific detector."""
        return self.get(f'detectors.{detector_name}', {})

    def get_corrector_config(self, corrector_name: str) -> Dict[str, Any]:
        """Get configuration for a specific corrector."""
        return self.get(f'correctors.{corrector_name}', {})

    def is_detector_enabled(self, detector_name: str) -> bool:
        """Check if a detector is enabled."""
        return self.get(f'detectors.{detector_name}.enabled', False)

    def is_corrector_enabled(self, corrector_name: str) -> bool:
        """Check if a corrector is enabled."""
        return self.get(f'correctors.{corrector_name}.enabled', False)
