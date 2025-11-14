"""Metric calculation utilities."""

from typing import Dict, List, Any
import numpy as np


def calculate_metric_statistics(values: List[float]) -> Dict[str, float]:
    """
    Calculate statistics for a list of metric values.

    Args:
        values: List of metric values

    Returns:
        Dictionary with mean, std, min, max, variance
    """
    if not values:
        return {
            'mean': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0,
            'variance': 0.0
        }

    arr = np.array(values)

    return {
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr)),
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'variance': float(np.var(arr))
    }


def calculate_moving_average(
    values: List[float],
    window_size: int = 5
) -> List[float]:
    """
    Calculate moving average of values.

    Args:
        values: List of values
        window_size: Size of moving window

    Returns:
        List of moving averages
    """
    if len(values) < window_size:
        return values

    arr = np.array(values)
    weights = np.ones(window_size) / window_size
    ma = np.convolve(arr, weights, mode='valid')

    # Pad beginning with original values
    result = values[:window_size-1] + ma.tolist()
    return result


def calculate_relative_change(
    current_value: float,
    previous_value: float
) -> float:
    """
    Calculate relative change between two values.

    Args:
        current_value: Current metric value
        previous_value: Previous metric value

    Returns:
        Relative change (percentage)
    """
    if previous_value == 0:
        return 0.0

    return (current_value - previous_value) / abs(previous_value)


def normalize_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
    """
    Normalize metrics to [0, 1] range.

    Args:
        metrics: Dictionary of metrics

    Returns:
        Normalized metrics
    """
    if not metrics:
        return {}

    values = np.array(list(metrics.values()))
    min_val = np.min(values)
    max_val = np.max(values)

    if max_val == min_val:
        return {k: 0.5 for k in metrics.keys()}

    normalized = {}
    for key, value in metrics.items():
        normalized[key] = (value - min_val) / (max_val - min_val)

    return normalized


def calculate_metric_gap(
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    metric_name: str = 'loss',
    relative: bool = True
) -> float:
    """
    Calculate gap between training and validation metrics.

    Args:
        train_metrics: Training metrics
        val_metrics: Validation metrics
        metric_name: Name of metric to compare
        relative: Use relative gap (True) or absolute (False)

    Returns:
        Gap value
    """
    train_val = train_metrics.get(metric_name, 0.0)
    val_val = val_metrics.get(metric_name, 0.0)

    if relative and train_val != 0:
        return abs(val_val - train_val) / abs(train_val)
    else:
        return abs(val_val - train_val)


def exponential_moving_average(
    values: List[float],
    alpha: float = 0.3
) -> List[float]:
    """
    Calculate exponential moving average.

    Args:
        values: List of values
        alpha: Smoothing factor (0 < alpha <= 1)

    Returns:
        List of EMA values
    """
    if not values:
        return []

    ema = [values[0]]

    for value in values[1:]:
        ema.append(alpha * value + (1 - alpha) * ema[-1])

    return ema


def detect_trend(
    values: List[float],
    window_size: int = 5
) -> str:
    """
    Detect trend in metric values.

    Args:
        values: List of metric values
        window_size: Window for trend detection

    Returns:
        'increasing', 'decreasing', or 'stable'
    """
    if len(values) < window_size:
        return 'stable'

    recent = values[-window_size:]
    arr = np.array(recent)
    x = np.arange(len(arr))

    # Linear regression slope
    slope = np.polyfit(x, arr, 1)[0]

    # Threshold for considering trend significant
    threshold = np.std(arr) * 0.1

    if slope > threshold:
        return 'increasing'
    elif slope < -threshold:
        return 'decreasing'
    else:
        return 'stable'
