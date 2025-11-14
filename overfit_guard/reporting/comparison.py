"""
Multi-style comparison reporting for Overfit Guard.

Supports three presentation styles:
- research: Paper-ready, precise, no emojis
- marketing: Story-driven, emojis, executive-friendly
- debug: Raw structured data for troubleshooting
"""

import math
from pprint import pprint
from typing import Dict, Any, Optional

# Tolerances for "no meaningful change"
ACCURACY_TOL = 0.1   # 0.1 percentage points
GAP_TOL = 1.0        # 1% relative change
EPS = 1e-8


def compute_overfit_guard_summary(
    history_baseline: Dict[str, list],
    history_guard: Dict[str, list],
    test_metric_baseline: float,
    test_metric_guard: float,
    monitor: Any,
    metric_name: str = 'accuracy',
    higher_is_better: bool = True
) -> Dict[str, Any]:
    """
    Compute all comparison metrics between baseline and guard runs.

    Args:
        history_baseline: Training history without guard
        history_guard: Training history with guard
        test_metric_baseline: Test metric without guard
        test_metric_guard: Test metric with guard
        monitor: Overfit guard monitor instance
        metric_name: Name of the metric being tracked
        higher_is_better: Whether higher metric values are better

    Returns:
        Comprehensive summary dictionary with all metrics
    """
    # Determine metric keys based on metric_name
    train_key = f'train_{metric_name}' if f'train_{metric_name}' in history_baseline else 'train_metric'
    val_key = f'val_{metric_name}' if f'val_{metric_name}' in history_baseline else 'val_metric'

    # Final metrics
    train_metric_base = history_baseline[train_key][-1]
    val_metric_base = history_baseline[val_key][-1]
    train_metric_guard = history_guard[train_key][-1]
    val_metric_guard = history_guard[val_key][-1]

    # Train–val gaps (train - val for metrics where higher is better)
    if higher_is_better:
        gap_baseline = train_metric_base - val_metric_base
        gap_guard = train_metric_guard - val_metric_guard
    else:
        gap_baseline = val_metric_base - train_metric_base
        gap_guard = val_metric_guard - train_metric_guard

    # Test metric change
    if higher_is_better:
        test_improvement = (test_metric_guard - test_metric_baseline) * 100
    else:
        test_improvement = (test_metric_baseline - test_metric_guard) * 100

    # Gap reduction (absolute and % of baseline magnitude)
    gap_reduction = gap_baseline - gap_guard
    if abs(gap_baseline) > EPS:
        gap_reduction_pct = (gap_reduction / abs(gap_baseline)) * 100
    else:
        gap_reduction_pct = 0.0

    # Guard summary from monitor
    raw_summary = {}
    if hasattr(monitor, 'monitor'):
        raw_summary = monitor.monitor.get_summary()
    elif hasattr(monitor, 'monitor_obj'):
        raw_summary = monitor.monitor_obj.get_summary()
    elif hasattr(monitor, 'get_summary'):
        raw_summary = monitor.get_summary()

    detections = raw_summary.get('overfitting_detected', 0)
    corrections = raw_summary.get('corrections_applied', 0)
    detection_rate = raw_summary.get('overfitting_rate', 0.0)
    active_detectors = raw_summary.get('active_detectors', [])
    active_correctors = raw_summary.get('active_correctors', [])

    # Training efficiency
    epochs_baseline = len(history_baseline[train_key])
    epochs_guard = len(history_guard[train_key])
    epochs_saved = epochs_baseline - epochs_guard

    return {
        'baseline': {
            'test_metric': test_metric_baseline,
            'train_metric': train_metric_base,
            'val_metric': val_metric_base,
            'gap': gap_baseline,
            'epochs': epochs_baseline,
        },
        'guard': {
            'test_metric': test_metric_guard,
            'train_metric': train_metric_guard,
            'val_metric': val_metric_guard,
            'gap': gap_guard,
            'epochs': epochs_guard,
        },
        'delta': {
            'test_metric_pct_points': test_improvement,
            'gap_abs': gap_reduction,
            'gap_rel_pct': gap_reduction_pct,
            'epochs_saved': epochs_saved,
            'time_savings_pct': (epochs_saved / epochs_baseline * 100) if epochs_baseline > 0 else 0,
        },
        'guard_activity': {
            'detections': detections,
            'corrections': corrections,
            'detection_rate': detection_rate,
            'active_detectors': active_detectors,
            'active_correctors': active_correctors,
        },
        'meta': {
            'metric_name': metric_name,
            'higher_is_better': higher_is_better,
            'accuracy_tol_pct_points': ACCURACY_TOL,
            'gap_tol_pct': GAP_TOL,
        },
    }


def print_overfit_guard_summary(
    summary: Dict[str, Any],
    style: str = "marketing",
    include_recommendations: bool = True
) -> None:
    """
    Pretty-print the summary in different styles.

    Args:
        summary: Output from compute_overfit_guard_summary
        style: "marketing", "research", or "debug"
        include_recommendations: Whether to include actionable recommendations
    """
    base = summary['baseline']
    guard = summary['guard']
    delta = summary['delta']
    ga = summary['guard_activity']
    meta = summary['meta']

    metric_name = meta['metric_name']
    higher_is_better = meta['higher_is_better']
    test_improvement = delta['test_metric_pct_points']
    gap_reduction_pct = delta['gap_rel_pct']

    if style == "debug":
        print("\n" + "=" * 80)
        print("OVERFIT GUARD DEBUG SUMMARY")
        print("=" * 80)
        pprint(summary)
        print("=" * 80 + "\n")
        return

    # Shared header
    print("\n" + "=" * 80)
    title = f"Overfit Guard – Final Comparison ({metric_name.upper()})"
    if style == "marketing":
        print("🏆 " + title)
    else:
        print(title)
    print("=" * 80)

    # Baseline block
    if style == "marketing":
        print("\n📊 WITHOUT Overfit Guard (Baseline):")
    else:
        print("\nBaseline (no Overfit Guard):")
    print(f"   Test {metric_name}: {base['test_metric']:.4f}")
    print(f"   Train–Val Gap: {base['gap']:.4f}")
    print(f"   Final Train: {base['train_metric']:.4f}")
    print(f"   Final Val:   {base['val_metric']:.4f}")
    print(f"   Epochs:      {base['epochs']}")

    # Guard block
    if style == "marketing":
        print("\n🛡️  WITH Overfit Guard:")
    else:
        print("\nWith Overfit Guard:")
    print(f"   Test {metric_name}: {guard['test_metric']:.4f}")
    print(f"   Train–Val Gap: {guard['gap']:.4f}")
    print(f"   Final Train: {guard['train_metric']:.4f}")
    print(f"   Final Val:   {guard['val_metric']:.4f}")
    print(f"   Epochs:      {guard['epochs']}")

    # Improvements block
    if style == "marketing":
        print("\n💡 Improvements:")
    else:
        print("\nDifferences (Guard vs Baseline):")

    print(f"   Test {metric_name} Δ: {test_improvement:+.2f} percentage points")
    print(f"   Gap Reduction:   {delta['gap_abs']:+.4f} "
          f"({gap_reduction_pct:+.1f}% of baseline)")

    if delta['epochs_saved'] > 0:
        print(f"   Epochs Saved:    {delta['epochs_saved']} "
              f"({delta['time_savings_pct']:.1f}% time reduction)")
    elif delta['epochs_saved'] < 0:
        print(f"   Extra Epochs:    {abs(delta['epochs_saved'])} "
              f"({abs(delta['time_savings_pct']):.1f}% more training)")

    # Guard activity
    if style == "marketing":
        print(f"\n🔧 Guard Activity:")
    else:
        print(f"\nGuard Activity:")
    print(f"   Detections:       {ga['detections']}")
    print(f"   Corrections:      {ga['corrections']}")
    print(f"   Detection Rate:   {ga['detection_rate']:.1%}")
    if ga['active_detectors']:
        print(f"   Active Detectors: {', '.join(ga['active_detectors'])}")
    if ga['active_correctors']:
        print(f"   Active Correctors: {', '.join(ga['active_correctors'])}")

    # Interpretation
    print("\n" + "-" * 80)
    if style == "marketing":
        print("📝 Interpretation:")
    else:
        print("Interpretation:")

    # Metric interpretation
    if abs(test_improvement) < meta['accuracy_tol_pct_points']:
        if style == "marketing":
            print(f"   ✅ Test {metric_name} is effectively unchanged with Overfit Guard.")
        else:
            print(f"   Test {metric_name} is essentially unchanged (within tolerance).")
    elif test_improvement > 0:
        if style == "marketing":
            print(f"   ✅ Test {metric_name} improved by {test_improvement:.2f} percentage points!")
        else:
            print(f"   Test {metric_name} improved by {test_improvement:.2f} percentage points.")
    else:
        if style == "marketing":
            print(f"   ⚠️  Test {metric_name} decreased by {abs(test_improvement):.2f} percentage points.")
            print("      This is an acceptable trade-off when prioritizing robustness and generalization.")
        else:
            print(f"   Test {metric_name} decreased by {abs(test_improvement):.2f} percentage points.")
            print("   This may indicate overly aggressive correction or a very clean dataset.")

    # Gap interpretation
    if delta['gap_abs'] > 0 and abs(gap_reduction_pct) > meta['gap_tol_pct']:
        if style == "marketing":
            print(f"   ✅ Train–val gap reduced by {gap_reduction_pct:.1f}%, "
                  "indicating stronger generalization!")
        else:
            print(f"   Train–val gap reduced by {gap_reduction_pct:.1f}%, "
                  "suggesting improved generalization.")
    elif delta['gap_abs'] > 0:
        print(f"   Train–val gap slightly reduced ({gap_reduction_pct:.1f}%).")
    elif abs(gap_reduction_pct) < meta['gap_tol_pct']:
        print("   Train–val gap essentially unchanged.")
    else:
        if style == "marketing":
            print(f"   ⚠️  Train–val gap increased by {abs(gap_reduction_pct):.1f}%.")
            print("      Consider tuning Overfit Guard sensitivity for this dataset.")
        else:
            print(f"   Train–val gap increased by {abs(gap_reduction_pct):.1f}%.")
            print("   Consider adjusting detection thresholds or corrector strength.")

    # Recommendations
    if include_recommendations and style != "debug":
        print("\n" + "-" * 80)
        if style == "marketing":
            print("💡 Recommendations:")
        else:
            print("Recommendations:")

        _print_recommendations(summary, style)

    print("\n" + "=" * 80 + "\n")


def _print_recommendations(summary: Dict[str, Any], style: str) -> None:
    """Print actionable recommendations based on results."""
    delta = summary['delta']
    ga = summary['guard_activity']
    meta = summary['meta']

    recommendations = []

    # Detection-based recommendations
    if ga['detection_rate'] > 0.5:
        recommendations.append(
            "High detection rate suggests aggressive overfitting. "
            "Consider increasing dataset size or model regularization."
        )
    elif ga['detection_rate'] < 0.1 and ga['detections'] > 0:
        recommendations.append(
            "Low detection rate indicates occasional overfitting. "
            "Guard is effectively preventing issues."
        )

    # Correction-based recommendations
    if ga['corrections'] == 0 and ga['detections'] > 0:
        recommendations.append(
            "Detections occurred but no corrections applied. "
            "Consider enabling auto_correct or lowering correction thresholds."
        )
    elif ga['corrections'] > ga['detections'] * 0.8:
        recommendations.append(
            "High correction rate. If test performance decreased, "
            "consider increasing min_severity_for_correction."
        )

    # Performance-based recommendations
    if delta['test_metric_pct_points'] < -1.0:
        recommendations.append(
            f"Test {meta['metric_name']} decreased significantly. "
            "Try: (1) Reduce correction aggressiveness, "
            "(2) Increase correction_cooldown, "
            "(3) Use detection-only mode first."
        )

    # Time efficiency recommendations
    if delta['epochs_saved'] > 5:
        recommendations.append(
            f"Early stopping saved {delta['epochs_saved']} epochs "
            f"({delta['time_savings_pct']:.0f}% time reduction). "
            "This is a strong ROI for using Overfit Guard!"
        )

    # Print recommendations
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            if style == "marketing":
                print(f"   {i}. 💡 {rec}")
            else:
                print(f"   {i}. {rec}")
    else:
        if style == "marketing":
            print("   ✅ Results look good! Continue using current configuration.")
        else:
            print("   No specific recommendations. Current configuration appears appropriate.")


def export_summary_to_dict(summary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Export summary in a flat format suitable for logging/tracking.

    Returns:
        Flattened dictionary with all metrics
    """
    return {
        # Baseline metrics
        'baseline_test_metric': summary['baseline']['test_metric'],
        'baseline_train_metric': summary['baseline']['train_metric'],
        'baseline_val_metric': summary['baseline']['val_metric'],
        'baseline_gap': summary['baseline']['gap'],
        'baseline_epochs': summary['baseline']['epochs'],

        # Guard metrics
        'guard_test_metric': summary['guard']['test_metric'],
        'guard_train_metric': summary['guard']['train_metric'],
        'guard_val_metric': summary['guard']['val_metric'],
        'guard_gap': summary['guard']['gap'],
        'guard_epochs': summary['guard']['epochs'],

        # Deltas
        'test_metric_improvement_pct': summary['delta']['test_metric_pct_points'],
        'gap_reduction_abs': summary['delta']['gap_abs'],
        'gap_reduction_pct': summary['delta']['gap_rel_pct'],
        'epochs_saved': summary['delta']['epochs_saved'],
        'time_savings_pct': summary['delta']['time_savings_pct'],

        # Guard activity
        'detections': summary['guard_activity']['detections'],
        'corrections': summary['guard_activity']['corrections'],
        'detection_rate': summary['guard_activity']['detection_rate'],

        # Meta
        'metric_name': summary['meta']['metric_name'],
        'higher_is_better': summary['meta']['higher_is_better'],
    }
