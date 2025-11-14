"""
Debug-focused reporting for detailed diagnostics and troubleshooting.

Features:
- Detailed intervention logs
- Model state snapshots
- Configuration analysis
- Performance profiling
- Integration testing
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import json


class DebugReporter:
    """Generate detailed debug reports for troubleshooting."""

    def __init__(self):
        self.logs = []
        self.interventions = []
        self.state_snapshots = []

    def log_detection(
        self,
        epoch: int,
        detector_name: str,
        severity: str,
        details: Dict[str, Any]
    ) -> None:
        """Log a detection event."""
        event = {
            'timestamp': datetime.now().isoformat(),
            'type': 'detection',
            'epoch': epoch,
            'detector': detector_name,
            'severity': severity,
            'details': details,
        }
        self.logs.append(event)

    def log_correction(
        self,
        epoch: int,
        corrector_name: str,
        action: str,
        parameters: Dict[str, Any]
    ) -> None:
        """Log a correction event."""
        event = {
            'timestamp': datetime.now().isoformat(),
            'type': 'correction',
            'epoch': epoch,
            'corrector': corrector_name,
            'action': action,
            'parameters': parameters,
        }
        self.logs.append(event)
        self.interventions.append(event)

    def snapshot_model_state(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        model_info: Optional[Dict[str, Any]] = None
    ) -> None:
        """Capture model state snapshot."""
        snapshot = {
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'model_info': model_info or {},
        }
        self.state_snapshots.append(snapshot)

    def generate_diagnostic_report(self, verbose: bool = True) -> str:
        """
        Generate comprehensive diagnostic report.

        Args:
            verbose: Include detailed logs

        Returns:
            Formatted diagnostic report
        """
        report = []
        report.append("=" * 100)
        report.append("🔍 OVERFIT GUARD - DIAGNOSTIC REPORT")
        report.append("=" * 100)
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total Events: {len(self.logs)}")
        report.append(f"Interventions: {len(self.interventions)}")
        report.append(f"State Snapshots: {len(self.state_snapshots)}\n")

        # Event Summary
        report.append("\n📊 EVENT SUMMARY")
        report.append("─" * 100)

        detection_count = sum(1 for log in self.logs if log['type'] == 'detection')
        correction_count = sum(1 for log in self.logs if log['type'] == 'correction')

        report.append(f"Detections: {detection_count}")
        report.append(f"Corrections: {correction_count}")
        report.append(f"Correction Rate: {correction_count/detection_count*100:.1f}%" if detection_count > 0 else "N/A")

        # Severity Breakdown
        if detection_count > 0:
            report.append("\n\n📈 SEVERITY BREAKDOWN")
            report.append("─" * 100)

            severities = {}
            for log in self.logs:
                if log['type'] == 'detection':
                    sev = log.get('severity', 'UNKNOWN')
                    severities[sev] = severities.get(sev, 0) + 1

            for sev, count in sorted(severities.items()):
                pct = count / detection_count * 100
                report.append(f"{sev:<15}: {count:>5} ({pct:>5.1f}%)")

        # Intervention Timeline
        if self.interventions:
            report.append("\n\n⚡ INTERVENTION TIMELINE")
            report.append("─" * 100)
            report.append(f"{'Epoch':<8} {'Corrector':<25} {'Action':<30} {'Parameters'}")
            report.append("─" * 100)

            for intervention in self.interventions[:20]:  # Show first 20
                epoch = intervention.get('epoch', '?')
                corrector = intervention.get('corrector', 'Unknown')[:24]
                action = intervention.get('action', 'Unknown')[:29]
                params = str(intervention.get('parameters', {}))[:30]
                report.append(f"{epoch:<8} {corrector:<25} {action:<30} {params}")

            if len(self.interventions) > 20:
                report.append(f"... and {len(self.interventions) - 20} more")

        # Detailed Logs
        if verbose and self.logs:
            report.append("\n\n📋 DETAILED EVENT LOG")
            report.append("─" * 100)

            for i, log in enumerate(self.logs[:50], 1):  # Show first 50
                report.append(f"\n[{i}] {log['type'].upper()} at epoch {log.get('epoch', '?')}")
                report.append(f"    Timestamp: {log['timestamp']}")

                if log['type'] == 'detection':
                    report.append(f"    Detector: {log.get('detector', 'Unknown')}")
                    report.append(f"    Severity: {log.get('severity', 'Unknown')}")
                    if log.get('details'):
                        report.append(f"    Details: {log['details']}")

                elif log['type'] == 'correction':
                    report.append(f"    Corrector: {log.get('corrector', 'Unknown')}")
                    report.append(f"    Action: {log.get('action', 'Unknown')}")
                    if log.get('parameters'):
                        report.append(f"    Parameters: {log['parameters']}")

            if len(self.logs) > 50:
                report.append(f"\n... and {len(self.logs) - 50} more events")

        # State Evolution
        if self.state_snapshots:
            report.append("\n\n📸 STATE EVOLUTION")
            report.append("─" * 100)
            report.append(f"{'Epoch':<8} {'Train Loss':<15} {'Val Loss':<15} {'Gap':<15}")
            report.append("─" * 100)

            for snapshot in self.state_snapshots[::5]:  # Every 5th snapshot
                epoch = snapshot['epoch']
                train_loss = snapshot['train_metrics'].get('loss', 0)
                val_loss = snapshot['val_metrics'].get('loss', 0)
                gap = train_loss - val_loss

                report.append(f"{epoch:<8} {train_loss:<15.4f} {val_loss:<15.4f} {gap:<15.4f}")

        report.append("\n" + "=" * 100 + "\n")

        return "\n".join(report)

    def export_logs_json(self, filepath: str) -> None:
        """Export all logs to JSON file."""
        data = {
            'logs': self.logs,
            'interventions': self.interventions,
            'state_snapshots': self.state_snapshots,
            'summary': {
                'total_events': len(self.logs),
                'total_interventions': len(self.interventions),
                'total_snapshots': len(self.state_snapshots),
            }
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def analyze_intervention_effectiveness(self) -> Dict[str, Any]:
        """
        Analyze effectiveness of interventions.

        Returns:
            Analysis results
        """
        if not self.interventions or not self.state_snapshots:
            return {'error': 'Insufficient data for analysis'}

        # Find state changes after interventions
        effectiveness = {
            'total_interventions': len(self.interventions),
            'improvements': 0,
            'neutral': 0,
            'degradations': 0,
            'details': []
        }

        for intervention in self.interventions:
            epoch = intervention['epoch']

            # Find snapshots before and after
            before = None
            after = None

            for snapshot in self.state_snapshots:
                if snapshot['epoch'] < epoch:
                    before = snapshot
                elif snapshot['epoch'] > epoch and after is None:
                    after = snapshot

            if before and after:
                # Compare gaps
                before_gap = abs(before['train_metrics'].get('loss', 0) - before['val_metrics'].get('loss', 0))
                after_gap = abs(after['train_metrics'].get('loss', 0) - after['val_metrics'].get('loss', 0))

                change = after_gap - before_gap
                if change < -0.01:  # Improvement
                    effectiveness['improvements'] += 1
                    result = 'improved'
                elif change > 0.01:  # Degradation
                    effectiveness['degradations'] += 1
                    result = 'degraded'
                else:  # Neutral
                    effectiveness['neutral'] += 1
                    result = 'neutral'

                effectiveness['details'].append({
                    'epoch': epoch,
                    'corrector': intervention.get('corrector'),
                    'before_gap': before_gap,
                    'after_gap': after_gap,
                    'change': change,
                    'result': result
                })

        return effectiveness

    def generate_troubleshooting_guide(self, summary: Dict[str, Any]) -> str:
        """
        Generate troubleshooting recommendations based on results.

        Args:
            summary: Output from compute_overfit_guard_summary

        Returns:
            Troubleshooting guide
        """
        delta = summary['delta']
        ga = summary['guard_activity']

        guide = []
        guide.append("=" * 80)
        guide.append("🔧 TROUBLESHOOTING GUIDE")
        guide.append("=" * 80)

        issues_found = False

        # Check for problematic patterns
        if delta['test_metric_pct_points'] < -2.0:
            issues_found = True
            guide.append("\n⚠️  ISSUE: Significant test performance degradation")
            guide.append("\nPossible Causes:")
            guide.append("1. Corrections are too aggressive")
            guide.append("2. Correction cooldown is too short")
            guide.append("3. Detection thresholds are too sensitive")
            guide.append("\nRecommended Actions:")
            guide.append("- Increase min_severity_for_correction to 'SEVERE'")
            guide.append("- Increase correction_cooldown to 10+")
            guide.append("- Try detection-only mode (auto_correct=False)")
            guide.append("- Review individual corrector settings")

        if ga['detection_rate'] > 0.8:
            issues_found = True
            guide.append("\n⚠️  ISSUE: Very high detection rate")
            guide.append("\nPossible Causes:")
            guide.append("1. Dataset is too small")
            guide.append("2. Model is too complex")
            guide.append("3. Detection thresholds are too strict")
            guide.append("\nRecommended Actions:")
            guide.append("- Collect more training data")
            guide.append("- Reduce model complexity")
            guide.append("- Increase gap_threshold_moderate in detector config")

        if ga['corrections'] == 0 and ga['detections'] > 10:
            issues_found = True
            guide.append("\n⚠️  ISSUE: Many detections but no corrections")
            guide.append("\nPossible Causes:")
            guide.append("1. auto_correct is disabled")
            guide.append("2. min_severity_for_correction is too high")
            guide.append("3. Correctors cannot handle this model type")
            guide.append("\nRecommended Actions:")
            guide.append("- Enable auto_correct=True")
            guide.append("- Lower min_severity_for_correction")
            guide.append("- Check corrector compatibility with your framework")

        if delta['gap_reduction_pct'] < 0:
            issues_found = True
            guide.append("\n⚠️  ISSUE: Gap increased instead of decreased")
            guide.append("\nPossible Causes:")
            guide.append("1. Corrections are counterproductive")
            guide.append("2. Model needs different regularization approach")
            guide.append("\nRecommended Actions:")
            guide.append("- Try different corrector strategies")
            guide.append("- Disable specific correctors that may not fit")
            guide.append("- Use manual hyperparameter tuning")

        if not issues_found:
            guide.append("\n✅ No issues detected. System is functioning well!")
            guide.append("\nBest Practices:")
            guide.append("- Continue monitoring over multiple runs")
            guide.append("- Collect metrics for different datasets")
            guide.append("- Share results with team for validation")

        guide.append("\n" + "=" * 80 + "\n")

        return "\n".join(guide)

    def compare_configurations(
        self,
        results: List[Dict[str, Any]],
        config_names: List[str]
    ) -> str:
        """
        Compare results across different configurations.

        Args:
            results: List of summary dictionaries
            config_names: List of configuration names

        Returns:
            Comparison report
        """
        report = []
        report.append("=" * 100)
        report.append("⚙️  CONFIGURATION COMPARISON")
        report.append("=" * 100)
        report.append("")

        header = f"{'Configuration':<25} {'Test Δ':<15} {'Gap Red':<15} {'Detections':<15} {'Corrections':<15}"
        report.append(header)
        report.append("─" * 100)

        for name, summary in zip(config_names, results):
            delta = summary['delta']
            ga = summary['guard_activity']

            row = (f"{name:<25} "
                   f"{delta['test_metric_pct_points']:+.2f}pp{'':<7} "
                   f"{delta['gap_reduction_pct']:+.1f}%{'':<9} "
                   f"{ga['detections']:<15} "
                   f"{ga['corrections']:<15}")
            report.append(row)

        report.append("")
        report.append("=" * 100)
        report.append("")

        return "\n".join(report)
