"""
Research-grade reporting for academic papers and technical documentation.

Features:
- LaTeX table generation
- Statistical significance testing
- Reproducibility information
- Publication-ready plots
- BibTeX citations
"""

import json
from typing import Dict, Any, List, Optional
from datetime import datetime
import platform


class ResearchReporter:
    """Generate research-paper ready reports and exports."""

    def __init__(self, experiment_name: str = "overfit_guard_experiment"):
        self.experiment_name = experiment_name
        self.timestamp = datetime.now().isoformat()
        self.reproducibility_info = self._collect_reproducibility_info()

    def _collect_reproducibility_info(self) -> Dict[str, Any]:
        """Collect environment and version information for reproducibility."""
        import sys

        info = {
            'timestamp': self.timestamp,
            'python_version': sys.version,
            'platform': platform.platform(),
            'platform_details': {
                'system': platform.system(),
                'release': platform.release(),
                'version': platform.version(),
                'machine': platform.machine(),
                'processor': platform.processor(),
            }
        }

        # Try to get package versions
        try:
            import overfit_guard
            info['overfit_guard_version'] = getattr(overfit_guard, '__version__', 'unknown')
        except:
            info['overfit_guard_version'] = 'unknown'

        try:
            import torch
            info['pytorch_version'] = torch.__version__
            info['cuda_available'] = torch.cuda.is_available()
            if torch.cuda.is_available():
                info['cuda_version'] = torch.version.cuda
        except:
            pass

        try:
            import tensorflow as tf
            info['tensorflow_version'] = tf.__version__
        except:
            pass

        try:
            import sklearn
            info['sklearn_version'] = sklearn.__version__
        except:
            pass

        return info

    def generate_latex_table(self, summary: Dict[str, Any], caption: str = "") -> str:
        """
        Generate LaTeX table from comparison summary.

        Args:
            summary: Output from compute_overfit_guard_summary
            caption: Optional table caption

        Returns:
            LaTeX table code
        """
        base = summary['baseline']
        guard = summary['guard']
        delta = summary['delta']
        meta = summary['meta']
        metric_name = meta['metric_name']

        latex = []
        latex.append("\\begin{table}[htbp]")
        latex.append("\\centering")
        if caption:
            latex.append(f"\\caption{{{caption}}}")
        latex.append("\\label{tab:overfit_guard_results}")
        latex.append("\\begin{tabular}{lcc|c}")
        latex.append("\\hline")
        latex.append(f"\\textbf{{Metric}} & \\textbf{{Baseline}} & \\textbf{{With Guard}} & \\textbf{{$\\Delta$}} \\\\")
        latex.append("\\hline")
        latex.append(f"Test {metric_name} & {base['test_metric']:.4f} & {guard['test_metric']:.4f} & {delta['test_metric_pct_points']:+.2f}\\% \\\\")
        latex.append(f"Train {metric_name} & {base['train_metric']:.4f} & {guard['train_metric']:.4f} & -- \\\\")
        latex.append(f"Val {metric_name} & {base['val_metric']:.4f} & {guard['val_metric']:.4f} & -- \\\\")
        latex.append(f"Train-Val Gap & {base['gap']:.4f} & {guard['gap']:.4f} & {delta['gap_reduction_pct']:+.1f}\\% \\\\")
        latex.append(f"Epochs & {base['epochs']} & {guard['epochs']} & {delta['epochs_saved']:+d} \\\\")
        latex.append("\\hline")
        latex.append("\\end{tabular}")
        latex.append("\\end{table}")

        return "\n".join(latex)

    def generate_latex_results_section(
        self,
        summary: Dict[str, Any],
        dataset_description: str = "",
        model_description: str = ""
    ) -> str:
        """
        Generate a complete LaTeX results section.

        Args:
            summary: Output from compute_overfit_guard_summary
            dataset_description: Description of the dataset
            model_description: Description of the model architecture

        Returns:
            LaTeX section code
        """
        ga = summary['guard_activity']
        meta = summary['meta']
        delta = summary['delta']

        latex = []
        latex.append("\\section{Results}")
        latex.append("")

        if dataset_description:
            latex.append(f"\\subsection{{Dataset}}")
            latex.append(dataset_description)
            latex.append("")

        if model_description:
            latex.append(f"\\subsection{{Model Architecture}}")
            latex.append(model_description)
            latex.append("")

        latex.append("\\subsection{Overfitting Detection and Mitigation}")
        latex.append("")
        latex.append("We evaluated the Overfit Guard system on the experimental setup described above. ")
        latex.append(f"The system detected {ga['detections']} overfitting events during training ")
        latex.append(f"(detection rate: {ga['detection_rate']:.1%}) and applied {ga['corrections']} ")
        latex.append("automatic corrections. ")
        latex.append("")

        latex.append(f"As shown in Table~\\ref{{tab:overfit_guard_results}}, ")
        if delta['gap_reduction_pct'] > 0:
            latex.append(f"the train-validation gap was reduced by {delta['gap_reduction_pct']:.1f}\\%, ")
            latex.append("indicating improved generalization. ")

        if delta['test_metric_pct_points'] > 0:
            latex.append(f"Test {meta['metric_name']} improved by {delta['test_metric_pct_points']:.2f} percentage points. ")
        elif abs(delta['test_metric_pct_points']) < meta['accuracy_tol_pct_points']:
            latex.append(f"Test {meta['metric_name']} remained effectively unchanged. ")
        else:
            latex.append(f"Test {meta['metric_name']} decreased slightly by {abs(delta['test_metric_pct_points']):.2f} percentage points, ")
            latex.append("which represents an acceptable trade-off for improved robustness. ")

        if delta['epochs_saved'] > 0:
            latex.append(f"Additionally, early stopping triggered {delta['epochs_saved']} epochs earlier, ")
            latex.append(f"reducing training time by {delta['time_savings_pct']:.1f}\\%. ")

        latex.append("")
        latex.append(self.generate_latex_table(summary, "Comparison of model performance with and without Overfit Guard."))

        return "\n".join(latex)

    def generate_bibtex_citation(self) -> str:
        """Generate BibTeX citation for Overfit Guard."""
        return """@software{overfit_guard,
  title = {Overfit Guard: Automatic Overfitting Detection and Correction for Machine Learning},
  author = {Pendleton, Michael},
  year = {2025},
  url = {https://github.com/Core-Creates/overfit-guard},
  version = {1.0.0}
}"""

    def save_reproducibility_manifest(self, filepath: str, config: Dict[str, Any]) -> None:
        """
        Save complete reproducibility manifest.

        Args:
            filepath: Path to save JSON manifest
            config: Configuration dictionary used in experiment
        """
        manifest = {
            'experiment_name': self.experiment_name,
            'timestamp': self.timestamp,
            'reproducibility': self.reproducibility_info,
            'configuration': config,
        }

        with open(filepath, 'w') as f:
            json.dump(manifest, f, indent=2)

    def generate_statistical_tests(
        self,
        baseline_scores: List[float],
        guard_scores: List[float]
    ) -> Dict[str, Any]:
        """
        Perform statistical significance tests.

        Args:
            baseline_scores: List of scores from baseline runs
            guard_scores: List of scores from guard runs

        Returns:
            Dictionary with test results
        """
        try:
            from scipy import stats
            import numpy as np

            # Paired t-test
            t_stat, t_pvalue = stats.ttest_rel(guard_scores, baseline_scores)

            # Wilcoxon signed-rank test (non-parametric alternative)
            w_stat, w_pvalue = stats.wilcoxon(guard_scores, baseline_scores)

            # Effect size (Cohen's d)
            mean_diff = np.mean(guard_scores) - np.mean(baseline_scores)
            pooled_std = np.sqrt((np.std(baseline_scores)**2 + np.std(guard_scores)**2) / 2)
            cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0

            return {
                'paired_t_test': {
                    'statistic': float(t_stat),
                    'p_value': float(t_pvalue),
                    'significant_at_0.05': t_pvalue < 0.05,
                },
                'wilcoxon_test': {
                    'statistic': float(w_stat),
                    'p_value': float(w_pvalue),
                    'significant_at_0.05': w_pvalue < 0.05,
                },
                'effect_size': {
                    'cohens_d': float(cohens_d),
                    'interpretation': self._interpret_cohens_d(cohens_d),
                },
                'descriptive_stats': {
                    'baseline_mean': float(np.mean(baseline_scores)),
                    'baseline_std': float(np.std(baseline_scores)),
                    'guard_mean': float(np.mean(guard_scores)),
                    'guard_std': float(np.std(guard_scores)),
                }
            }
        except ImportError:
            return {
                'error': 'scipy not installed. Install with: pip install scipy'
            }

    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size."""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"

    def generate_methods_section(self, config: Dict[str, Any]) -> str:
        """
        Generate LaTeX methods section describing Overfit Guard configuration.

        Args:
            config: Configuration dictionary

        Returns:
            LaTeX methods section
        """
        latex = []
        latex.append("\\subsection{Overfitting Detection and Mitigation}")
        latex.append("")
        latex.append("We employed the Overfit Guard system \\cite{overfit_guard} to automatically ")
        latex.append("detect and mitigate overfitting during training. The system uses multiple ")
        latex.append("detection strategies including train-validation gap monitoring, learning curve ")
        latex.append("analysis, and statistical tests. ")
        latex.append("")

        latex.append("\\textbf{Configuration:} ")
        if config.get('auto_correct'):
            latex.append("Automatic correction was enabled with a minimum severity threshold of ")
            latex.append(f"{config.get('min_severity_for_correction', 'MODERATE')}. ")
        else:
            latex.append("Detection-only mode was used without automatic correction. ")

        latex.append(f"A correction cooldown period of {config.get('correction_cooldown', 5)} epochs ")
        latex.append("was enforced between consecutive interventions to prevent over-correction. ")

        return "\n".join(latex)
