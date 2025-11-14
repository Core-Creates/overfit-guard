"""
Model card generation for ML model documentation and compliance.

Follows the Model Cards for Model Reporting framework:
https://arxiv.org/abs/1810.03993
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import json


class ModelCardGenerator:
    """Generate standardized model cards for documentation and compliance."""

    def __init__(self):
        self.timestamp = datetime.now().isoformat()

    def generate_model_card(
        self,
        model_details: Dict[str, Any],
        training_details: Dict[str, Any],
        evaluation_results: Dict[str, Any],
        overfit_guard_summary: Optional[Dict[str, Any]] = None,
        intended_use: Optional[Dict[str, Any]] = None,
        ethical_considerations: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive model card.

        Args:
            model_details: Model architecture and specifications
            training_details: Training procedure and hyperparameters
            evaluation_results: Performance metrics
            overfit_guard_summary: Overfit Guard monitoring results
            intended_use: Intended use cases and users
            ethical_considerations: Ethical considerations and limitations

        Returns:
            Complete model card dictionary
        """
        card = {
            'model_card_version': '1.0',
            'generated_date': self.timestamp,

            'model_details': {
                'name': model_details.get('name', 'Unknown Model'),
                'version': model_details.get('version', '1.0'),
                'description': model_details.get('description', ''),
                'architecture': model_details.get('architecture', {}),
                'framework': model_details.get('framework', 'Unknown'),
                'parameters': model_details.get('parameters', {}),
                'license': model_details.get('license', 'Proprietary'),
                'citation': model_details.get('citation', ''),
            },

            'intended_use': intended_use or {
                'primary_uses': [],
                'primary_users': [],
                'out_of_scope_uses': [],
            },

            'factors': {
                'relevant_factors': [],
                'evaluation_factors': [],
            },

            'metrics': {
                'performance_metrics': evaluation_results,
                'decision_thresholds': {},
            },

            'training_data': training_details.get('dataset', {}),

            'evaluation_data': training_details.get('validation_set', {}),

            'training_procedure': {
                'preprocessing': training_details.get('preprocessing', []),
                'optimizer': training_details.get('optimizer', {}),
                'training_regime': training_details.get('regime', {}),
            },

            'quantitative_analyses': {
                'performance': evaluation_results,
            },

            'ethical_considerations': ethical_considerations or [],

            'caveats_and_recommendations': [],
        }

        # Add Overfit Guard specific information
        if overfit_guard_summary:
            card['overfit_guard_monitoring'] = {
                'enabled': True,
                'detections': overfit_guard_summary['guard_activity']['detections'],
                'corrections': overfit_guard_summary['guard_activity']['corrections'],
                'detection_rate': overfit_guard_summary['guard_activity']['detection_rate'],
                'generalization_improvement': overfit_guard_summary['delta']['gap_reduction_pct'],
                'active_detectors': overfit_guard_summary['guard_activity']['active_detectors'],
                'active_correctors': overfit_guard_summary['guard_activity']['active_correctors'],
                'training_efficiency': {
                    'epochs_saved': overfit_guard_summary['delta']['epochs_saved'],
                    'time_savings_pct': overfit_guard_summary['delta']['time_savings_pct'],
                }
            }

        return card

    def export_to_markdown(self, model_card: Dict[str, Any]) -> str:
        """
        Export model card to Markdown format.

        Args:
            model_card: Model card dictionary

        Returns:
            Markdown formatted model card
        """
        md = []

        # Header
        md.append(f"# Model Card: {model_card['model_details']['name']}")
        md.append(f"\n*Generated: {model_card['generated_date']}*")
        md.append(f"\n*Version: {model_card['model_details']['version']}*\n")

        # Model Details
        md.append("## Model Details")
        md.append(f"\n**Description:** {model_card['model_details']['description']}\n")
        md.append(f"**Framework:** {model_card['model_details']['framework']}\n")
        md.append(f"**License:** {model_card['model_details']['license']}\n")

        if model_card['model_details'].get('architecture'):
            md.append("### Architecture")
            for key, value in model_card['model_details']['architecture'].items():
                md.append(f"- **{key}:** {value}")
            md.append("")

        # Intended Use
        md.append("## Intended Use")
        if model_card['intended_use']['primary_uses']:
            md.append("\n### Primary Uses")
            for use in model_card['intended_use']['primary_uses']:
                md.append(f"- {use}")

        if model_card['intended_use']['primary_users']:
            md.append("\n### Primary Users")
            for user in model_card['intended_use']['primary_users']:
                md.append(f"- {user}")

        if model_card['intended_use']['out_of_scope_uses']:
            md.append("\n### Out-of-Scope Uses")
            for use in model_card['intended_use']['out_of_scope_uses']:
                md.append(f"- {use}")
        md.append("")

        # Performance Metrics
        md.append("## Performance Metrics")
        for metric, value in model_card['metrics']['performance_metrics'].items():
            if isinstance(value, (int, float)):
                md.append(f"- **{metric}:** {value:.4f}")
            else:
                md.append(f"- **{metric}:** {value}")
        md.append("")

        # Overfit Guard Monitoring
        if 'overfit_guard_monitoring' in model_card:
            og = model_card['overfit_guard_monitoring']
            md.append("## Overfit Guard Monitoring")
            md.append(f"\n**Status:** {'Enabled' if og['enabled'] else 'Disabled'}\n")
            md.append(f"- **Overfitting Detections:** {og['detections']}")
            md.append(f"- **Auto-Corrections Applied:** {og['corrections']}")
            md.append(f"- **Detection Rate:** {og['detection_rate']:.1%}")
            md.append(f"- **Generalization Improvement:** {og['generalization_improvement']:.1f}%")

            if og['training_efficiency']['epochs_saved'] > 0:
                md.append(f"- **Training Time Saved:** {og['training_efficiency']['time_savings_pct']:.0f}% "
                         f"({og['training_efficiency']['epochs_saved']} epochs)")

            if og['active_detectors']:
                md.append(f"\n**Active Detectors:** {', '.join(og['active_detectors'])}")
            if og['active_correctors']:
                md.append(f"\n**Active Correctors:** {', '.join(og['active_correctors'])}")
            md.append("")

        # Training Procedure
        md.append("## Training Procedure")
        if model_card['training_procedure'].get('optimizer'):
            md.append("\n### Optimizer")
            for key, value in model_card['training_procedure']['optimizer'].items():
                md.append(f"- **{key}:** {value}")

        if model_card['training_procedure'].get('preprocessing'):
            md.append("\n### Preprocessing")
            for step in model_card['training_procedure']['preprocessing']:
                md.append(f"- {step}")
        md.append("")

        # Ethical Considerations
        if model_card['ethical_considerations']:
            md.append("## Ethical Considerations")
            for consideration in model_card['ethical_considerations']:
                md.append(f"- {consideration}")
            md.append("")

        # Caveats
        if model_card['caveats_and_recommendations']:
            md.append("## Caveats and Recommendations")
            for caveat in model_card['caveats_and_recommendations']:
                md.append(f"- {caveat}")
            md.append("")

        return "\n".join(md)

    def export_to_json(self, model_card: Dict[str, Any], filepath: str) -> None:
        """
        Export model card to JSON file.

        Args:
            model_card: Model card dictionary
            filepath: Output file path
        """
        with open(filepath, 'w') as f:
            json.dump(model_card, f, indent=2)

    def export_to_html(self, model_card: Dict[str, Any]) -> str:
        """
        Export model card to HTML format.

        Args:
            model_card: Model card dictionary

        Returns:
            HTML formatted model card
        """
        html = []
        html.append("<!DOCTYPE html>")
        html.append("<html>")
        html.append("<head>")
        html.append(f"<title>Model Card: {model_card['model_details']['name']}</title>")
        html.append("<style>")
        html.append("""
            body { font-family: Arial, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; }
            h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
            h2 { color: #34495e; margin-top: 30px; border-bottom: 2px solid #95a5a6; padding-bottom: 5px; }
            h3 { color: #7f8c8d; }
            .metric { background: #ecf0f1; padding: 10px; margin: 5px 0; border-radius: 5px; }
            .overfit-guard { background: #d5f4e6; padding: 15px; border-left: 4px solid #27ae60; margin: 10px 0; }
            .warning { background: #ffeaa7; padding: 10px; border-left: 4px solid #fdcb6e; margin: 10px 0; }
        """)
        html.append("</style>")
        html.append("</head>")
        html.append("<body>")

        # Header
        html.append(f"<h1>Model Card: {model_card['model_details']['name']}</h1>")
        html.append(f"<p><em>Generated: {model_card['generated_date']}</em></p>")
        html.append(f"<p><em>Version: {model_card['model_details']['version']}</em></p>")

        # Model Details
        html.append("<h2>Model Details</h2>")
        html.append(f"<p><strong>Description:</strong> {model_card['model_details']['description']}</p>")
        html.append(f"<p><strong>Framework:</strong> {model_card['model_details']['framework']}</p>")

        # Performance Metrics
        html.append("<h2>Performance Metrics</h2>")
        for metric, value in model_card['metrics']['performance_metrics'].items():
            if isinstance(value, (int, float)):
                html.append(f"<div class='metric'><strong>{metric}:</strong> {value:.4f}</div>")
            else:
                html.append(f"<div class='metric'><strong>{metric}:</strong> {value}</div>")

        # Overfit Guard
        if 'overfit_guard_monitoring' in model_card:
            og = model_card['overfit_guard_monitoring']
            html.append("<h2>Overfit Guard Monitoring</h2>")
            html.append("<div class='overfit-guard'>")
            html.append(f"<p><strong>Status:</strong> {'Enabled ✓' if og['enabled'] else 'Disabled'}</p>")
            html.append(f"<p><strong>Detections:</strong> {og['detections']}</p>")
            html.append(f"<p><strong>Corrections:</strong> {og['corrections']}</p>")
            html.append(f"<p><strong>Detection Rate:</strong> {og['detection_rate']:.1%}</p>")
            html.append(f"<p><strong>Generalization Improvement:</strong> {og['generalization_improvement']:.1f}%</p>")
            html.append("</div>")

        # Ethical Considerations
        if model_card['ethical_considerations']:
            html.append("<h2>Ethical Considerations</h2>")
            html.append("<div class='warning'>")
            html.append("<ul>")
            for consideration in model_card['ethical_considerations']:
                html.append(f"<li>{consideration}</li>")
            html.append("</ul>")
            html.append("</div>")

        html.append("</body>")
        html.append("</html>")

        return "\n".join(html)
