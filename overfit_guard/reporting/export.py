"""
Multi-format export utilities for Overfit Guard results.

Supports exporting to:
- JSON (for APIs and data exchange)
- CSV (for spreadsheets and analysis)
- PDF (for reports and documentation)
- LaTeX (for academic papers)
- HTML (for web dashboards)
"""

import json
import csv
from typing import Dict, Any, List, Optional
from pathlib import Path


class ReportExporter:
    """Export Overfit Guard results to multiple formats."""

    def __init__(self):
        pass

    def to_json(self, data: Dict[str, Any], filepath: str, pretty: bool = True) -> None:
        """
        Export to JSON file.

        Args:
            data: Data to export
            filepath: Output file path
            pretty: Whether to use pretty formatting
        """
        with open(filepath, 'w') as f:
            if pretty:
                json.dump(data, f, indent=2)
            else:
                json.dump(data, f)

    def to_csv(self, summary: Dict[str, Any], filepath: str) -> None:
        """
        Export summary to CSV file.

        Args:
            summary: Output from compute_overfit_guard_summary
            filepath: Output file path
        """
        # Flatten summary for CSV
        rows = []

        # Add baseline row
        rows.append({
            'Configuration': 'Baseline',
            'Test Metric': summary['baseline']['test_metric'],
            'Train Metric': summary['baseline']['train_metric'],
            'Val Metric': summary['baseline']['val_metric'],
            'Gap': summary['baseline']['gap'],
            'Epochs': summary['baseline']['epochs'],
        })

        # Add guard row
        rows.append({
            'Configuration': 'With Guard',
            'Test Metric': summary['guard']['test_metric'],
            'Train Metric': summary['guard']['train_metric'],
            'Val Metric': summary['guard']['val_metric'],
            'Gap': summary['guard']['gap'],
            'Epochs': summary['guard']['epochs'],
        })

        # Add delta row
        rows.append({
            'Configuration': 'Improvement',
            'Test Metric': summary['delta']['test_metric_pct_points'],
            'Train Metric': '',
            'Val Metric': '',
            'Gap': summary['delta']['gap_reduction_pct'],
            'Epochs': summary['delta']['epochs_saved'],
        })

        # Write CSV
        if rows:
            with open(filepath, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

    def to_pdf(self, content: str, filepath: str, title: str = "Overfit Guard Report") -> None:
        """
        Export to PDF file (requires reportlab).

        Args:
            content: Text content to export
            filepath: Output file path
            title: Document title
        """
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
            from reportlab.lib.units import inch

            # Create PDF
            doc = SimpleDocTemplate(filepath, pagesize=letter)
            styles = getSampleStyleSheet()
            story = []

            # Add title
            story.append(Paragraph(title, styles['Title']))
            story.append(Spacer(1, 0.2 * inch))

            # Add content (split by lines)
            for line in content.split('\n'):
                if line.strip():
                    # Determine style based on content
                    if line.startswith('#'):
                        story.append(Paragraph(line.replace('#', ''), styles['Heading1']))
                    elif line.startswith('=='):
                        story.append(Spacer(1, 0.1 * inch))
                    else:
                        story.append(Paragraph(line, styles['Normal']))
                else:
                    story.append(Spacer(1, 0.1 * inch))

            # Build PDF
            doc.build(story)

        except ImportError:
            # Fallback: save as text file
            txt_path = filepath.replace('.pdf', '.txt')
            with open(txt_path, 'w') as f:
                f.write(f"{title}\n\n{content}")
            print(f"reportlab not installed. Saved as text: {txt_path}")
            print("Install with: pip install reportlab")

    def to_html_table(self, summary: Dict[str, Any]) -> str:
        """
        Export summary as HTML table.

        Args:
            summary: Output from compute_overfit_guard_summary

        Returns:
            HTML table string
        """
        html = []
        html.append("<table border='1' cellpadding='5' cellspacing='0'>")
        html.append("<thead>")
        html.append("<tr>")
        html.append("<th>Metric</th>")
        html.append("<th>Baseline</th>")
        html.append("<th>With Guard</th>")
        html.append("<th>Improvement</th>")
        html.append("</tr>")
        html.append("</thead>")
        html.append("<tbody>")

        # Test metric row
        html.append("<tr>")
        html.append(f"<td><strong>Test {summary['meta']['metric_name']}</strong></td>")
        html.append(f"<td>{summary['baseline']['test_metric']:.4f}</td>")
        html.append(f"<td>{summary['guard']['test_metric']:.4f}</td>")
        html.append(f"<td>{summary['delta']['test_metric_pct_points']:+.2f}%</td>")
        html.append("</tr>")

        # Gap row
        html.append("<tr>")
        html.append("<td><strong>Train-Val Gap</strong></td>")
        html.append(f"<td>{summary['baseline']['gap']:.4f}</td>")
        html.append(f"<td>{summary['guard']['gap']:.4f}</td>")
        html.append(f"<td>{summary['delta']['gap_reduction_pct']:+.1f}%</td>")
        html.append("</tr>")

        # Epochs row
        html.append("<tr>")
        html.append("<td><strong>Epochs</strong></td>")
        html.append(f"<td>{summary['baseline']['epochs']}</td>")
        html.append(f"<td>{summary['guard']['epochs']}</td>")
        html.append(f"<td>{summary['delta']['epochs_saved']:+d}</td>")
        html.append("</tr>")

        html.append("</tbody>")
        html.append("</table>")

        return "\n".join(html)

    def to_markdown_table(self, summary: Dict[str, Any]) -> str:
        """
        Export summary as Markdown table.

        Args:
            summary: Output from compute_overfit_guard_summary

        Returns:
            Markdown table string
        """
        md = []
        md.append("| Metric | Baseline | With Guard | Improvement |")
        md.append("|--------|----------|------------|-------------|")

        # Test metric
        md.append(f"| Test {summary['meta']['metric_name']} | "
                 f"{summary['baseline']['test_metric']:.4f} | "
                 f"{summary['guard']['test_metric']:.4f} | "
                 f"{summary['delta']['test_metric_pct_points']:+.2f}% |")

        # Gap
        md.append(f"| Train-Val Gap | "
                 f"{summary['baseline']['gap']:.4f} | "
                 f"{summary['guard']['gap']:.4f} | "
                 f"{summary['delta']['gap_reduction_pct']:+.1f}% |")

        # Epochs
        md.append(f"| Epochs | "
                 f"{summary['baseline']['epochs']} | "
                 f"{summary['guard']['epochs']} | "
                 f"{summary['delta']['epochs_saved']:+d} |")

        return "\n".join(md)

    def export_complete_report(
        self,
        summary: Dict[str, Any],
        output_dir: str,
        formats: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """
        Export complete report in multiple formats.

        Args:
            summary: Output from compute_overfit_guard_summary
            output_dir: Output directory
            formats: List of formats to export (json, csv, html, md)

        Returns:
            Dictionary mapping format to filepath
        """
        if formats is None:
            formats = ['json', 'csv', 'html', 'md']

        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        exports = {}

        # JSON export
        if 'json' in formats:
            json_path = output_dir / 'overfit_guard_results.json'
            self.to_json(summary, str(json_path))
            exports['json'] = str(json_path)

        # CSV export
        if 'csv' in formats:
            csv_path = output_dir / 'overfit_guard_results.csv'
            self.to_csv(summary, str(csv_path))
            exports['csv'] = str(csv_path)

        # HTML export
        if 'html' in formats:
            html_path = output_dir / 'overfit_guard_results.html'
            html_content = self._generate_full_html_report(summary)
            with open(html_path, 'w') as f:
                f.write(html_content)
            exports['html'] = str(html_path)

        # Markdown export
        if 'md' in formats:
            md_path = output_dir / 'overfit_guard_results.md'
            md_content = self._generate_markdown_report(summary)
            with open(md_path, 'w') as f:
                f.write(md_content)
            exports['md'] = str(md_path)

        return exports

    def _generate_full_html_report(self, summary: Dict[str, Any]) -> str:
        """Generate complete HTML report."""
        from .comparison import print_overfit_guard_summary
        from io import StringIO
        import sys

        # Capture printed output
        old_stdout = sys.stdout
        sys.stdout = StringIO()

        print_overfit_guard_summary(summary, style="marketing", include_recommendations=True)

        output = sys.stdout.getvalue()
        sys.stdout = old_stdout

        # Convert to HTML
        html = []
        html.append("<!DOCTYPE html>")
        html.append("<html><head>")
        html.append("<title>Overfit Guard Results</title>")
        html.append("<style>")
        html.append("body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; "
                   "max-width: 1000px; margin: 0 auto; padding: 20px; background: #f5f5f5; }")
        html.append("pre { background: white; padding: 20px; border-radius: 8px; "
                   "box-shadow: 0 2px 4px rgba(0,0,0,0.1); overflow-x: auto; }")
        html.append("</style>")
        html.append("</head><body>")
        html.append(f"<pre>{output}</pre>")
        html.append(self.to_html_table(summary))
        html.append("</body></html>")

        return "\n".join(html)

    def _generate_markdown_report(self, summary: Dict[str, Any]) -> str:
        """Generate complete Markdown report."""
        from .comparison import print_overfit_guard_summary
        from io import StringIO
        import sys

        # Capture printed output
        old_stdout = sys.stdout
        sys.stdout = StringIO()

        print_overfit_guard_summary(summary, style="research", include_recommendations=True)

        output = sys.stdout.getvalue()
        sys.stdout = old_stdout

        # Add table
        md = []
        md.append("# Overfit Guard Results\n")
        md.append("## Summary Table\n")
        md.append(self.to_markdown_table(summary))
        md.append("\n## Detailed Report\n")
        md.append("```")
        md.append(output)
        md.append("```")

        return "\n".join(md)
