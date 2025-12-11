"""
MaxQ Report Generator

Generates PDF reports for experiments and evaluations.
"""

from io import BytesIO
from datetime import datetime
from typing import Dict, Any, Optional

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


class EvalReportGenerator:
    """Generate PDF reports for MaxQ evaluations."""

    def __init__(self):
        if not REPORTLAB_AVAILABLE:
            raise ImportError("reportlab is required for PDF generation. Install with: pip install reportlab")

    def generate(
        self,
        project_name: str,
        project_id: str,
        experiment_name: str,
        experiment_id: str,
        embedding_model: str,
        search_strategy: str,
        metrics: Dict[str, Any],
        eval_id: Optional[str] = None
    ) -> bytes:
        """
        Generate a PDF report with experiment and evaluation results.

        Returns PDF as bytes.
        """
        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )

        styles = getSampleStyleSheet()
        story = []

        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.HexColor('#1a1a1a')
        )
        story.append(Paragraph("MaxQ Evaluation Report", title_style))

        # Subtitle with timestamp
        subtitle_style = ParagraphStyle(
            'Subtitle',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.gray,
            spaceAfter=20
        )
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        story.append(Paragraph(f"Generated: {timestamp}", subtitle_style))
        story.append(Spacer(1, 20))

        # Project Info Section
        section_style = ParagraphStyle(
            'Section',
            parent=styles['Heading2'],
            fontSize=14,
            spaceBefore=20,
            spaceAfter=10,
            textColor=colors.HexColor('#333333')
        )
        story.append(Paragraph("Project Information", section_style))

        project_data = [
            ["Project Name", project_name],
            ["Project ID", project_id],
            ["Experiment", experiment_name],
            ["Experiment ID", experiment_id],
        ]
        if eval_id:
            project_data.append(["Evaluation ID", eval_id])

        project_table = Table(project_data, colWidths=[2 * inch, 4 * inch])
        project_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f5f5f5')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dddddd')),
        ]))
        story.append(project_table)
        story.append(Spacer(1, 20))

        # Configuration Section
        story.append(Paragraph("Configuration", section_style))

        config_data = [
            ["Embedding Model", embedding_model],
            ["Search Strategy", search_strategy],
        ]

        config_table = Table(config_data, colWidths=[2 * inch, 4 * inch])
        config_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f5f5f5')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dddddd')),
        ]))
        story.append(config_table)
        story.append(Spacer(1, 20))

        # Metrics Section
        story.append(Paragraph("Evaluation Metrics", section_style))

        # Build metrics table
        metrics_data = [["Metric", "Candidate", "Baseline", "Delta"]]

        if "ndcg" in metrics:
            ndcg = metrics["ndcg"]
            metrics_data.append([
                "nDCG@10",
                str(ndcg.get("candidate", "-")),
                str(ndcg.get("baseline", "-")),
                str(ndcg.get("delta", "-"))
            ])

        if "latency" in metrics:
            latency = metrics["latency"]
            metrics_data.append([
                "Latency (p50)",
                str(latency.get("candidate", "-")),
                str(latency.get("baseline", "-")),
                str(latency.get("delta", "-"))
            ])

        if "hit_rate" in metrics:
            hit_rate = metrics["hit_rate"]
            metrics_data.append([
                "Hit Rate @5",
                str(hit_rate.get("candidate", "-")),
                str(hit_rate.get("baseline", "-")),
                str(hit_rate.get("delta", "-"))
            ])

        if "mrr" in metrics:
            mrr = metrics["mrr"]
            metrics_data.append([
                "MRR",
                str(mrr.get("candidate", "-")),
                str(mrr.get("baseline", "-")),
                str(mrr.get("delta", "-"))
            ])

        # Add any other metrics
        for key, value in metrics.items():
            if key not in ["ndcg", "latency", "hit_rate", "mrr"]:
                if isinstance(value, dict):
                    metrics_data.append([
                        key,
                        str(value.get("candidate", value.get("value", "-"))),
                        str(value.get("baseline", "-")),
                        str(value.get("delta", "-"))
                    ])
                else:
                    metrics_data.append([key, str(value), "-", "-"])

        metrics_table = Table(metrics_data, colWidths=[1.5 * inch, 1.5 * inch, 1.5 * inch, 1.5 * inch])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#333333')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dddddd')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f9f9f9')]),
        ]))
        story.append(metrics_table)
        story.append(Spacer(1, 40))

        # Footer
        footer_style = ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontSize=9,
            textColor=colors.gray,
            alignment=1  # Center
        )
        story.append(Paragraph(f"Generated by MaxQ v0.1.0", footer_style))
        story.append(Paragraph("https://github.com/maxq", footer_style))

        # Build PDF
        doc.build(story)

        return buffer.getvalue()
