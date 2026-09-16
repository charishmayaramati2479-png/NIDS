from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from datetime import datetime
import pandas as pd

class ReportGenerator:
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#003366')
        )
    
    def generate_report(self, predictions, filename='nids_report.pdf'):
        """Generate PDF report"""
        doc = SimpleDocTemplate(
            filename,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        story = []
        
        # Title
        story.append(Paragraph("Network Intrusion Detection Report", self.title_style))
        story.append(Spacer(1, 0.25*inch))
        
        # Date
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", self.styles['Normal']))
        story.append(Spacer(1, 0.25*inch))
        
        # Summary
        story.append(Paragraph("Summary", self.styles['Heading2']))
        total = len(predictions)
        attacks = sum(1 for p in predictions if p.get('prediction', 0) == 1)
        
        story.append(Paragraph(f"Total Connections Analyzed: {total}", self.styles['Normal']))
        story.append(Paragraph(f"Attacks Detected: {attacks}", self.styles['Normal']))
        story.append(Paragraph(f"Normal Traffic: {total - attacks}", self.styles['Normal']))
        story.append(Spacer(1, 0.25*inch))
        
        # Table
        story.append(Paragraph("Detailed Results", self.styles['Heading2']))
        
        table_data = [['#', 'Prediction', 'Confidence', 'Attack Type']]
        for i, pred in enumerate(predictions[:20], 1):
            table_data.append([
                str(i),
                '🚨 Attack' if pred.get('prediction', 0) == 1 else '✅ Normal',
                f"{pred.get('confidence', 0):.2%}",
                pred.get('attack_type', 'N/A')
            ])
        
        table = Table(table_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(table)
        
        # Build PDF
        doc.build(story)
        print(f"✅ PDF Report generated: {filename}")
        return filename

if __name__ == "__main__":
    sample_predictions = [
        {'prediction': 0, 'confidence': 0.95, 'attack_type': 'Normal'},
        {'prediction': 1, 'confidence': 0.88, 'attack_type': 'DoS'}
    ]
    ReportGenerator().generate_report(sample_predictions, 'test_report.pdf')