"""
Generates official submission PDFs:
1. docs/architecture_diagram.pdf
2. docs/presentation.pdf
"""

import os
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Group

DOCS_DIR = os.path.dirname(os.path.abspath(__file__))

def create_architecture_diagram():
    pdf_path = os.path.join(DOCS_DIR, "architecture_diagram.pdf")
    doc = SimpleDocTemplate(pdf_path, pagesize=landscape(letter),
                            rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=22,
        textColor=colors.HexColor('#082B46'),
        spaceAfter=15,
        alignment=1
    )
    sub_style = ParagraphStyle(
        'DocSub',
        parent=styles['Normal'],
        fontSize=12,
        textColor=colors.HexColor('#4B6474'),
        spaceAfter=25,
        alignment=1
    )
    
    elements = []
    elements.append(Paragraph("GlucoBloom: System Architecture & Data Flow", title_style))
    elements.append(Paragraph("Personalized Digital Twin for Continuous Glucose Forecasting • Unstop Digital Twin Challenge 2026", sub_style))
    
    # Draw Architecture Block Diagram
    d = Drawing(720, 360)
    
    # Background Canvas
    d.add(Rect(0, 0, 720, 360, fillColor=colors.HexColor('#F8FAFC'), strokeColor=colors.HexColor('#E2E8F0'), strokeWidth=1, rx=8, ry=8))
    
    # Layer 1: Data Ingestion
    d.add(Rect(20, 40, 150, 290, fillColor=colors.HexColor('#EFF6FF'), strokeColor=colors.HexColor('#3B82F6'), strokeWidth=2, rx=6, ry=6))
    d.add(String(35, 305, "1. SENSOR INGESTION", fontName="Helvetica-Bold", fontSize=11, fillColor=colors.HexColor('#1E3A8A')))
    d.add(String(35, 275, "• Continuous CGM (5m)", fontName="Helvetica", fontSize=9, fillColor=colors.HexColor('#1E293B')))
    d.add(String(35, 255, "• Carbohydrate Logs", fontName="Helvetica", fontSize=9, fillColor=colors.HexColor('#1E293B')))
    d.add(String(35, 235, "• Step / Exercise Rate", fontName="Helvetica", fontSize=9, fillColor=colors.HexColor('#1E293B')))
    d.add(String(35, 215, "• Heart Rate & HRV", fontName="Helvetica", fontSize=9, fillColor=colors.HexColor('#1E293B')))
    d.add(String(35, 195, "• Sleep Duration & Quality", fontName="Helvetica", fontSize=9, fillColor=colors.HexColor('#1E293B')))
    d.add(String(35, 155, "Data Quality Guard:", fontName="Helvetica-Bold", fontSize=9, fillColor=colors.HexColor('#B45309')))
    d.add(String(35, 135, "• Dropout Detection (>15m)", fontName="Helvetica", fontSize=8, fillColor=colors.HexColor('#475569')))
    d.add(String(35, 115, "• Compression Outlier Filter", fontName="Helvetica", fontSize=8, fillColor=colors.HexColor('#475569')))
    d.add(String(35, 95, "• Strict Time Regularization", fontName="Helvetica", fontSize=8, fillColor=colors.HexColor('#475569')))
    
    # Arrow 1 -> 2
    d.add(Line(170, 185, 200, 185, strokeColor=colors.HexColor('#3B82F6'), strokeWidth=3))
    
    # Layer 2: Feature Engineering & Preprocessing
    d.add(Rect(200, 40, 150, 290, fillColor=colors.HexColor('#F0FDF4'), strokeColor=colors.HexColor('#22C55E'), strokeWidth=2, rx=6, ry=6))
    d.add(String(215, 305, "2. FEATURE VECTOR", fontName="Helvetica-Bold", fontSize=11, fillColor=colors.HexColor('#14532D')))
    d.add(String(215, 275, "• 12 Lags: t-55m to t", fontName="Helvetica", fontSize=9, fillColor=colors.HexColor('#1E293B')))
    d.add(String(215, 255, "• 15m & 30m Velocities", fontName="Helvetica", fontSize=9, fillColor=colors.HexColor('#1E293B')))
    d.add(String(215, 235, "• Rolling Mean & Volatility", fontName="Helvetica", fontSize=9, fillColor=colors.HexColor('#1E293B')))
    d.add(String(215, 215, "• Circadian Diurnal sin/cos", fontName="Helvetica", fontSize=9, fillColor=colors.HexColor('#1E293B')))
    d.add(String(215, 195, "• Cumulative 30m Carbs", fontName="Helvetica", fontSize=9, fillColor=colors.HexColor('#1E293B')))
    d.add(String(215, 175, "• Cumulative 30m Steps", fontName="Helvetica", fontSize=9, fillColor=colors.HexColor('#1E293B')))
    d.add(String(215, 135, "Split Protocol:", fontName="Helvetica-Bold", fontSize=9, fillColor=colors.HexColor('#15803D')))
    d.add(String(215, 115, "• Strict Patient-Holdout", fontName="Helvetica", fontSize=8, fillColor=colors.HexColor('#475569')))
    d.add(String(215, 95, "• No Random Row Leakage", fontName="Helvetica", fontSize=8, fillColor=colors.HexColor('#475569')))
    
    # Arrow 2 -> 3
    d.add(Line(350, 185, 380, 185, strokeColor=colors.HexColor('#22C55E'), strokeWidth=3))
    
    # Layer 3: Two-Level AI Digital Twin Engine
    d.add(Rect(380, 40, 160, 290, fillColor=colors.HexColor('#FEF3C7'), strokeColor=colors.HexColor('#F59E0B'), strokeWidth=2, rx=6, ry=6))
    d.add(String(395, 305, "3. TWIN AI ENGINE", fontName="Helvetica-Bold", fontSize=11, fillColor=colors.HexColor('#78350F')))
    d.add(String(395, 275, "Level 1: General Model", fontName="Helvetica-Bold", fontSize=9, fillColor=colors.HexColor('#92400E')))
    d.add(String(395, 255, "• XGBoost Regressor", fontName="Helvetica", fontSize=8, fillColor=colors.HexColor('#1E293B')))
    d.add(String(395, 240, "• 160 Train Patients", fontName="Helvetica", fontSize=8, fillColor=colors.HexColor('#1E293B')))
    d.add(String(395, 225, "• 7.64 mg/dL MAE", fontName="Helvetica", fontSize=8, fillColor=colors.HexColor('#15803D')))
    d.add(String(395, 195, "Level 2: Personal Adapter", fontName="Helvetica-Bold", fontSize=9, fillColor=colors.HexColor('#92400E')))
    d.add(String(395, 175, "• Fine-tuned on User CGM", fontName="Helvetica", fontSize=8, fillColor=colors.HexColor('#1E293B')))
    d.add(String(395, 160, "• 6.44 mg/dL MAE (+26.4%)", fontName="Helvetica", fontSize=8, fillColor=colors.HexColor('#15803D')))
    d.add(String(395, 130, "Uncertainty Quantiles:", fontName="Helvetica-Bold", fontSize=9, fillColor=colors.HexColor('#92400E')))
    d.add(String(395, 110, "• 10th & 90th Percentiles", fontName="Helvetica", fontSize=8, fillColor=colors.HexColor('#1E293B')))
    d.add(String(395, 95, "• 80% Cone (85% Coverage)", fontName="Helvetica", fontSize=8, fillColor=colors.HexColor('#1E293B')))
    
    # Arrow 3 -> 4
    d.add(Line(540, 185, 570, 185, strokeColor=colors.HexColor('#F59E0B'), strokeWidth=3))
    
    # Layer 4: Clinical Interface & Decision Support
    d.add(Rect(570, 40, 130, 290, fillColor=colors.HexColor('#FAF5FF'), strokeColor=colors.HexColor('#A855F7'), strokeWidth=2, rx=6, ry=6))
    d.add(String(580, 305, "4. CLINICAL UI", fontName="Helvetica-Bold", fontSize=11, fillColor=colors.HexColor('#581C87')))
    d.add(String(580, 275, "• 30m Forecast Cone", fontName="Helvetica", fontSize=8, fillColor=colors.HexColor('#1E293B')))
    d.add(String(580, 255, "• Twin Evolution", fontName="Helvetica", fontSize=8, fillColor=colors.HexColor('#1E293B')))
    d.add(String(580, 235, "• Time Machine Replay", fontName="Helvetica", fontSize=8, fillColor=colors.HexColor('#1E293B')))
    d.add(String(580, 215, "• Clarke Error Grid", fontName="Helvetica-Bold", fontSize=8, fillColor=colors.HexColor('#15803D')))
    d.add(String(580, 200, "  (100% Zones A+B)", fontName="Helvetica", fontSize=7.5, fillColor=colors.HexColor('#15803D')))
    d.add(String(580, 175, "• Fan of Futures Sim", fontName="Helvetica", fontSize=8, fillColor=colors.HexColor('#1E293B')))
    d.add(String(580, 155, "• Food & Carb Diary", fontName="Helvetica", fontSize=8, fillColor=colors.HexColor('#1E293B')))
    d.add(String(580, 135, "• AGP Clinical Export", fontName="Helvetica", fontSize=8, fillColor=colors.HexColor('#1E293B')))
    d.add(String(580, 115, "• Fast API + Streamlit", fontName="Helvetica", fontSize=8, fillColor=colors.HexColor('#1E293B')))
    
    elements.append(d)
    doc.build(elements)
    print("Saved docs/architecture_diagram.pdf")

def create_presentation_pdf():
    pdf_path = os.path.join(DOCS_DIR, "presentation.pdf")
    doc = SimpleDocTemplate(pdf_path, pagesize=landscape(letter),
                            rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'SlideTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#082B46'),
        spaceAfter=10
    )
    subtitle_style = ParagraphStyle(
        'SlideSubtitle',
        parent=styles['Normal'],
        fontSize=13,
        textColor=colors.HexColor('#2563EB'),
        spaceAfter=20
    )
    body_style = ParagraphStyle(
        'SlideBody',
        parent=styles['Normal'],
        fontSize=11,
        textColor=colors.HexColor('#1E293B'),
        leading=16,
        spaceAfter=12
    )
    callout_style = ParagraphStyle(
        'SlideCallout',
        parent=styles['Normal'],
        fontSize=12,
        textColor=colors.HexColor('#15803D'),
        backColor=colors.HexColor('#DCFCE7'),
        borderColor=colors.HexColor('#86EFAC'),
        borderWidth=1,
        borderPadding=10,
        spaceBefore=15,
        spaceAfter=15,
        leading=16
    )

    story = []
    
    # Slide 1: Title
    story.append(Paragraph("GlucoBloom: Personalized Digital Twin for Glucose Forecasting", title_style))
    story.append(Paragraph("Digital Twin Challenge 2026 • Prototype & Code Submission", subtitle_style))
    story.append(Paragraph("<b>Healthcare Focus:</b> Non-invasive glycemic forecasting, personalized metabolic adaptation, and clinical safety.", body_style))
    story.append(Paragraph("<b>Target Users:</b> Individuals with Type 1 & Type 2 Diabetes, endocrinologists, and clinical care teams.", body_style))
    story.append(Paragraph("<b>Core Innovation:</b> Moving from retrospective glucose monitoring to prospective, uncertainty-calibrated 30-minute forward forecasting with personal digital twin evolution.", callout_style))
    story.append(PageBreak())
    
    # Slide 2: Problem Statement & Healthcare Use Case
    story.append(Paragraph("Problem Statement & Clinical Healthcare Need", title_style))
    story.append(Paragraph("Why Traditional Glucose Trackers Fail", subtitle_style))
    story.append(Paragraph("• <b>Retrospective Blindspot:</b> Commercial CGMs display where glucose was 5 minutes ago. Patients lack actionable foresight to prevent hypoglycemia (&lt;70 mg/dL) or postprandial spikes (&gt;180 mg/dL) before symptoms manifest.", body_style))
    story.append(Paragraph("• <b>Biological Non-Linearity:</b> Identical carbohydrate meals provoke drastically different glycemic excursions across individuals due to distinct insulin sensitivities, gastric emptying kinetics, and circadian phases.", body_style))
    story.append(Paragraph("• <b>Alarm Fatigue & Dangerous Over-Correction:</b> Static population models fail to adapt, frequently causing false alarms or misleading patients into insulin stacking.", body_style))
    story.append(PageBreak())
    
    # Slide 3: 2-Level Digital Twin Architecture
    story.append(Paragraph("Two-Level Digital Twin Architecture", title_style))
    story.append(Paragraph("Scalable Population Knowledge + Rapid Personal Adaptation", subtitle_style))
    story.append(Paragraph("• <b>Level 1: General Foundation Model (XGBoost Regressor):</b> Trained across 160 diabetic patients (598,960 readings) to master universal glucose kinetics, velocity, and diurnal basals.", body_style))
    story.append(Paragraph("• <b>Level 2: Personalized Digital Twin Adapter:</b> Continued gradient boosting fine-tuning on an individual's personal history, capturing unique carb sensitivities without catastrophic forgetting.", body_style))
    story.append(Paragraph("• <b>Uncertainty Quantification (Quantile Regressors):</b> Specialized models for 10th and 90th percentiles, yielding a calibrated 80% prediction interval cone.", body_style))
    story.append(PageBreak())
    
    # Slide 4: Quantitative Benchmarks & Results
    story.append(Paragraph("Empirical Results on 25 Held-Out Test Patients", title_style))
    story.append(Paragraph("Zero Patient Leakage • 93,568 Test Readings", subtitle_style))
    
    table_data = [
        ["Model / Strategy", "MAE (mg/dL)", "RMSE (mg/dL)", "Improvement vs Baseline", "Clinical Safety"],
        ["Persistence Baseline (t = t+30m)", "17.20", "29.59", "0.0% (Reference)", "High lag error"],
        ["Linear Regression Benchmark", "10.89", "15.60", "+36.7%", "Partial tracking"],
        ["XGBoost General Foundation", "7.64", "11.94", "+55.6%", "Strong accuracy"],
        ["Personalized Twin (Patient 175)", "6.44", "9.81", "+26.4% Personalization", "100.0% Zone A+B"]
    ]
    t = Table(table_data, colWidths=[200, 100, 100, 160, 110])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#082B46')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 10),
        ('BOTTOMPADDING', (0,0), (-1,0), 8),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#F8FAFC')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('FONTSIZE', (0,1), (-1,-1), 9),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F1F5F9')])
    ]))
    story.append(t)
    story.append(Spacer(1, 15))
    story.append(Paragraph("<b>Clinical Safety:</b> In Clarke Error Grid Analysis (EGA), <b>100.0% of predictions fall in Zones A + B</b> (99.6% in Zone A alone), confirming zero dangerous treatment errors.", callout_style))
    story.append(PageBreak())
    
    # Slide 5: The 5 Signature Features
    story.append(Paragraph("The Five Signature Innovations of GlucoBloom", title_style))
    story.append(Paragraph("Demo-Ready, Clinically Grounded Features", subtitle_style))
    story.append(Paragraph("1. <b>🔮 30-Minute Forecast Cone:</b> Real-time trajectory with calibrated 80% uncertainty bounds + CSV uploader.", body_style))
    story.append(Paragraph("2. <b>🧬 Digital Twin Evolution:</b> Visual audit proving error reduction from 8.74 down to 6.44 mg/dL as personal data grows.", body_style))
    story.append(Paragraph("3. <b>⏳ Glucose Time Machine & Clarke Grid:</b> Step-by-step retrospective comparison against reality with gold-standard medical validation.", body_style))
    story.append(Paragraph("4. <b>🧪 Scenario Studio ('Fan of Futures'):</b> Branching 3-path simulations (Sedentary vs 15m walk vs 30m brisk walk) + Food Diary.", body_style))
    story.append(Paragraph("5. <b>🛡️ Smart Data Quality Checker:</b> Sensor anomaly, dropout, and compression artifact detection to protect prediction confidence.", body_style))
    story.append(PageBreak())
    
    # Slide 6: Ethics, Privacy & Production Architecture
    story.append(Paragraph("Ethics, Privacy & Production Architecture", title_style))
    story.append(Paragraph("Safety First • Production Ready", subtitle_style))
    story.append(Paragraph("• <b>Local Data Sovereignty:</b> Raw clinical CGM data is never uploaded to remote cloud databases; strictly git-ignored.", body_style))
    story.append(Paragraph("• <b>De-Identification & Zero PHI:</b> Only anonymized physiological telemetry is processed.", body_style))
    story.append(Paragraph("• <b>Feature Trust Badges:</b> Every screen explicitly badged 🟢 WORKING, 🟡 DEMONSTRATION, or ⚪ FUTURE.", body_style))
    story.append(Paragraph("• <b>Dual Architecture:</b> High-performance FastAPI REST API on port 8000 + interactive Streamlit suite on port 8501.", body_style))
    
    doc.build(story)
    print("Saved docs/presentation.pdf")

if __name__ == "__main__":
    create_architecture_diagram()
    create_presentation_pdf()
