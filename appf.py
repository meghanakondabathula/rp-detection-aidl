import os
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from torchvision.models import efficientnet_b4, vit_b_16
from flask import Flask, request, render_template, redirect, url_for, flash, session, send_file, make_response, jsonify
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from PIL import Image
from flask_cors import CORS
import cv2
import io
import requests
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from models import db, User, PredictionResult
from translations import translations, languages
# For PDF charts
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64


def generate_timeline_chart(predictions_asc):
    """Generate timeline chart and return as base64 encoded image"""
    if not predictions_asc or len(predictions_asc) < 2:
        return None
    
    plt.figure(figsize=(10, 4))
    dates = [p.created_at.strftime('%Y-%m-%d') for p in predictions_asc]
    values = [0 if p.prediction == 'Healthy' else 1 for p in predictions_asc]
    
    plt.plot(dates, values, marker='o', color='#dc3545', linewidth=2, markersize=8)
    plt.ylim(-0.2, 1.2)
    plt.yticks([0, 1], ['Healthy', 'RP'])
    plt.xlabel('Date')
    plt.ylabel('Health Status')
    plt.title('Health Trend Over Time')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    chart_b64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    return chart_b64


def generate_monthly_chart(predictions_asc):
    """Generate monthly breakdown chart and return as base64 encoded image"""
    if not predictions_asc:
        return None
    
    monthly_data = {}
    for p in predictions_asc:
        month_key = p.created_at.strftime('%Y-%m')
        if month_key not in monthly_data:
            monthly_data[month_key] = {'healthy': 0, 'rp': 0}
        if p.prediction == "Healthy":
            monthly_data[month_key]['healthy'] += 1
        else:
            monthly_data[month_key]['rp'] += 1
    
    if not monthly_data:
        return None
    
    plt.figure(figsize=(10, 4))
    months = sorted(monthly_data.keys())
    healthy_vals = [monthly_data[m]['healthy'] for m in months]
    rp_vals = [monthly_data[m]['rp'] for m in months]
    
    x = range(len(months))
    plt.bar(x, healthy_vals, label='Healthy', color='#28a745', width=0.4)
    plt.bar([i + 0.4 for i in x], rp_vals, label='RP Detected', color='#dc3545', width=0.4)
    plt.xticks([i + 0.2 for i in x], months, rotation=45)
    plt.xlabel('Month')
    plt.ylabel('Number of Tests')
    plt.title('Monthly Breakdown')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    chart_b64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    return chart_b64


def calculate_tracking_data(predictions_asc, predictions):
    """Calculate tracking data for disease progression"""
    rp_predictions = [p for p in predictions_asc if p.prediction == "Retinitis Pigmentosa" and p.stage is not None]
    
    if not predictions:
        return None, None, None
    
    # Get latest prediction
    latest_prediction = predictions[0] if predictions else None
    
    progression_status = "healthy"
    progression_message = "No RP detected - Healthy"
    
    if latest_prediction and latest_prediction.prediction == "Healthy":
        progression_status = "healthy"
        progression_message = "Latest scan shows a healthy retina. If RP was previously detected, clinical verification is recommended."
    elif len(rp_predictions) >= 2:
        latest_rp = rp_predictions[-1]
        previous_rp = rp_predictions[-2]
        stage_order = {"Early": 1, "Moderate": 2, "Severe": 3}
        latest_stage_val = stage_order.get(latest_rp.stage, 0)
        previous_stage_val = stage_order.get(previous_rp.stage, 0)
        
        if latest_stage_val > previous_stage_val:
            progression_status = "worsening"
            progression_message = f"Condition showing progression from {previous_rp.stage} to {latest_rp.stage}"
        elif latest_stage_val < previous_stage_val:
            progression_status = "improving"
            progression_message = f"Condition showing improvement from {previous_rp.stage} to {latest_rp.stage}"
        else:
            progression_status = "stable"
            progression_message = f"Condition remains stable at {latest_rp.stage} stage"
    elif len(rp_predictions) == 1:
        progression_status = "monitoring"
        progression_message = f"First RP detection - {rp_predictions[0].stage} stage. Regular monitoring recommended."
    
    # Stage progression
    stage_progression = []
    for p in rp_predictions:
        stage_progression.append({'date': p.created_at.strftime('%Y-%m-%d'), 'stage': p.stage})
    
    return progression_status, progression_message, stage_progression


def get_translations(lang_code):
    return translations.get(lang_code, translations['en'])

class HybridModel(nn.Module):
    def __init__(self, num_classes=2):
        super(HybridModel, self).__init__()
        self.effnet = efficientnet_b4(weights='DEFAULT')
        self.effnet.classifier = nn.Identity()
        self.vit = vit_b_16(weights='DEFAULT')
        self.vit.heads = nn.Identity()
        combined_dim = 1792 + 768
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        f1 = self.effnet(x)
        f2 = self.vit(x)
        combined = torch.cat((f1, f2), dim=1)
        return self.classifier(combined)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classes = ["Healthy", "Retinitis Pigmentosa"]
model = HybridModel(num_classes=len(classes)).to(device)
model.load_state_dict(torch.load("hybrid_model.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def generate_gradcam(model, img_tensor):
    model.eval()
    features, gradients = [], []
    def forward_hook(module, input, output):
        features.append(output)
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])
    last_conv = model.effnet.features[-1]
    handle_fw = last_conv.register_forward_hook(forward_hook)
    handle_bw = last_conv.register_backward_hook(backward_hook)
    output = model(img_tensor)
    pred_class = output.argmax(dim=1).item()
    model.zero_grad()
    output[0, pred_class].backward()
    grads = gradients[0].mean(dim=[2, 3], keepdim=True)
    fmap = features[0]
    cam = (grads * fmap).sum(dim=1).squeeze().detach().cpu().numpy()
    cam = np.maximum(cam, 0)
    cam = cam / np.max(cam)
    handle_fw.remove()
    handle_bw.remove()
    return cam

def estimate_stage(heatmap):
    normalized = heatmap / np.max(heatmap)
    activation_ratio = np.sum(normalized > 0.30) / normalized.size
    if activation_ratio < 0.15:
        return "Early"
    elif activation_ratio < 0.30:
        return "Moderate"
    else:
        return "Severe"

def is_retinal_image(img_path):
    filename = os.path.basename(img_path).lower()
    keywords = ["retina", "fundus", "uwf", "fa", "faf", "eye"]
    if any(k in filename for k in keywords):
        return True
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 7)
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, 100, param1=50, param2=30, minRadius=50, maxRadius=300)
    return circles is not None

app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = 'your-secret-key-change-in-production'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///rp_detection.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)
CORS(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/set_language/<lang_code>")
def set_language(lang_code):
    session['language'] = lang_code
    return redirect(request.referrer or url_for("welcome"))

@app.route("/")
def welcome():
    lang = session.get('language', 'en')
    t = get_translations(lang)
    return render_template("welcome.html", t=t, languages=languages, current_lang=lang, user=current_user if current_user.is_authenticated else None)

@app.route("/index")
@login_required
def index():
    lang = session.get('language', 'en')
    t = get_translations(lang)
    return render_template("index.html", t=t, languages=languages, current_lang=lang, user=current_user)

@app.route("/register", methods=["GET", "POST"])
def register():
    lang = session.get('language', 'en')
    t = get_translations(lang)
    if request.method == "POST":
        username = request.form.get("username")
        email = request.form.get("email")
        password = request.form.get("password")
        if User.query.filter_by(username=username).first():
            flash("Username already exists!", "error")
            return redirect(url_for("register"))
        if User.query.filter_by(email=email).first():
            flash("Email already registered!", "error")
            return redirect(url_for("register"))
        new_user = User(username=username, email=email)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()
        flash("Registration successful! Please login.", "success")
        return redirect(url_for("login"))
    return render_template("register.html", t=t, languages=languages, current_lang=lang)

@app.route("/login", methods=["GET", "POST"])
def login():
    lang = session.get('language', 'en')
    t = get_translations(lang)
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user)
            flash("Login successful!", "success")
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid username or password!", "error")
    return render_template("login.html", t=t, languages=languages, current_lang=lang)

@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You have been logged out.", "info")
    return redirect(url_for("welcome"))

@app.route("/dashboard")
@login_required
def dashboard():
    lang = session.get('language', 'en')
    t = get_translations(lang)
    predictions = PredictionResult.query.filter_by(user_id=current_user.id).order_by(PredictionResult.created_at.desc()).all()
    predictions_asc = PredictionResult.query.filter_by(user_id=current_user.id).order_by(PredictionResult.created_at.asc()).all()
    totalTests = len(predictions)
    healthy_count = sum(1 for p in predictions if p.prediction == "Healthy")
    rp_count = sum(1 for p in predictions if p.prediction == "Retinitis Pigmentosa")
    early_count = sum(1 for p in predictions if p.prediction == "Retinitis Pigmentosa" and p.stage == "Early")
    moderate_count = sum(1 for p in predictions if p.prediction == "Retinitis Pigmentosa" and p.stage == "Moderate")
    severe_count = sum(1 for p in predictions if p.prediction == "Retinitis Pigmentosa" and p.stage == "Severe")
    stats = {"total_tests": totalTests, "healthy_count": healthy_count, "rp_count": rp_count, "early_count": early_count, "moderate_count": moderate_count, "severe_count": severe_count}
    timeline_data = []
    for p in predictions_asc:
        timeline_data.append({'date': p.created_at.strftime('%Y-%m-%d'), 'prediction': p.prediction, 'stage': p.stage, 'confidence': p.confidence})
    rp_predictions = [p for p in predictions_asc if p.prediction == "Retinitis Pigmentosa" and p.stage is not None]
    stage_progression = []
    for p in rp_predictions:
        stage_progression.append({'date': p.created_at.strftime('%Y-%m-%d'), 'stage': p.stage})
    monthly_data = {}
    for p in predictions_asc:
        month_key = p.created_at.strftime('%Y-%m')
        if month_key not in monthly_data:
            monthly_data[month_key] = {'healthy': 0, 'rp': 0}
        if p.prediction == "Healthy":
            monthly_data[month_key]['healthy'] += 1
        else:
            monthly_data[month_key]['rp'] += 1
    monthly_trend = []
    for month in sorted(monthly_data.keys()):
        monthly_trend.append({'month': month, 'healthy': monthly_data[month]['healthy'], 'rp': monthly_data[month]['rp']})
        progression_status = "stable"
    progression_message = "No RP detected yet"

    # check latest overall prediction
    latest_prediction = predictions[0] if predictions else None

    if latest_prediction and latest_prediction.prediction == "Healthy":
        progression_status = "healthy"
        progression_message = "Latest scan shows a healthy retina. If RP was previously detected, clinical verification is recommended."

    elif len(rp_predictions) >= 2:
        latest_rp = rp_predictions[-1]
        previous_rp = rp_predictions[-2]

        stage_order = {"Early": 1, "Moderate": 2, "Severe": 3}

        latest_stage_val = stage_order.get(latest_rp.stage, 0)
        previous_stage_val = stage_order.get(previous_rp.stage, 0)

        if latest_stage_val > previous_stage_val:
            progression_status = "worsening"
            progression_message = f"Condition showing progression from {previous_rp.stage} to {latest_rp.stage}"

        elif latest_stage_val < previous_stage_val:
            progression_status = "improving"
            progression_message = f"Condition showing improvement from {previous_rp.stage} to {latest_rp.stage}"

        else:
            progression_status = "stable"
            progression_message = f"Condition remains stable at {latest_rp.stage} stage"

    elif len(rp_predictions) == 1:
        progression_status = "monitoring"
        progression_message = f"First RP detection - {rp_predictions[0].stage} stage. Regular monitoring recommended."

    tracking_data = {
        'timeline_data': timeline_data,
        'stage_progression': stage_progression,
        'monthly_trend': monthly_trend,
        'progression_status': progression_status,
        'progression_message': progression_message,
        'has_rp_history': len(rp_predictions) > 0,
        'total_rp_tests': len(rp_predictions)
    }

    return render_template(
        "dashboard.html",
        predictions=predictions,
        stats=stats,
        user=current_user,
        t=t,
        languages=languages,
        current_lang=lang,
        tracking_data=tracking_data
    )
def generate_pdf_report(prediction, user, gradcam_path=None):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    story = []
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=20, textColor=colors.HexColor('#1a237e'), alignment=TA_CENTER, spaceAfter=20)
    heading_style = ParagraphStyle('CustomHeading', parent=styles['Heading2'], fontSize=12, textColor=colors.HexColor('#0d47a1'), spaceBefore=15, spaceAfter=10)
    normal_style = ParagraphStyle('CustomNormal', parent=styles['Normal'], fontSize=10, spaceAfter=5)
    story.append(Paragraph("RETINITIS PIGMENTOSA DETECTION REPORT", title_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph("PATIENT INFORMATION", heading_style))
    patient_data = [['Patient Name:', user.username], ['Email:', user.email], ['Report Date:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')]]
    patient_table = Table(patient_data, colWidths=[2*inch, 4*inch])
    patient_table.setStyle(TableStyle([('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e3f2fd')), ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'), ('FONTNAME', (1, 0), (1, -1), 'Helvetica'), ('FONTSIZE', (0, 0), (-1, -1), 10), ('TEXTCOLOR', (0, 0), (-1, -1), colors.black), ('BOTTOMPADDING', (0, 0), (-1, -1), 8), ('TOPPADDING', (0, 0), (-1, -1), 8)]))
    story.append(patient_table)
    story.append(Paragraph("TEST DETAILS", heading_style))
    test_data = [['Test ID:', str(prediction.id)], ['Test Date:', prediction.created_at.strftime('%Y-%m-%d %H:%M:%S')]]
    test_table = Table(test_data, colWidths=[2*inch, 4*inch])
    test_table.setStyle(TableStyle([('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e8f5e9')), ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'), ('FONTNAME', (1, 0), (1, -1), 'Helvetica'), ('FONTSIZE', (0, 0), (-1, -1), 10), ('TEXTCOLOR', (0, 0), (-1, -1), colors.black), ('BOTTOMPADDING', (0, 0), (-1, -1), 8), ('TOPPADDING', (0, 0), (-1, -1), 8)]))
    story.append(test_table)
    story.append(Paragraph("ANALYSIS RESULTS", heading_style))
    result_color = '#ffcdd2' if prediction.prediction == "Retinitis Pigmentosa" else '#c8e6c9'
    result_data = [['Prediction:', prediction.prediction], ['Confidence:', prediction.confidence], ['Disease Stage:', prediction.stage if prediction.stage else 'N/A']]
    result_table = Table(result_data, colWidths=[2*inch, 4*inch])
    result_table.setStyle(TableStyle([('BACKGROUND', (0, 0), (0, -1), colors.HexColor(result_color)), ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'), ('FONTNAME', (1, 0), (1, -1), 'Helvetica'), ('FONTSIZE', (0, 0), (-1, -1), 10), ('TEXTCOLOR', (0, 0), (-1, -1), colors.black), ('BOTTOMPADDING', (0, 0), (-1, -1), 8), ('TOPPADDING', (0, 0), (-1, -1), 8)]))
    story.append(result_table)
    if gradcam_path and os.path.exists(gradcam_path):
        story.append(Spacer(1, 15))
        story.append(Paragraph("GRAD-CAM VISUALIZATION", heading_style))
        story.append(Spacer(1, 5))
        try:
            img = RLImage(gradcam_path, width=3*inch, height=3*inch)
            story.append(img)
        except Exception as e:
            story.append(Paragraph(f"Could not load Grad-CAM image: {str(e)}", normal_style))
    story.append(Spacer(1, 15))
    story.append(Paragraph("MEDICAL RECOMMENDATION", heading_style))
    if prediction.prediction == "Retinitis Pigmentosa":
        if prediction.stage == "Early":
            recommendations = "<b>Status: EARLY STAGE DETECTION</b><br/><br/>Recommendations:<br/>1. Schedule regular eye examinations every 6 months<br/>2. Consider genetic counseling<br/>3. Monitor visual field changes<br/>4. Discuss low vision aids with specialist<br/>5. Consider vitamin A supplementation (consult doctor first)"
        elif prediction.stage == "Moderate":
            recommendations = "<b>Status: MODERATE STAGE DETECTION</b><br/><br/>Recommendations:<br/>1. Consult with retinal specialist immediately<br/>2. Consider vitamin A supplementation (consult doctor)<br/>3. Begin low vision rehabilitation<br/>4. Regular visual field testing<br/>5. Discuss dietary modifications with nutritionist"
        else:
            recommendations = "<b>Status: SEVERE STAGE DETECTION</b><br/><br/>Recommendations:<br/>1. URGENT: Consult with retinal specialist<br/>2. Consider surgical options if applicable<br/>3. Begin comprehensive low vision program<br/>4. Psychological support recommended<br/>5. Explore assistive technology devices"
    else:
        recommendations = "<b>Status: HEALTHY</b><br/><br/>Recommendations:<br/>1. Continue regular eye checkups<br/>2. Maintain healthy lifestyle<br/>3. Protect eyes from UV exposure<br/>4. Follow proper eye hygiene practices"
    story.append(Paragraph(recommendations, normal_style))
    story.append(Spacer(1, 30))
    footer_text = " Retinitis Pigmentosa Detection System"
    story.append(Paragraph(footer_text, ParagraphStyle('Footer', parent=styles['Normal'], fontSize=8, alignment=TA_CENTER, textColor=colors.gray)))
    doc.build(story)
    buffer.seek(0)
    return buffer

def generate_dashboard_pdf(user, predictions, stats, predictions_asc=None, timeline_chart_b64=None, monthly_chart_b64=None, progression_status=None, progression_message=None, stage_progression=None):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    story = []
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=20, textColor=colors.HexColor('#1a237e'), alignment=TA_CENTER, spaceAfter=20)
    heading_style = ParagraphStyle('CustomHeading', parent=styles['Heading2'], fontSize=12, textColor=colors.HexColor('#0d47a1'), spaceBefore=15, spaceAfter=10)
    normal_style = ParagraphStyle('CustomNormal', parent=styles['Normal'], fontSize=10, spaceAfter=5)
    story.append(Paragraph("PATIENT DASHBOARD REPORT", title_style))
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ParagraphStyle('SubTitle', parent=styles['Normal'], fontSize=10, alignment=TA_CENTER)))
    story.append(Spacer(1, 10))
    story.append(Paragraph("PATIENT INFORMATION", heading_style))
    patient_data = [['Patient Name:', user.username], ['Email:', user.email], ['Total Tests:', str(stats['total_tests'])]]
    patient_table = Table(patient_data, colWidths=[2*inch, 4*inch])
    patient_table.setStyle(TableStyle([('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e3f2fd')), ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'), ('FONTNAME', (1, 0), (1, -1), 'Helvetica'), ('FONTSIZE', (0, 0), (-1, -1), 10), ('TEXTCOLOR', (0, 0), (-1, -1), colors.black), ('BOTTOMPADDING', (0, 0), (-1, -1), 8), ('TOPPADDING', (0, 0), (-1, -1), 8)]))
    story.append(patient_table)
    story.append(Paragraph("STATISTICS SUMMARY", heading_style))
    stats_data = [['Total Tests', 'Healthy', 'RP Detected', 'Early', 'Moderate', 'Severe'], [str(stats['total_tests']), str(stats['healthy_count']), str(stats['rp_count']), str(stats['early_count']), str(stats['moderate_count']), str(stats['severe_count'])]]
    stats_table = Table(stats_data, colWidths=[1.2*inch, 1.2*inch, 1.2*inch, 1.2*inch, 1.2*inch, 1.2*inch])
    stats_table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0d47a1')), ('TEXTCOLOR', (0, 0), (-1, 0), colors.white), ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'), ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#e3f2fd')), ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'), ('FONTSIZE', (0, 0), (-1, -1), 9), ('TEXTCOLOR', (0, 0), (-1, -1), colors.black), ('BOTTOMPADDING', (0, 0), (-1, -1), 8), ('TOPPADDING', (0, 0), (-1, -1), 8), ('ALIGN', (0, 0), (-1, -1), 'CENTER')]))
    story.append(stats_table)
    
    # Add Tracking Message Section
    if progression_status and progression_message:
        story.append(Spacer(1, 15))
        story.append(Paragraph("DISEASE TRACKING", heading_style))
        status_colors = {'stable': '#28a745', 'worsening': '#dc3545', 'improving': '#28a745', 'monitoring': '#ffc107', 'healthy': '#28a745'}
        status_labels = {'stable': 'Stable', 'worsening': 'Worsening', 'improving': 'Improving', 'monitoring': 'Monitoring', 'healthy': 'Healthy'}
        status_color = status_colors.get(progression_status, '#ffc107')
        status_label = status_labels.get(progression_status, 'Monitoring')
        tracking_data = [['Status:', status_label], ['Message:', progression_message]]
        tracking_table = Table(tracking_data, colWidths=[1.5*inch, 4.5*inch])
        tracking_table.setStyle(TableStyle([('BACKGROUND', (0, 0), (0, -1), colors.HexColor(status_color)), ('TEXTCOLOR', (0, 0), (0, -1), colors.white), ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'), ('FONTNAME', (1, 0), (1, -1), 'Helvetica'), ('FONTSIZE', (0, 0), (-1, -1), 10), ('TEXTCOLOR', (1, 0), (1, -1), colors.black), ('BOTTOMPADDING', (0, 0), (-1, -1), 10), ('TOPPADDING', (0, 0), (-1, -1), 10), ('BACKGROUND', (1, 0), (1, -1), colors.HexColor('#f8f9fa'))]))
        story.append(tracking_table)
        if stage_progression:
            story.append(Spacer(1, 10))
            story.append(Paragraph("Stage Progression:", normal_style))
            stage_data = [['Date', 'Stage']]
            for stage in stage_progression:
                stage_data.append([stage['date'], stage['stage']])
            if len(stage_data) > 1:
                stage_table = Table(stage_data, colWidths=[2*inch, 2*inch])
                stage_table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0d47a1')), ('TEXTCOLOR', (0, 0), (-1, 0), colors.white), ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'), ('BACKGROUND', (0, 1), (-1, -1), colors.white), ('FONTSIZE', (0, 0), (-1, -1), 9), ('TEXTCOLOR', (0, 0), (-1, -1), colors.black), ('BOTTOMPADDING', (0, 0), (-1, -1), 6), ('TOPPADDING', (0, 0), (-1, -1), 6), ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)]))
                story.append(stage_table)
    
    # Add Charts
    if timeline_chart_b64:
        story.append(Spacer(1, 15))
        story.append(Paragraph("HEALTH TREND OVER TIME", heading_style))
        try:
            img_data = base64.b64decode(timeline_chart_b64)
            img_buffer = io.BytesIO(img_data)
            img = RLImage(img_buffer, width=5*inch, height=2.5*inch)
            story.append(img)
        except Exception as e:
            story.append(Paragraph(f"Could not load timeline chart: {str(e)}", normal_style))
    
    if monthly_chart_b64:
        story.append(Spacer(1, 15))
        story.append(Paragraph("MONTHLY BREAKDOWN", heading_style))
        try:
            img_data = base64.b64decode(monthly_chart_b64)
            img_buffer = io.BytesIO(img_data)
            img = RLImage(img_buffer, width=5*inch, height=2.5*inch)
            story.append(img)
        except Exception as e:
            story.append(Paragraph(f"Could not load monthly chart: {str(e)}", normal_style))
    
    story.append(Paragraph("PREDICTION HISTORY", heading_style))
    if predictions:
        history_data = [['Date', 'Prediction', 'Confidence', 'Stage']]
        for pred in predictions[:20]:
            history_data.append([pred.created_at.strftime('%Y-%m-%d %H:%M'), pred.prediction, pred.confidence, pred.stage if pred.stage else 'N/A'])
        history_table = Table(history_data, colWidths=[1.8*inch, 2*inch, 1.2*inch, 1.2*inch])
        history_table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0d47a1')), ('TEXTCOLOR', (0, 0), (-1, 0), colors.white), ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'), ('BACKGROUND', (0, 1), (-1, -1), colors.white), ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'), ('FONTSIZE', (0, 0), (-1, -1), 8), ('TEXTCOLOR', (0, 0), (-1, -1), colors.black), ('BOTTOMPADDING', (0, 0), (-1, -1), 6), ('TOPPADDING', (0, 0), (-1, -1), 6), ('ALIGN', (0, 0), (-1, -1), 'LEFT'), ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)]))
        story.append(history_table)
        if len(predictions) > 20:
            story.append(Spacer(1, 10))
            story.append(Paragraph(f"Note: Showing last 20 of {len(predictions)} predictions", ParagraphStyle('Note', parent=styles['Normal'], fontSize=8, textColor=colors.gray)))
    else:
        story.append(Paragraph("No predictions found.", normal_style))
    story.append(Spacer(1, 30))
    footer_text = "<br/><br/><b>This is an automated report generated by RP Detection System using AI technology.</b><br/><i>Consult with healthcare professionals for medical advice.</i><br/><br/>© 2025 Retinitis Pigmentosa Detection System"
    story.append(Paragraph(footer_text, ParagraphStyle('Footer', parent=styles['Normal'], fontSize=8, alignment=TA_CENTER, textColor=colors.gray)))
    doc.build(story)
    buffer.seek(0)
    return buffer

@app.route("/download_report/<int:prediction_id>")
@login_required
def download_report(prediction_id):
    prediction = PredictionResult.query.filter_by(id=prediction_id, user_id=current_user.id).first()
    if not prediction:
        flash("Report not found!", "error")
        return redirect(url_for("dashboard"))
    gradcam_path = prediction.gradcam_path if prediction.prediction == "Retinitis Pigmentosa" else None
    pdf_buffer = generate_pdf_report(prediction, current_user, gradcam_path)
    response = make_response(pdf_buffer.getvalue())
    response.headers['Content-Type'] = 'application/pdf'
    filename = f"RP_Report_{prediction.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    response.headers['Content-Disposition'] = f'attachment; filename={filename}'
    return response

@app.route("/download_dashboard")
@login_required
def download_dashboard():
    # Get predictions in both orders
    predictions = PredictionResult.query.filter_by(user_id=current_user.id).order_by(PredictionResult.created_at.desc()).all()
    predictions_asc = PredictionResult.query.filter_by(user_id=current_user.id).order_by(PredictionResult.created_at.asc()).all()
    
    # Calculate statistics
    totalTests = len(predictions)
    healthy_count = sum(1 for p in predictions if p.prediction == "Healthy")
    rp_count = sum(1 for p in predictions if p.prediction == "Retinitis Pigmentosa")
    early_count = sum(1 for p in predictions if p.prediction == "Retinitis Pigmentosa" and p.stage == "Early")
    moderate_count = sum(1 for p in predictions if p.prediction == "Retinitis Pigmentosa" and p.stage == "Moderate")
    severe_count = sum(1 for p in predictions if p.prediction == "Retinitis Pigmentosa" and p.stage == "Severe")
    stats = {"total_tests": totalTests, "healthy_count": healthy_count, "rp_count": rp_count, "early_count": early_count, "moderate_count": moderate_count, "severe_count": severe_count}
    
    # Generate charts
    timeline_chart_b64 = generate_timeline_chart(predictions_asc)
    monthly_chart_b64 = generate_monthly_chart(predictions_asc)
    
    # Calculate tracking data
    progression_status, progression_message, stage_progression = calculate_tracking_data(predictions_asc, predictions)
    
    # Generate PDF with charts and tracking
    pdf_buffer = generate_dashboard_pdf(
        current_user, 
        predictions, 
        stats,
        predictions_asc=predictions_asc,
        timeline_chart_b64=timeline_chart_b64,
        monthly_chart_b64=monthly_chart_b64,
        progression_status=progression_status,
        progression_message=progression_message,
        stage_progression=stage_progression
    )
    
    response = make_response(pdf_buffer.getvalue())
    response.headers['Content-Type'] = 'application/pdf'
    filename = f"RP_Dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    response.headers['Content-Disposition'] = f'attachment; filename={filename}'
    return response

@app.route("/download_latest_report")
@login_required
def download_latest_report():
    prediction = PredictionResult.query.filter_by(user_id=current_user.id).order_by(PredictionResult.created_at.desc()).first()
    if not prediction:
        flash("No predictions found!", "error")
        return redirect(url_for("dashboard"))
    return redirect(url_for('download_report', prediction_id=prediction.id))

@app.route("/predict", methods=["POST"])
@login_required
def predict():
    lang = session.get('language', 'en')
    t = get_translations(lang)
    if "file" not in request.files:
        return render_template("result.html", t=t, prediction="No file uploaded", confidence="N/A", user=current_user if current_user.is_authenticated else None, languages=languages, current_lang=lang)
    file = request.files["file"]
    if file.filename == "":
        return render_template("result.html", t=t, prediction="No file selected", confidence="N/A", user=current_user if current_user.is_authenticated else None, languages=languages, current_lang=lang)
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    if not is_retinal_image(file_path):
        return render_template("result.html", t=t, prediction="⚠️ Please upload a retinal image (UWF, Fundus, or FAF scan).", confidence="N/A", stage=None, gradcam=None, user=current_user if current_user.is_authenticated else None, languages=languages, current_lang=lang)
    try:
        img = Image.open(file_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)
        label = classes[pred.item()]
        confidence = f"{conf.item() * 100:.2f}%"
        stage = None
        gradcam_path = None
        if label == "Retinitis Pigmentosa":
            heatmap = generate_gradcam(model, img_tensor)
            stage = estimate_stage(heatmap)
            img_np = np.array(img.resize((224, 224)))
            heatmap_resized = cv2.resize(heatmap, (224, 224))
            heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(heatmap_color, 0.5, img_np, 0.5, 0)
            gradcam_filename = f"gradcam_{file.filename}"
            gradcam_path = os.path.join("static", "uploads", gradcam_filename)
            cv2.imwrite(gradcam_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        
        saved_prediction_id = None
        if current_user.is_authenticated:
            prediction_result = PredictionResult(
                user_id=current_user.id,
                prediction=label,
                confidence=confidence,
                stage=stage,
                gradcam_path=gradcam_path,
                image_filename=file.filename
            )
            db.session.add(prediction_result)
            db.session.commit()
            saved_prediction_id = prediction_result.id
        return render_template("result.html", t=t, prediction=label, confidence=confidence, stage=stage, gradcam=gradcam_path, user=current_user if current_user.is_authenticated else None, languages=languages, current_lang=lang, prediction_id=saved_prediction_id)
    except Exception as e:
        return render_template("result.html", t=t, prediction="Error", confidence=str(e), user=current_user if current_user.is_authenticated else None, languages=languages, current_lang=lang)

GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_HERE"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"

SYSTEM_PROMPT = """You are an AI medical assistant that ONLY answers questions related to Retinitis and Retinitis Pigmentosa (RP). Rules: * Answer ONLY retinal disease related questions * If off-topic, respond: 'I am designed to answer only questions related to Retinitis and retinal diseases.'"""

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "").strip()
    if not user_message:
        return jsonify({"error": "No message provided"}), 400
    chat_history = session.get("chat_history", [])
    messages = [{"role": "user", "parts": [{"text": SYSTEM_PROMPT}]}]
    for msg in chat_history[-5:]:
        messages.append(msg)
    messages.append({"role": "user", "parts": [{"text": user_message}]})
    try:
        payload = {"contents": messages, "generationConfig": {"temperature": 0.7, "maxOutputTokens": 500}}
        params = {"key": GEMINI_API_KEY}
        response = requests.post(GEMINI_API_URL, params=params, json=payload, timeout=30)
        if response.status_code == 200:
            result = response.json()
            bot_response = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            if not bot_response:
                bot_response = "I apologize, but I couldn't generate a response. Please try again."
        else:
            bot_response = get_fallback_response(user_message)
    except Exception as e:
        print(f"Chat API Error: {e}")
        bot_response = get_fallback_response(user_message)
    chat_history.append({"role": "user", "parts": [{"text": user_message}]})
    chat_history.append({"role": "model", "parts": [{"text": bot_response}]})
    session["chat_history"] = chat_history[-10:]
    return jsonify({"response": bot_response})

def get_fallback_response(user_message):
    user_message_lower = user_message.lower()
    off_topic_keywords = ["weather", "sports", "politics", "music", "movie", "recipe", "stock", "crypto", "bitcoin", "news", "celebrity", "game"]
    for keyword in off_topic_keywords:
        if keyword in user_message_lower:
            return "I am designed to answer only questions related to Retinitis and retinal diseases. Please ask a relevant question."
    rp_responses = {
        "symptoms": "Common symptoms of Retinitis Pigmentosa include: • Night blindness (nyctalopia) • Peripheral vision loss (tunnel vision) • Difficulty seeing in low light • Gradual loss of central vision",
        "cause": "Retinitis Pigmentosa is primarily a genetic disorder caused by mutations in various genes (over 60 genes identified).",
        "diagnosis": "RP is diagnosed through: • Eye Scanning images like Colour fundus, FAF, Ultrs-Wield Field Images  • Genetic testing • Visual field testing  ",
        "treatment": "Currently, there is no cure for RP. But yes, researchers are exploring new treatments such as gene therapy, stem cell therapy, retinal implants, and neuroprotective drugs to slow or potentially reverse vision loss caused by Retinitis Pigmentosa.",
        "progression": "RP typically progresses slowly over years: • Early: Night blindness, peripheral vision loss • Moderate: Tunnel vision • Late: Significant vision loss",
        "inheritance": "RP can be inherited in multiple ways: • Autosomal dominant (most common) • Autosomal recessive • X-linked",
        "food": "Retinitis Pigmentosa (RP) is a genetic disorder, meaning it is mainly caused by inherited gene mutations. So:❗ There is no specific food that can completely prevent RP.However, certain nutritious food may help like: Vitamin A, Omega Fatty Acids, Vitamin C, Vitamin E, Zinc.",
        "risk_factors": "Risk factors for Retinitis Pigmentosa include: • Family history of RP • Inherited genetic mutations • Consanguineous marriages (marriage within close relatives) • Certain rare syndromes associated with RP.",
        "early_signs": "Early signs of Retinitis Pigmentosa may include: • Difficulty seeing at night • Trouble adjusting from bright light to dark environments • Reduced side vision • Frequent tripping over objects in dim light.",
        "stages": "Retinitis Pigmentosa progresses in stages: • Early Stage: Night blindness and mild peripheral vision loss • Middle Stage: Tunnel vision and reduced light sensitivity • Advanced Stage: Significant loss of vision and difficulty recognizing faces.",
        "prevention": "Since Retinitis Pigmentosa is a genetic condition, complete prevention is not possible. However: • Genetic counseling can help families understand risks • Early eye examinations may help detect the disease earlier • Maintaining a healthy lifestyle may support overall eye health.",
        "lifestyle": "People with Retinitis Pigmentosa can improve daily life by: • Using brighter lighting indoors • Wearing UV-protective sunglasses outdoors • Using mobility aids if needed • Regular eye checkups with an ophthalmologist.",
        "assistive_technology": "Assistive technologies for RP patients include: • Screen readers for digital devices • Magnification tools • Smart glasses for vision assistance • Mobile apps designed for visually impaired users.",
        "doctor_visit": "You should consult an eye specialist if you notice: • Sudden difficulty seeing at night • Gradual narrowing of your side vision • Trouble adapting to dark environments • Any unexplained changes in your vision.",
        "normal_vs_rp": "Normal vision allows a wide field of view and good night vision. In Retinitis Pigmentosa: • Peripheral vision gradually narrows • Night vision becomes poor • Vision may eventually become tunnel-like.",
        "contagious": "Retinitis Pigmentosa is NOT contagious. It cannot spread from one person to another. It is an inherited genetic disorder caused by mutations in specific genes.",
        "age_onset": "Retinitis Pigmentosa symptoms usually begin during childhood, teenage years, or early adulthood. However, the age of onset may vary depending on the genetic mutation involved.",
        "blindness": "In advanced stages, Retinitis Pigmentosa can lead to severe vision loss. However, many individuals retain some central vision for many years with proper care and monitoring.",
        "future_treatment": "Research for Retinitis Pigmentosa is ongoing. Potential future treatments include: • Gene therapy • Stem cell therapy • Retinal implants • Advanced drug therapies.",
        "support": "Living with Retinitis Pigmentosa can be challenging. Support options include: • Counseling services • Vision rehabilitation programs • Support groups for visually impaired individuals.",
        "driving": "People with Retinitis Pigmentosa may find driving difficult due to reduced peripheral and night vision. Regular eye examinations help determine whether driving is safe.",
        "daily_life": "Many people with Retinitis Pigmentosa lead productive lives by adapting their environment with proper lighting, assistive technologies, and mobility training.",
        "retina_damage": "Retinitis Pigmentosa is a degenerative disease that affects the retina, the light-sensitive layer at the back of the eye. In RP, photoreceptor cells called rods gradually die first. Rod cells are responsible for night vision and peripheral vision. As the disease progresses, cone cells, which control central and color vision, may also become damaged. This leads to gradual narrowing of the visual field and reduced visual clarity.",
        "photoreceptors": "The retina contains two main types of photoreceptor cells: rods and cones. Rod cells help detect light in low-light conditions and allow us to see at night. Cone cells help detect color and fine details in bright light. In Retinitis Pigmentosa, rod cells are usually affected first, causing night blindness and peripheral vision loss. Later, cone cells may also deteriorate, leading to central vision problems.",
        "genetics_detail": "Retinitis Pigmentosa is caused by mutations in genes responsible for maintaining healthy photoreceptor cells. Scientists have identified more than 60 genes linked to RP, including RHO, USH2A, RPGR, and PRPF31. These mutations disrupt normal retinal cell function, eventually leading to degeneration of photoreceptors and progressive vision loss.",
        "fundus_features": "Eye scans of patients with Retinitis Pigmentosa often show characteristic retinal features. These include bone-spicule shaped pigment deposits in the retina, narrowing of retinal blood vessels, and a pale optic disc. These changes are visible in color fundus images and help ophthalmologists diagnose the condition.",
        "faf_scan": "Fundus Autofluorescence (FAF) imaging is used to analyze metabolic activity in retinal cells. In Retinitis Pigmentosa patients, FAF images may reveal abnormal patterns of autofluorescence caused by accumulation of lipofuscin in retinal pigment epithelium cells.",
        "gene_therapy": "Gene therapy is an emerging treatment approach for Retinitis Pigmentosa. This technique involves delivering a healthy copy of a defective gene into retinal cells using viral vectors. The goal is to restore or preserve photoreceptor function and slow the progression of vision loss.",
        "retinal_implant": "Retinal implants, also known as bionic eyes, are experimental devices designed to partially restore vision in individuals with severe retinal degeneration. These implants convert visual information captured by a camera into electrical signals that stimulate remaining retinal cells.",
        "associated_syndromes": "Retinitis Pigmentosa can sometimes occur as part of broader genetic syndromes. One example is Usher syndrome, where patients experience both RP and hearing loss. Another example is Bardet-Biedl syndrome, which may involve vision problems, obesity, and kidney abnormalities.",
        "psychological_impact": "Gradual vision loss due to Retinitis Pigmentosa can have emotional and psychological effects. Patients may experience anxiety, frustration, or reduced independence. Counseling, rehabilitation programs, and support groups can help individuals adapt to these challenges.",
        "rehabilitation": "Vision rehabilitation programs help people with retinal diseases maximize their remaining vision. These programs may include orientation and mobility training, assistive technologies, magnification devices, and adaptive strategies for daily activities.",
        "prevalence": "Retinitis Pigmentosa is considered a rare disease but affects approximately 1 in 3,000 to 1 in 4,000 people worldwide. Because it is inherited, the condition often appears in multiple members of the same family.",
        "night_blindness_reason": "Night blindness occurs in Retinitis Pigmentosa because rod photoreceptor cells are affected first. Rod cells are responsible for vision in low-light conditions. When these cells gradually degenerate, the eyes become less sensitive to dim light, making it difficult to see at night or in dark environments.",
        "tunnel_vision_reason": "Tunnel vision in Retinitis Pigmentosa happens due to progressive loss of peripheral photoreceptor cells in the retina. As these cells deteriorate, the visual field narrows, causing patients to see only a small central portion of their surroundings while losing side vision.",
        "rpe_role": "The retinal pigment epithelium (RPE) is a supportive layer beneath the photoreceptors that helps maintain retinal health. In Retinitis Pigmentosa, dysfunction in photoreceptors can also affect the RPE, further contributing to retinal degeneration.",
        "early_diagnosis": "Early diagnosis of Retinitis Pigmentosa can help patients understand the condition sooner and adopt lifestyle adjustments. It also allows doctors to monitor disease progression and recommend supportive vision aids when necessary.",
        "eye_protection": "People with Retinitis Pigmentosa are often advised to protect their eyes from excessive sunlight by wearing UV-protective sunglasses. This may help reduce additional stress on retinal cells.",
        "cure": "Retinitis Pigmentosa currently has no complete cure. However, several treatments and research approaches aim to slow the progression of the disease and preserve vision for a longer time. These include gene therapy, retinal implants, and supportive vision aids.",
        "age": "Symptoms of Retinitis Pigmentosa may begin in childhood, teenage years, or early adulthood. The exact age of onset varies depending on the specific genetic mutation and inheritance pattern.",
        "changes": "Lifestyle adjustments such as using brighter lighting indoors, wearing UV-protective sunglasses, and using assistive vision technologies can help individuals with Retinitis Pigmentosa manage daily activities more comfortably.",
        "diet": "While diet cannot cure Retinitis Pigmentosa, certain nutrients like Vitamin A, Omega-3 fatty acids, Vitamin C, Vitamin E, and Zinc may support general eye health. Patients should consult a doctor before taking supplements.",
        "children": "Yes, children can inherit Retinitis Pigmentosa if one or both parents carry the genetic mutation responsible for the disease. Genetic counseling helps families understand inheritance risks.",
        "progressive": "Retinitis Pigmentosa is called progressive because the damage to photoreceptor cells occurs gradually over time. This means vision loss develops slowly over several years or decades.",
        "retinal_diseases": "Major retinal diseases include Retinitis Pigmentosa, Age-Related Macular Degeneration, Diabetic Retinopathy, Retinal Detachment, Retinal Vein Occlusion, Stargardt Disease, and Retinopathy of Prematurity."
    }
    for key, response in rp_responses.items():
        if key in user_message_lower:
            return response
    return "I'm specifically designed to answer questions about Retinitis Pigmentosa and retinal diseases."

@app.route("/clear_chat", methods=["POST"])
def clear_chat():
    session.pop("chat_history", None)
    return jsonify({"success": True})

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(host="0.0.0.0", port=5000, debug=True)   