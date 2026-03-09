import os
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from torchvision.models import efficientnet_b4, vit_b_16
from flask import Flask, request, render_template, redirect, url_for, flash, send_file, make_response
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from PIL import Image
from flask_cors import CORS
import cv2
import torch.nn.functional as F
from io import BytesIO
from datetime import datetime
from weasyprint import HTML
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

print("🚀 Starting app...")

# -------------------------------
# Database Setup
# -------------------------------
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config['SECRET_KEY'] = 'your-secret-key-change-in-production'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///instance/rp_detection.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Create database tables
with app.app_context():
    db.create_all()
    print("✅ Database tables created!")

# -------------------------------
# Database Models
# -------------------------------
class User(db.Model):
    """User model for authentication"""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship to prediction results
    predictions = db.relationship('PredictionResult', backref='user', lazy=True, cascade='all, delete-orphan')
    
    def set_password(self, password):
        """Hash and set the user's password"""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Verify the user's password"""
        return check_password_hash(self.password_hash, password)
    
    def __repr__(self):
        return f'<User {self.username}>'


class PredictionResult(db.Model):
    """Model to store prediction results for each user"""
    __tablename__ = 'prediction_results'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    prediction = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.String(20), nullable=False)
    stage = db.Column(db.String(20), nullable=True)
    gradcam_path = db.Column(db.String(256), nullable=True)
    image_filename = db.Column(db.String(256), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<PredictionResult user_id={self.user_id} prediction={self.prediction}>'


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# -------------------------------
# Hybrid Model (SAME AS TRAINING)
# -------------------------------
class HybridModel(nn.Module):
    def __init__(self, num_classes=2):
        super(HybridModel, self).__init__()

        self.effnet = efficientnet_b4(weights=None)
        self.effnet.classifier = nn.Identity()

        self.vit = vit_b_16(weights=None)
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
        vit_input = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        f2 = self.vit(vit_input)

        combined = torch.cat((f1, f2), dim=1)
        return self.classifier(combined)

# -------------------------------
# Load Model
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classes = ["Healthy", "Retinitis Pigmentosa"]

model = HybridModel(num_classes=len(classes)).to(device)

try:
    model.load_state_dict(torch.load("hybrid_model.pth", map_location=device))
    print("✅ Hybrid Model Loaded!")
except Exception as e:
    print("❌ Model loading failed:", e)
    exit()

model.eval()

# -------------------------------
# Image Transform (MATCH TRAINING)
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -------------------------------
# Grad-CAM
# -------------------------------
def generate_gradcam(model, img_tensor):
    features = []
    gradients = []

    def forward_hook(module, input, output):
        features.append(output)

    target_layer = model.effnet.features[-2]

    target_layer.register_forward_hook(forward_hook)
    target_layer.register_full_backward_hook(
        lambda m, gin, gout: gradients.append(gout[0])
    )

    output = model(img_tensor)
    pred_class = output.argmax(dim=1)

    model.zero_grad()
    output[0, pred_class].backward()

    grads = gradients[0].mean(dim=[2, 3], keepdim=True)
    fmap = features[0]

    cam = (grads * fmap).sum(dim=1).squeeze().detach().cpu().numpy()
    cam = np.maximum(cam, 0)
    cam = cam - np.min(cam)
    cam = cam / (np.max(cam) + 1e-8)

    return cam

# -------------------------------
# Stage Estimation
# -------------------------------
def estimate_stage(heatmap):
    normalized = heatmap / (np.max(heatmap) + 1e-8)
    activation_ratio = np.sum(normalized > 0.25) / normalized.size

    if activation_ratio < 0.10:
        return "Early"
    elif activation_ratio < 0.25:
        return "Moderate"
    else:
        return "Severe"

# -------------------------------
# Retinal Image Validation
# -------------------------------
def is_retinal_image(img_path):
    filename = os.path.basename(img_path).lower()
    keywords = ["retina", "fundus", "uwf", "fa", "faf", "eye"]

    if any(k in filename for k in keywords):
        return True

    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 7)
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, 100,
                               param1=50, param2=30,
                               minRadius=50, maxRadius=300)

    return circles is not None

# -------------------------------
# Flask Setup
# -------------------------------
# Note: app already initialized above with SQLAlchemy
CORS(app)

UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -------------------------------
# Authentication Routes
# -------------------------------
@app.route("/login", methods=["GET", "POST"])
def login():
    """Handle user login"""
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            login_user(user)
            flash("Login successful!", "success")
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid username or password", "error")
    
    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    """Handle user registration"""
    if request.method == "POST":
        username = request.form.get("username")
        email = request.form.get("email")
        password = request.form.get("password")
        
        # Check if user already exists
        if User.query.filter_by(username=username).first():
            flash("Username already exists", "error")
            return redirect(url_for("register"))
        
        if User.query.filter_by(email=email).first():
            flash("Email already registered", "error")
            return redirect(url_for("register"))
        
        # Create new user
        new_user = User(username=username, email=email)
        new_user.set_password(password)
        
        db.session.add(new_user)
        db.session.commit()
        
        flash("Registration successful! Please login.", "success")
        return redirect(url_for("login"))
    
    return render_template("register.html")


@app.route("/dashboard")
@login_required
def dashboard():
    """Display user dashboard with stats and predictions"""
    # Get user's predictions
    predictions = PredictionResult.query.filter_by(user_id=current_user.id).order_by(PredictionResult.created_at.desc()).all()
    
    # Calculate statistics
    total_tests = len(predictions)
    healthy_count = sum(1 for p in predictions if p.prediction == "Healthy")
    rp_count = sum(1 for p in predictions if p.prediction == "Retinitis Pigmentosa")
    
    # Stage counts for RP predictions
    early_count = sum(1 for p in predictions if p.prediction == "Retinitis Pigmentosa" and p.stage == "Early")
    moderate_count = sum(1 for p in predictions if p.prediction == "Retinitis Pigmentosa" and p.stage == "Moderate")
    severe_count = sum(1 for p in predictions if p.prediction == "Retinitis Pigmentosa" and p.stage == "Severe")
    
    stats = {
        "total_tests": total_tests,
        "healthy_count": healthy_count,
        "rp_count": rp_count,
        "early_count": early_count,
        "moderate_count": moderate_count,
        "severe_count": severe_count
    }
    
    # Build tracking data for charts
    tracking_data = None
    if rp_count > 0:
        # Get timeline data
        rp_predictions = [p for p in predictions if p.prediction == "Retinitis Pigmentosa"]
        timeline_data = []
        monthly_trend = {}
        
        for pred in sorted(rp_predictions, key=lambda x: x.created_at):
            date_str = pred.created_at.strftime("%Y-%m-%d")
            timeline_data.append({
                "date": date_str,
                "prediction": pred.prediction
            })
            
            month_key = pred.created_at.strftime("%Y-%m")
            if month_key not in monthly_trend:
                monthly_trend[month_key] = {"month": month_key, "healthy": 0, "rp": 0}
            monthly_trend[month_key]["rp"] += 1
        
        # Add healthy counts to monthly trend
        for pred in predictions:
            month_key = pred.created_at.strftime("%Y-%m")
            if month_key in monthly_trend:
                if pred.prediction == "Healthy":
                    monthly_trend[month_key]["healthy"] += 1
        
        # Determine progression status
        if len(rp_predictions) > 1:
            stages = [p.stage for p in rp_predictions if p.stage]
            if stages:
                if stages[-1] == "Early":
                    progression_status = "monitoring"
                    progression_message = "Your latest RP detection is at an early stage. Regular monitoring is recommended."
                elif stages[-1] == "Moderate":
                    if "Severe" in stages[:-1]:
                        progression_status = "stable"
                        progression_message = "Your condition appears to be stable. Continue regular checkups."
                    else:
                        progression_status = "worsening"
                        progression_message = "Your condition shows signs of progression. Please consult your doctor."
                elif stages[-1] == "Severe":
                    progression_status = "worsening"
                    progression_message = "Your condition requires attention. Please consult your healthcare provider."
                else:
                    progression_status = "monitoring"
                    progression_message = "Regular monitoring is recommended."
            else:
                progression_status = "monitoring"
                progression_message = "Regular monitoring is recommended."
        else:
            progression_status = "monitoring"
            progression_message = "This is your first RP detection. Regular monitoring is recommended."
        
        # Build stage progression
        stage_progression = []
        for pred in sorted(rp_predictions, key=lambda x: x.created_at):
            if pred.stage:
                stage_progression.append({
                    "date": pred.created_at.strftime("%Y-%m-%d"),
                    "stage": pred.stage
                })
        
        tracking_data = {
            "timeline_data": timeline_data,
            "monthly_trend": list(monthly_trend.values()),
            "progression_status": progression_status,
            "progression_message": progression_message,
            "stage_progression": stage_progression
        }
    
    return render_template("dashboard.html", 
                         user=current_user, 
                         stats=stats, 
                         predictions=predictions,
                         tracking_data=tracking_data)


@app.route("/logout")
@login_required
def logout():
    """Handle user logout"""
    logout_user()
    flash("You have been logged out", "success")
    return redirect(url_for("welcome"))


@app.route("/download_dashboard")
@login_required
def download_dashboard():
    """Generate and download dashboard report as PDF with graphs and tracking"""
    # Get user's predictions
    predictions = PredictionResult.query.filter_by(user_id=current_user.id).order_by(PredictionResult.created_at.desc()).all()
    
    # Calculate statistics
    total_tests = len(predictions)
    healthy_count = sum(1 for p in predictions if p.prediction == "Healthy")
    rp_count = sum(1 for p in predictions if p.prediction == "Retinitis Pigmentosa")
    
    # Stage counts for RP predictions
    early_count = sum(1 for p in predictions if p.prediction == "Retinitis Pigmentosa" and p.stage == "Early")
    moderate_count = sum(1 for p in predictions if p.prediction == "Retinitis Pigmentosa" and p.stage == "Moderate")
    severe_count = sum(1 for p in predictions if p.prediction == "Retinitis Pigmentosa" and p.stage == "Severe")
    
    stats = {
        "total_tests": total_tests,
        "healthy_count": healthy_count,
        "rp_count": rp_count,
        "early_count": early_count,
        "moderate_count": moderate_count,
        "severe_count": severe_count
    }
    
    # Build tracking data for charts (same as dashboard)
    tracking_data = None
    chart_images = []
    
    if rp_count > 0:
        # Get timeline data
        rp_predictions = [p for p in predictions if p.prediction == "Retinitis Pigmentosa"]
        timeline_data = []
        monthly_trend = {}
        
        for pred in sorted(rp_predictions, key=lambda x: x.created_at):
            date_str = pred.created_at.strftime("%Y-%m-%d")
            timeline_data.append({
                "date": date_str,
                "prediction": pred.prediction
            })
            
            month_key = pred.created_at.strftime("%Y-%m")
            if month_key not in monthly_trend:
                monthly_trend[month_key] = {"month": month_key, "healthy": 0, "rp": 0}
            monthly_trend[month_key]["rp"] += 1
        
        # Add healthy counts to monthly trend
        for pred in predictions:
            month_key = pred.created_at.strftime("%Y-%m")
            if month_key in monthly_trend:
                if pred.prediction == "Healthy":
                    monthly_trend[month_key]["healthy"] += 1
        
        # Determine progression status
        if len(rp_predictions) > 1:
            stages = [p.stage for p in rp_predictions if p.stage]
            if stages:
                if stages[-1] == "Early":
                    progression_status = "monitoring"
                    progression_message = "Your latest RP detection is at an early stage. Regular monitoring is recommended."
                elif stages[-1] == "Moderate":
                    if "Severe" in stages[:-1]:
                        progression_status = "stable"
                        progression_message = "Your condition appears to be stable. Continue regular checkups."
                    else:
                        progression_status = "worsening"
                        progression_message = "Your condition shows signs of progression. Please consult your doctor."
                elif stages[-1] == "Severe":
                    progression_status = "worsening"
                    progression_message = "Your condition requires attention. Please consult your healthcare provider."
                else:
                    progression_status = "monitoring"
                    progression_message = "Regular monitoring is recommended."
            else:
                progression_status = "monitoring"
                progression_message = "Regular monitoring is recommended."
        else:
            progression_status = "monitoring"
            progression_message = "This is your first RP detection. Regular monitoring is recommended."
        
        # Build stage progression
        stage_progression = []
        for pred in sorted(rp_predictions, key=lambda x: x.created_at):
            if pred.stage:
                stage_progression.append({
                    "date": pred.created_at.strftime("%Y-%m-%d"),
                    "stage": pred.stage
                })
        
        tracking_data = {
            "timeline_data": timeline_data,
            "monthly_trend": list(monthly_trend.values()),
            "progression_status": progression_status,
            "progression_message": progression_message,
            "stage_progression": stage_progression
        }
        
        # Generate Timeline Chart
        if timeline_data:
            plt.figure(figsize=(10, 5))
            dates = [d['date'] for d in timeline_data]
            values = [0 if d['prediction'] == 'Healthy' else 1 for d in timeline_data]
            plt.plot(dates, values, marker='o', color='#dc3545', linewidth=2, markersize=8)
            plt.ylim(-0.2, 1.2)
            plt.yticks([0, 1], ['Healthy', 'RP'])
            plt.xlabel('Date')
            plt.ylabel('Health Status')
            plt.title('Health Trend Over Time')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save to base64
            timeline_buffer = BytesIO()
            plt.savefig(timeline_buffer, format='png', dpi=100, bbox_inches='tight')
            timeline_buffer.seek(0)
            timeline_chart = base64.b64encode(timeline_buffer.getvalue()).decode()
            chart_images.append(('timeline', timeline_chart))
            plt.close()
        
        # Generate Monthly Trend Chart
        if monthly_trend:
            plt.figure(figsize=(10, 5))
            months = [m['month'] for m in monthly_trend.values()]
            healthy_vals = [m['healthy'] for m in monthly_trend.values()]
            rp_vals = [m['rp'] for m in monthly_trend.values()]
            
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
            
            # Save to base64
            monthly_buffer = BytesIO()
            plt.savefig(monthly_buffer, format='png', dpi=100, bbox_inches='tight')
            monthly_buffer.seek(0)
            monthly_chart = base64.b64encode(monthly_buffer.getvalue()).decode()
            chart_images.append(('monthly', monthly_chart))
            plt.close()
    
    # Get chart base64 images
    timeline_chart_img = ""
    monthly_chart_img = ""
    for chart_type, chart_b64 in chart_images:
        if chart_type == 'timeline':
            timeline_chart_img = chart_b64
        elif chart_type == 'monthly':
            monthly_chart_img = chart_b64
    
    # Generate HTML for PDF with charts and tracking
    tracking_html = ""
    if tracking_data:
        status_colors = {
            'stable': '#28a745',
            'worsening': '#dc3545',
            'improving': '#28a745',
            'monitoring': '#ffc107'
        }
        status_labels = {
            'stable': '✓ Stable',
            'worsening': '⚠ Worsening',
            'improving': '↑ Improving',
            'monitoring': '👁 Monitoring'
        }
        status_color = status_colors.get(tracking_data['progression_status'], '#ffc107')
        status_label = status_labels.get(tracking_data['progression_status'], 'Monitoring')
        
        tracking_html = f"""
        <div style="margin: 30px 0; padding: 20px; border-radius: 10px; background: {status_color}20; border: 2px solid {status_color};">
            <h3 style="color: {status_color}; margin-top: 0;">Disease Tracking - {status_label}</h3>
            <p><strong>Message:</strong> {tracking_data['progression_message']}</p>
        </div>
        
        <h3>Stage Progression</h3>
        <table style="width: 100%; border-collapse: collapse; margin: 15px 0;">
            <tr style="background: #2a4d8f; color: white;">
                <th style="padding: 10px; border: 1px solid #ddd;">Date</th>
                <th style="padding: 10px; border: 1px solid #ddd;">Stage</th>
            </tr>
        """
        for stage in tracking_data.get('stage_progression', []):
            stage_color = '#ffc107' if stage['stage'] == 'Early' else ('#fd7e14' if stage['stage'] == 'Moderate' else '#dc3545')
            tracking_html += f"""
            <tr>
                <td style="padding: 10px; border: 1px solid #ddd;">{stage['date']}</td>
                <td style="padding: 10px; border: 1px solid #ddd;"><span style="background: {stage_color}; color: {'white' if stage['stage'] != 'Early' else 'black'}; padding: 5px 10px; border-radius: 4px;">{stage['stage']}</span></td>
            </tr>
            """
        tracking_html += "</table>"
        
        # Add charts if available
        if timeline_chart_img:
            tracking_html += f"""
            <div style="margin: 20px 0; page-break-inside: avoid;">
                <h3>Health Trend Over Time</h3>
                <img src="data:image/png;base64,{timeline_chart_img}" style="max-width: 100%; height: auto;">
            </div>
            """
        
        if monthly_chart_img:
            tracking_html += f"""
            <div style="margin: 20px 0; page-break-inside: avoid;">
                <h3>Monthly Breakdown</h3>
                <img src="data:image/png;base64,{monthly_chart_img}" style="max-width: 100%; height: auto;">
            </div>
            """
    
    # Stage statistics HTML
    stage_stats_html = ""
    if rp_count > 0:
        stage_stats_html = f"""
        <div class="stat-box"><strong>Early Stage:</strong> {early_count}</div>
        <div class="stat-box"><strong>Moderate Stage:</strong> {moderate_count}</div>
        <div class="stat-box"><strong>Severe Stage:</strong> {severe_count}</div>
        """
    
    # Generate HTML for PDF
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Dashboard Report - {current_user.username}</title>
        <style>
            body {{ font-family: Arial, sans-serif; padding: 20px; color: #333; }}
            h1 {{ color: #2a4d8f; margin-bottom: 5px; }}
            h2 {{ color: #2a4d8f; border-bottom: 2px solid #2a4d8f; padding-bottom: 10px; }}
            .stats {{ display: flex; gap: 20px; margin: 20px 0; flex-wrap: wrap; }}
            .stat-box {{ background: #f5f5f5; padding: 15px; border-radius: 8px; min-width: 120px; }}
            .stat-box.healthy {{ border-left: 4px solid #28a745; }}
            .stat-box.rp {{ border-left: 4px solid #dc3545; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background: #2a4d8f; color: white; }}
            tr:nth-child(even) {{ background: #f9f9f9; }}
            .header-info {{ background: #e9ecef; padding: 15px; border-radius: 8px; margin-bottom: 20px; }}
            .footer {{ margin-top: 30px; text-align: center; color: #666; font-size: 12px; }}
        </style>
    </head>
    <body>
        <h1>RP Detection Dashboard Report</h1>
        
        <div class="header-info">
            <p><strong>Patient:</strong> {current_user.username}</p>
            <p><strong>Report Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>
        </div>
        
        <h2>Statistics Overview</h2>
        <div class="stats">
            <div class="stat-box"><strong>Total Tests:</strong> {stats['total_tests']}</div>
            <div class="stat-box healthy"><strong>Healthy:</strong> {stats['healthy_count']}</div>
            <div class="stat-box rp"><strong>RP Detected:</strong> {stats['rp_count']}</div>
            {stage_stats_html}
        </div>
        
        {tracking_html}
        
        <h2>Prediction History</h2>
        <table>
            <tr>
                <th>Date</th>
                <th>Prediction</th>
                <th>Confidence</th>
                <th>Stage</th>
            </tr>
    """
    
    for pred in predictions:
        html_content += f"""
            <tr>
                <td>{pred.created_at.strftime('%Y-%m-%d %H:%M')}</td>
                <td>{pred.prediction}</td>
                <td>{pred.confidence}</td>
                <td>{pred.stage if pred.stage else 'N/A'}</td>
            </tr>
        """
    
    html_content += """
        </table>
        
        <div class="footer">
            <p>This is an automated report generated by the RP Detection System.</p>
            <p>Retinitis Pigmentosa Detection System</p>
        </div>
    </body>
    </html>
    """
    
    # Generate PDF
    pdf_file = HTML(string=html_content).write_pdf()
    
    return send_file(
        BytesIO(pdf_file),
        mimetype='application/pdf',
        as_attachment=True,
        download_name=f'dashboard_report_{current_user.username}.pdf'
    )


@app.route("/download_report/<int:prediction_id>")
@login_required
def download_report(prediction_id):
    """Download individual prediction report"""
    prediction = PredictionResult.query.filter_by(id=prediction_id, user_id=current_user.id).first_or_404()
    
    # Generate HTML for PDF
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Prediction Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; padding: 20px; }}
            h1 {{ color: #2a4d8f; }}
            .info {{ margin: 20px 0; }}
            .result {{ padding: 20px; border-radius: 8px; margin: 20px 0; }}
            .healthy {{ background: #d4edda; }}
            .rp {{ background: #f8d7da; }}
        </style>
    </head>
    <body>
        <h1>RP Detection Report</h1>
        
        <div class="info">
            <p><strong>User:</strong> {current_user.username}</p>
            <p><strong>Date:</strong> {prediction.created_at.strftime('%Y-%m-%d %H:%M')}</p>
        </div>
        
        <div class="result {'healthy' if prediction.prediction == 'Healthy' else 'rp'}">
            <h2>Result: {prediction.prediction}</h2>
            <p><strong>Confidence:</strong> {prediction.confidence}</p>
            <p><strong>Stage:</strong> {prediction.stage if prediction.stage else 'N/A'}</p>
        </div>
        
        <p><em>This is an automated report generated by the RP Detection System.</em></p>
    </body>
    </html>
    """
    
    # Generate PDF
    pdf_file = HTML(string=html_content).write_pdf()
    
    return send_file(
        BytesIO(pdf_file),
        mimetype='application/pdf',
        as_attachment=True,
        download_name=f'prediction_report_{prediction_id}.pdf'
    )

# -------------------------------
# Routes (UPDATED ✅)
# -------------------------------
@app.route("/")
def welcome():
    return render_template("welcome.html")

@app.route("/index")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return render_template("result.html", prediction="No file uploaded", confidence="N/A")

    file = request.files["file"]

    if file.filename == "":
        return render_template("result.html", prediction="No file selected", confidence="N/A")

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Validate retinal image
    if not is_retinal_image(file_path):
        return render_template(
            "result.html",
            prediction="⚠️ Upload retinal image (Fundus/UWF/FAF)",
            confidence="N/A",
            stage=None,
            gradcam=None
        )

    try:
        img = Image.open(file_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)

        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

        label = classes[pred.item()]
        confidence = f"{conf.item()*100:.2f}%"

        stage = None
        gradcam_path = None

        if label == "Retinitis Pigmentosa":
            heatmap = generate_gradcam(model, img_tensor)
            stage = estimate_stage(heatmap)

            img_np = np.array(img.resize((224, 224)))
            heatmap_resized = cv2.resize(heatmap, (224, 224))

            heatmap_color = cv2.applyColorMap(
                np.uint8(255 * heatmap_resized),
                cv2.COLORMAP_JET
            )

            overlay = cv2.addWeighted(img_np, 0.7, heatmap_color, 0.3, 0)

            gradcam_filename = f"gradcam_{file.filename}"
            gradcam_path = os.path.join("static", "uploads", gradcam_filename)

            cv2.imwrite(gradcam_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        return render_template(
            "result.html",
            prediction=label,
            confidence=confidence,
            stage=stage,
            gradcam=gradcam_path
        )

    except Exception as e:
        return render_template("result.html", prediction="Error", confidence=str(e))

# -------------------------------
# Run App
# -------------------------------
if __name__ == "__main__":
    print("🔥 Running Flask...")
    app.run(host="0.0.0.0", port=5000, debug=True)