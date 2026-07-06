from flask import Blueprint, render_template, jsonify
from flask_login import login_required, current_user

from app.services.form_service import list_forms

dashboard_bp = Blueprint("dashboard", __name__, template_folder="../templates/dashboard")


@dashboard_bp.route("/dashboard")
@login_required
def index():
    return render_template("dashboard/dashboard.html", user=current_user)


@dashboard_bp.route("/api/forms-summary")
@login_required
def api_forms_summary():
    forms = list_forms(current_user.id)
    total_responses = sum(len(f.responses) for f in forms)
    return jsonify({
        "form_count": len(forms),
        "response_count": total_responses,
        "forms": [f.to_dict() for f in forms],
    })
