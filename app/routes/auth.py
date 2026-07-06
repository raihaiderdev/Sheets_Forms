from flask import Blueprint, render_template, request, jsonify, redirect, url_for
from flask_login import login_user, logout_user, login_required, current_user

from app.services.auth_service import (
    validate_signup,
    validate_login,
    create_user,
    authenticate,
    ValidationError,
)

auth_bp = Blueprint("auth", __name__, template_folder="../templates/auth")


@auth_bp.route("/login", methods=["GET"])
def login_page():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard.index"))
    return render_template("auth/login.html")


@auth_bp.route("/api/signup", methods=["POST"])
def api_signup():
    payload = request.get_json(silent=True) or {}

    try:
        clean = validate_signup(payload)
    except ValidationError as exc:
        return jsonify({"ok": False, "errors": exc.errors}), 400

    user = create_user(clean)
    return jsonify({"ok": True, "message": "Account created.", "user": user.to_dict()}), 201


@auth_bp.route("/api/login", methods=["POST"])
def api_login():
    payload = request.get_json(silent=True) or {}

    try:
        clean = validate_login(payload)
    except ValidationError as exc:
        return jsonify({"ok": False, "errors": exc.errors}), 400

    user = authenticate(clean["email"], clean["password"])
    if not user:
        return jsonify({"ok": False, "errors": {"password": "Invalid email or password."}}), 401

    login_user(user, remember=True)
    return jsonify({"ok": True, "redirect": url_for("dashboard.index")}), 200


@auth_bp.route("/logout", methods=["POST"])
@login_required
def logout():
    logout_user()
    return redirect(url_for("auth.login_page"))
