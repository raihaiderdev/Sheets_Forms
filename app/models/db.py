from flask import request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager

db = SQLAlchemy()
login_manager = LoginManager()
login_manager.login_view = "auth.login"
login_manager.login_message = "Please sign in to access this page."
login_manager.login_message_category = "info"


@login_manager.unauthorized_handler
def unauthorized():
    """Return JSON 401 for API calls, redirect to login for page requests."""
    if request.path.startswith("/api/"):
        return jsonify({"ok": False, "error": "Authentication required."}), 401
    from flask import redirect, url_for
    return redirect(url_for("auth.login_page"))
