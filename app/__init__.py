import os
from flask import Flask, redirect, url_for

from app.models.db import db, login_manager
from config.config import config_by_name


def create_app(env=None):
    env = env or os.environ.get("FLASK_ENV", "development")

    app = Flask(__name__)
    selected_config = config_by_name.get(env, config_by_name["development"])
    app.config.from_object(selected_config)
    app.config["SQLALCHEMY_DATABASE_URI"] = selected_config.init_db_uri()

    # --- extensions ---
    db.init_app(app)
    login_manager.init_app(app)

    # --- blueprints ---
    from app.routes.auth import auth_bp
    from app.routes.dashboard import dashboard_bp
    from app.routes.forms import forms_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(forms_bp)

    @app.route("/")
    def root():
        return redirect(url_for("auth.login_page"))

    # Creates tables on first run if they don't exist yet.
    # For real migrations going forward, switch to Flask-Migrate (see requirements.txt).
    with app.app_context():
        db.create_all()

    return app
