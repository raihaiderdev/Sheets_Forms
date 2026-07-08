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
    app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB max upload

    # --- extensions ---
    db.init_app(app)
    login_manager.init_app(app)

    # --- blueprints ---
    from app.routes.auth import auth_bp
    from app.routes.dashboard import dashboard_bp
    from app.routes.forms import forms_bp
    from app.routes.photos import photos_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(forms_bp)
    app.register_blueprint(photos_bp)

    @app.route("/")
    def root():
        return redirect(url_for("auth.login_page"))

    # Creates tables on first run if they don't exist yet.
    # Also applies additive column migrations for existing tables.
    with app.app_context():
        db.create_all()
        _run_migrations(db)

    return app


def _run_migrations(db):
    """
    Safely add new columns to existing tables.
    Each statement is run independently — errors (column already exists) are ignored.
    """
    migrations = [
        # forms table — new columns added in v2
        "ALTER TABLE forms ADD COLUMN allow_multiple BOOLEAN NOT NULL DEFAULT FALSE",
        "ALTER TABLE forms ADD COLUMN unique_field_label VARCHAR(255)",
        # form_responses table — new columns added in v2
        "ALTER TABLE form_responses ADD COLUMN respondent_email VARCHAR(255)",
        "ALTER TABLE form_responses ADD COLUMN unique_key_value VARCHAR(500)",
        "ALTER TABLE form_responses ADD COLUMN updated_at TIMESTAMP WITHOUT TIME ZONE",
        # backfill updated_at for existing rows
        "UPDATE form_responses SET updated_at = submitted_at WHERE updated_at IS NULL",
        # indexes for dedup lookups
        "CREATE INDEX IF NOT EXISTS ix_form_responses_respondent_email ON form_responses (respondent_email)",
        "CREATE INDEX IF NOT EXISTS ix_form_responses_unique_key_value ON form_responses (unique_key_value)",
    ]

    with db.engine.connect() as conn:
        for sql in migrations:
            try:
                conn.execute(db.text(sql))
                conn.commit()
            except Exception:
                conn.rollback()   # column already exists or similar — safe to ignore
