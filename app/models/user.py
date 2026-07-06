from datetime import datetime, timezone
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

from app.models.db import db, login_manager


class User(UserMixin, db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)
    department = db.Column(db.String(150), nullable=False)
    office = db.Column(db.String(50), nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    is_active_flag = db.Column("is_active", db.Boolean, default=True, nullable=False)
    created_at = db.Column(
        db.DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )

    def set_password(self, raw_password: str) -> None:
        self.password_hash = generate_password_hash(raw_password)

    def check_password(self, raw_password: str) -> bool:
        return check_password_hash(self.password_hash, raw_password)

    # flask-login expects an `is_active` property
    @property
    def is_active(self):
        return self.is_active_flag

    def to_dict(self):
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "department": self.department,
            "office": self.office,
            "created_at": self.created_at.isoformat(),
        }

    def __repr__(self):
        return f"<User {self.username} ({self.office})>"


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))
