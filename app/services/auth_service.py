import re
from flask import current_app

from app.models import db, User

EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")


class ValidationError(Exception):
    """Raised when incoming signup/login data fails validation."""

    def __init__(self, errors: dict):
        self.errors = errors
        super().__init__(str(errors))


def validate_signup(data: dict) -> dict:
    """Validate and normalize signup payload. Raises ValidationError on failure."""
    errors = {}

    username = (data.get("username") or "").strip()
    email = (data.get("email") or "").strip().lower()
    department = (data.get("department") or "").strip()
    office = (data.get("office") or "").strip()
    password = data.get("password") or ""

    if len(username) < 3:
        errors["username"] = "Username must be at least 3 characters."

    if not EMAIL_RE.match(email):
        errors["email"] = "Enter a valid email address."

    if not department:
        errors["department"] = "Department is required."

    valid_offices = current_app.config.get("OFFICE_CHOICES", [])
    if office not in valid_offices:
        errors["office"] = "Select a valid office from the list."

    if len(password) < 8:
        errors["password"] = "Password must be at least 8 characters."

    if not errors:
        if User.query.filter_by(email=email).first():
            errors["email"] = "An account with this email already exists."
        elif User.query.filter_by(username=username).first():
            errors["username"] = "This username is already taken."

    if errors:
        raise ValidationError(errors)

    return {
        "username": username,
        "email": email,
        "department": department,
        "office": office,
        "password": password,
    }


def validate_login(data: dict) -> dict:
    errors = {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    if not EMAIL_RE.match(email):
        errors["email"] = "Enter a valid email address."
    if not password:
        errors["password"] = "Password is required."

    if errors:
        raise ValidationError(errors)

    return {"email": email, "password": password}


def create_user(clean_data: dict) -> User:
    user = User(
        username=clean_data["username"],
        email=clean_data["email"],
        department=clean_data["department"],
        office=clean_data["office"],
    )
    user.set_password(clean_data["password"])
    db.session.add(user)
    db.session.commit()
    return user


def authenticate(email: str, password: str):
    """Returns the User if credentials are valid, else None."""
    user = User.query.filter_by(email=email).first()
    if user and user.check_password(password):
        return user
    return None
