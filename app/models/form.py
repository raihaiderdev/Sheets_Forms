from datetime import datetime, timezone
from app.models.db import db


class Form(db.Model):
    __tablename__ = "forms"

    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(255), nullable=False)
    description = db.Column(db.Text, nullable=True)
    created_by = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    created_at = db.Column(
        db.DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )
    is_active = db.Column(db.Boolean, default=True, nullable=False)

    fields = db.relationship(
        "FormField", backref="form", cascade="all, delete-orphan", order_by="FormField.order"
    )
    responses = db.relationship(
        "FormResponse", backref="form", cascade="all, delete-orphan"
    )
    creator = db.relationship("User", backref="forms")

    def to_dict(self):
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat(),
            "is_active": self.is_active,
            "field_count": len(self.fields),
            "response_count": len(self.responses),
        }


class FormField(db.Model):
    __tablename__ = "form_fields"

    id = db.Column(db.Integer, primary_key=True)
    form_id = db.Column(db.Integer, db.ForeignKey("forms.id"), nullable=False)
    label = db.Column(db.String(255), nullable=False)
    field_type = db.Column(db.String(50), nullable=False, default="text")
    placeholder = db.Column(db.String(255), nullable=True)
    is_required = db.Column(db.Boolean, default=False, nullable=False)
    options = db.Column(db.Text, nullable=True)   # comma-separated for select/radio
    order = db.Column(db.Integer, default=0, nullable=False)

    def to_dict(self):
        return {
            "id": self.id,
            "form_id": self.form_id,
            "label": self.label,
            "field_type": self.field_type,
            "placeholder": self.placeholder,
            "is_required": self.is_required,
            "options": self.options,
            "order": self.order,
        }


class FormResponse(db.Model):
    __tablename__ = "form_responses"

    id = db.Column(db.Integer, primary_key=True)
    form_id = db.Column(db.Integer, db.ForeignKey("forms.id"), nullable=False)
    submitted_by = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)
    submitted_at = db.Column(
        db.DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )

    answers = db.relationship(
        "FormFieldResponse", backref="response", cascade="all, delete-orphan"
    )
    submitter = db.relationship("User", backref="form_responses")

    def to_dict(self):
        return {
            "id": self.id,
            "form_id": self.form_id,
            "submitted_by": self.submitted_by,
            "submitted_at": self.submitted_at.isoformat(),
            "answers": [a.to_dict() for a in self.answers],
        }


class FormFieldResponse(db.Model):
    __tablename__ = "form_field_responses"

    id = db.Column(db.Integer, primary_key=True)
    response_id = db.Column(db.Integer, db.ForeignKey("form_responses.id"), nullable=False)
    field_id = db.Column(db.Integer, db.ForeignKey("form_fields.id"), nullable=False)
    value = db.Column(db.Text, nullable=True)

    field = db.relationship("FormField")

    def to_dict(self):
        return {
            "field_id": self.field_id,
            "label": self.field.label if self.field else "",
            "value": self.value,
        }
