from flask import Blueprint, render_template, request, jsonify, abort, url_for
from flask_login import login_required, current_user

from app.services.form_service import (
    parse_excel,
    create_form,
    get_form,
    list_forms,
    delete_form,
    submit_response,
)

forms_bp = Blueprint("forms", __name__, template_folder="../templates")

ALLOWED_EXTENSIONS = {"xlsx", "xls"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024   # 10 MB


def _ext_ok(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

@forms_bp.route("/forms")
@login_required
def forms_list():
    forms = list_forms(current_user.id)
    return render_template("forms/forms_list.html", user=current_user, forms=forms)


@forms_bp.route("/forms/new")
@login_required
def form_builder():
    return render_template("forms/form_builder.html", user=current_user)


@forms_bp.route("/forms/<int:form_id>/fill")
@login_required
def form_fill(form_id):
    form = get_form(form_id)
    if not form or not form.is_active:
        abort(404)
    return render_template("forms/form_fill.html", user=current_user, form=form)


@forms_bp.route("/forms/<int:form_id>/responses")
@login_required
def form_responses(form_id):
    form = get_form(form_id)
    if not form or form.created_by != current_user.id:
        abort(403)
    return render_template("forms/form_responses.html", user=current_user, form=form)


# ---------------------------------------------------------------------------
# API — Excel upload → smart header detection
# ---------------------------------------------------------------------------

@forms_bp.route("/api/forms/parse-excel", methods=["POST"])
@login_required
def api_parse_excel():
    if "file" not in request.files:
        return jsonify({"ok": False, "error": "No file uploaded."}), 400

    f = request.files["file"]
    if not f.filename or not _ext_ok(f.filename):
        return jsonify({"ok": False, "error": "Only .xlsx / .xls files are accepted."}), 400

    data = f.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        return jsonify({"ok": False, "error": "File too large (max 10 MB)."}), 413

    try:
        result = parse_excel(data)
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 422
    except Exception as exc:
        return jsonify({"ok": False, "error": f"Could not read file: {exc}"}), 422

    return jsonify({"ok": True, **result})


# ---------------------------------------------------------------------------
# API — Create form
# ---------------------------------------------------------------------------

@forms_bp.route("/api/forms", methods=["POST"])
@login_required
def api_create_form():
    payload = request.get_json(silent=True) or {}
    title = (payload.get("title") or "").strip()
    description = (payload.get("description") or "").strip()
    fields = payload.get("fields") or []

    if not title:
        return jsonify({"ok": False, "error": "Form title is required."}), 400
    if not fields:
        return jsonify({"ok": False, "error": "Add at least one field."}), 400

    form = create_form(title, description, current_user.id, fields)
    return jsonify({"ok": True, "form_id": form.id}), 201


# ---------------------------------------------------------------------------
# API — Delete form
# ---------------------------------------------------------------------------

@forms_bp.route("/api/forms/<int:form_id>", methods=["DELETE"])
@login_required
def api_delete_form(form_id):
    ok = delete_form(form_id, current_user.id)
    if not ok:
        return jsonify({"ok": False, "error": "Form not found or not yours."}), 404
    return jsonify({"ok": True})


# ---------------------------------------------------------------------------
# Public shareable fill page — no login required
# ---------------------------------------------------------------------------

@forms_bp.route("/f/<int:form_id>")
def public_form_fill(form_id):
    """Public URL that can be shared via WhatsApp/email for data collection."""
    form = get_form(form_id)
    if not form or not form.is_active:
        abort(404)
    return render_template("forms/public_fill.html", form=form)


@forms_bp.route("/api/public/forms/<int:form_id>/responses", methods=["POST"])
def api_public_submit(form_id):
    """Accept submissions from the public fill form (anonymous)."""
    form = get_form(form_id)
    if not form or not form.is_active:
        return jsonify({"ok": False, "error": "Form not found."}), 404

    payload = request.get_json(silent=True) or {}
    answers = payload.get("answers") or {}

    errors = {}
    for field in form.fields:
        val = str(answers.get(str(field.id), "")).strip()
        if field.is_required and not val:
            errors[str(field.id)] = f"{field.label} is required."

    if errors:
        return jsonify({"ok": False, "errors": errors}), 400

    response = submit_response(form_id, None, answers)
    return jsonify({"ok": True, "response_id": response.id}), 201


# ---------------------------------------------------------------------------
# API — Submit response (authenticated)
# ---------------------------------------------------------------------------

@forms_bp.route("/api/forms/<int:form_id>/responses", methods=["POST"])
@login_required
def api_submit_response(form_id):
    form = get_form(form_id)
    if not form or not form.is_active:
        return jsonify({"ok": False, "error": "Form not found."}), 404

    payload = request.get_json(silent=True) or {}
    answers = payload.get("answers") or {}

    errors = {}
    for field in form.fields:
        val = str(answers.get(str(field.id), "")).strip()
        if field.is_required and not val:
            errors[str(field.id)] = f"{field.label} is required."

    if errors:
        return jsonify({"ok": False, "errors": errors}), 400

    from flask_login import current_user
    uid = current_user.id if current_user.is_authenticated else None
    response = submit_response(form_id, uid, answers)
    return jsonify({"ok": True, "response_id": response.id}), 201


# ---------------------------------------------------------------------------
# API — Get form data (for dynamic rendering)
# ---------------------------------------------------------------------------

@forms_bp.route("/api/forms/<int:form_id>", methods=["GET"])
@login_required
def api_get_form(form_id):
    form = get_form(form_id)
    if not form:
        return jsonify({"ok": False}), 404
    data = form.to_dict()
    data["fields"] = [f.to_dict() for f in form.fields]
    return jsonify({"ok": True, "form": data})


# ---------------------------------------------------------------------------
# API — Get responses
# ---------------------------------------------------------------------------

@forms_bp.route("/api/forms/<int:form_id>/responses", methods=["GET"])
@login_required
def api_get_responses(form_id):
    form = get_form(form_id)
    if not form or form.created_by != current_user.id:
        return jsonify({"ok": False}), 403
    return jsonify({"ok": True, "responses": [r.to_dict() for r in form.responses]})
