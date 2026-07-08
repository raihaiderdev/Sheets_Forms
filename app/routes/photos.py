import io
import zipfile
from flask import Blueprint, render_template, request, jsonify, make_response, send_file
from flask_login import login_required, current_user

from app.services.photo_service import process_passport_photo

photos_bp = Blueprint("photos", __name__, template_folder="../templates")

ALLOWED_EXTS   = {"jpg", "jpeg", "png", "bmp", "webp", "tiff"}
MAX_FILE_BYTES = 20 * 1024 * 1024   # 20 MB per image
MAX_FILES      = 100


def _ext_ok(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTS


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

@photos_bp.route("/photos")
@login_required
def photo_tool():
    return render_template("photos/photo_tool.html", user=current_user)


# ---------------------------------------------------------------------------
# API — Process single image
# ---------------------------------------------------------------------------

@photos_bp.route("/api/photos/process", methods=["POST"])
@login_required
def api_process_photo():
    if "file" not in request.files:
        return jsonify({"ok": False, "error": "No file provided."}), 400

    f = request.files["file"]
    if not f.filename or not _ext_ok(f.filename):
        return jsonify({"ok": False, "error": "Unsupported file type."}), 400

    data = f.read(MAX_FILE_BYTES + 1)
    if len(data) > MAX_FILE_BYTES:
        return jsonify({"ok": False, "error": "File too large (max 20 MB)."}), 413

    try:
        result = process_passport_photo(data)
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 422
    except Exception as exc:
        return jsonify({"ok": False, "error": f"Processing failed: {exc}"}), 500

    # Return image as base64 for preview + download
    import base64
    b64 = base64.b64encode(result["image"]).decode()

    return jsonify({
        "ok":         True,
        "image_b64":  b64,
        "face_found": result["face_found"],
        "size_kb":    result["size_kb"],
        "filename":   f.filename,
    })


# ---------------------------------------------------------------------------
# API — Process batch and return ZIP
# ---------------------------------------------------------------------------

@photos_bp.route("/api/photos/batch", methods=["POST"])
@login_required
def api_batch_photos():
    files = request.files.getlist("files")
    if not files:
        return jsonify({"ok": False, "error": "No files uploaded."}), 400
    if len(files) > MAX_FILES:
        return jsonify({"ok": False, "error": f"Maximum {MAX_FILES} images at a time."}), 400

    zip_buf = io.BytesIO()
    results = []

    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for idx, f in enumerate(files, 1):
            if not f.filename or not _ext_ok(f.filename):
                results.append({"filename": f.filename, "ok": False, "error": "Unsupported type"})
                continue

            data = f.read(MAX_FILE_BYTES + 1)
            if len(data) > MAX_FILE_BYTES:
                results.append({"filename": f.filename, "ok": False, "error": "Too large"})
                continue

            try:
                res = process_passport_photo(data)
                # Save as passport_001_originalname.jpg
                stem = f.filename.rsplit(".", 1)[0]
                out_name = f"passport_{idx:03d}_{stem}.jpg"
                zf.writestr(out_name, res["image"])
                results.append({
                    "filename":   f.filename,
                    "out_name":   out_name,
                    "ok":         True,
                    "face_found": res["face_found"],
                    "size_kb":    res["size_kb"],
                })
            except Exception as exc:
                results.append({"filename": f.filename, "ok": False, "error": str(exc)})

    zip_buf.seek(0)
    response = make_response(zip_buf.read())
    response.headers["Content-Type"] = "application/zip"
    response.headers["Content-Disposition"] = 'attachment; filename="passport_photos.zip"'
    return response
