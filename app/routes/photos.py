import io
import zipfile
from flask import Blueprint, render_template, request, jsonify, make_response, send_file
from flask_login import login_required, current_user

from app.services.photo_service import process_passport_photo

photos_bp = Blueprint("photos", __name__, template_folder="../templates")

ALLOWED_EXTS   = {"jpg", "jpeg", "png", "bmp", "webp", "tiff", "tif", "heic", "heif"}
MAX_FILE_BYTES = 25 * 1024 * 1024   # 25 MB per image
MAX_FILES      = 100


def _ext_ok(filename: str) -> bool:
    """Case-insensitive extension check. Also accepts files with no extension (try anyway)."""
    if not filename:
        return False
    if "." not in filename:
        return True   # no extension — let the image decoder decide
    ext = filename.rsplit(".", 1)[1].lower().strip()
    return ext in ALLOWED_EXTS


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
        return jsonify({"ok": False, "error": "File too large (max 25 MB)."}), 413

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

@photos_bp.route("/api/photos/compress", methods=["POST"])
@login_required
def api_compress_photo():
    """Compress image to 10-25 KB without resizing or passport conversion."""
    if "file" not in request.files:
        return jsonify({"ok": False, "error": "No file provided."}), 400

    f = request.files["file"]
    if not f.filename or not _ext_ok(f.filename):
        return jsonify({"ok": False, "error": "Unsupported file type."}), 400

    data = f.read(MAX_FILE_BYTES + 1)
    if len(data) > MAX_FILE_BYTES:
        return jsonify({"ok": False, "error": "File too large (max 25 MB)."}), 413

    try:
        from PIL import Image, ImageOps
        import io as _io

        MIN_KB, MAX_KB = 10, 25

        img = Image.open(_io.BytesIO(data))
        img = ImageOps.exif_transpose(img).convert("RGB")

        def enc(im, q):
            buf = _io.BytesIO()
            im.save(buf, format="JPEG", quality=q, optimize=True)
            return buf.getvalue()

        orig_w, orig_h = img.size
        result = None

        # Phase 1: quality sweep at full resolution
        for q in range(10, 1, -1):
            d = enc(img, q)
            kb = len(d) / 1024
            if MIN_KB <= kb <= MAX_KB:
                result = d
                break

        # Phase 2: shrink if still too large
        if result is None:
            for scale in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.35, 0.3, 0.25, 0.20]:
                nw = max(int(orig_w * scale), 30)
                nh = max(int(orig_h * scale), 30)
                small = img.resize((nw, nh), Image.LANCZOS)
                for q in range(90, 2, -5):
                    d = enc(small, q)
                    kb = len(d) / 1024
                    if MIN_KB <= kb <= MAX_KB:
                        result = d
                        break
                if result:
                    break

        if result is None:
            result = enc(img.resize(
                (max(int(orig_w * 0.20), 30), max(int(orig_h * 0.20), 30)),
                Image.LANCZOS
            ), 60)

        import base64
        b64 = base64.b64encode(result).decode()
        size_kb = round(len(result) / 1024, 1)

        return jsonify({
            "ok": True,
            "image_b64": b64,
            "size_kb": size_kb,
            "filename": f.filename,
        })

    except Exception as exc:
        return jsonify({"ok": False, "error": f"Processing failed: {exc}"}), 500


@photos_bp.route("/api/photos/compress-batch", methods=["POST"])
@login_required
def api_compress_batch():
    """Compress multiple images and return as ZIP."""
    files = request.files.getlist("files")
    if not files:
        return jsonify({"ok": False, "error": "No files uploaded."}), 400
    if len(files) > MAX_FILES:
        return jsonify({"ok": False, "error": f"Maximum {MAX_FILES} images."}), 400

    from PIL import Image, ImageOps
    import io as _io

    MIN_KB, MAX_KB = 10, 25
    zip_buf = io.BytesIO()

    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for idx, f in enumerate(files, 1):
            if not f.filename or not _ext_ok(f.filename):
                continue
            raw = f.read(MAX_FILE_BYTES + 1)
            if len(raw) > MAX_FILE_BYTES:
                continue
            try:
                img = Image.open(_io.BytesIO(raw))
                img = ImageOps.exif_transpose(img).convert("RGB")
                orig_w, orig_h = img.size

                def enc(im, q):
                    buf = _io.BytesIO()
                    im.save(buf, format="JPEG", quality=q, optimize=True)
                    return buf.getvalue()

                result = None
                for q in range(10, 1, -1):
                    d = enc(img, q)
                    if MIN_KB <= len(d)/1024 <= MAX_KB:
                        result = d; break

                if result is None:
                    for scale in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.35, 0.3, 0.25, 0.20]:
                        small = img.resize((max(int(orig_w*scale),30), max(int(orig_h*scale),30)), Image.LANCZOS)
                        for q in range(90, 2, -5):
                            d = enc(small, q)
                            if MIN_KB <= len(d)/1024 <= MAX_KB:
                                result = d; break
                        if result: break

                if result is None:
                    result = enc(img.resize((max(int(orig_w*.20),30), max(int(orig_h*.20),30)), Image.LANCZOS), 60)

                stem = f.filename.rsplit(".", 1)[0]
                zf.writestr(f"compressed_{idx:03d}_{stem}.jpg", result)
            except Exception:
                continue

    zip_buf.seek(0)
    response = make_response(zip_buf.read())
    response.headers["Content-Type"] = "application/zip"
    response.headers["Content-Disposition"] = 'attachment; filename="compressed_images.zip"'
    return response
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
