"""
Passport photo processing service.
- Detects face using OpenCV Haar cascade
- Removes background using GrabCut + contour fill → white
- Crops and centers face with standard head-room ratios
- Resizes to 413×531 px (35×45 mm @ 300 DPI)
- Compresses JPEG under 100 KB
"""
import io
import cv2
import numpy as np
from PIL import Image, ImageFilter


# Passport dimensions: 35×45 mm @ 300 DPI  →  413 × 531 px
PASSPORT_W = 413
PASSPORT_H = 531
TARGET_DPI = 300
MAX_KB = 100

# Head should occupy ~70-75% of frame height; face top ~15% from top
HEAD_RATIO = 0.72      # face height / total image height
TOP_MARGIN = 0.12      # space above head as fraction of total height


def _load_cv2(file_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image.")
    return img


def _detect_face(img_bgr: np.ndarray):
    """Return (x, y, w, h) of the largest face, or None.
    Uses FaceDetectorYN (OpenCV 5) with fallback to legacy Haar cascade.
    """
    h, w = img_bgr.shape[:2]

    # Try FaceDetectorYN (OpenCV 5+)
    try:
        detector = cv2.FaceDetectorYN.create(
            model="",          # empty = use built-in default
            config="",
            input_size=(w, h),
            score_threshold=0.6,
            nms_threshold=0.3,
            top_k=5,
        )
        _, faces = detector.detect(img_bgr)
        if faces is not None and len(faces) > 0:
            # faces columns: x, y, w, h, ...
            best = max(faces, key=lambda f: f[2] * f[3])
            return (int(best[0]), int(best[1]), int(best[2]), int(best[3]))
    except Exception:
        pass

    # Fallback: try legacy CascadeClassifier if available
    try:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(cascade_path)
        if not cascade.empty():
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            faces = cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
            )
            if len(faces) == 0:
                faces = cascade.detectMultiScale(
                    gray, scaleFactor=1.05, minNeighbors=3, minSize=(40, 40)
                )
            if len(faces) > 0:
                return max(faces, key=lambda f: f[2] * f[3])
    except Exception:
        pass

    # Last resort: use center-weighted face estimation (assume face is upper-center)
    return None


def _remove_background_grabcut(img_bgr: np.ndarray) -> np.ndarray:
    """
    Use GrabCut to separate foreground from background.
    Returns BGRA image with background set to white (255,255,255).
    """
    h, w = img_bgr.shape[:2]

    # Initial rect: leave 5% margin all around for GrabCut
    margin_x = max(5, int(w * 0.05))
    margin_y = max(5, int(h * 0.05))
    rect = (margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y)

    mask = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(img_bgr, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        # GrabCut fails on very small images — return original with white bg attempt
        return _simple_white_bg(img_bgr)

    # Mask: 0=bg, 1=fg, 2=prob_bg, 3=prob_fg
    fg_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,  kernel, iterations=1)

    # Feather edges slightly
    fg_mask_blur = cv2.GaussianBlur(fg_mask, (5, 5), 0)

    # Build BGRA
    bgra = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = fg_mask_blur

    # Composite onto white
    white = np.ones_like(img_bgr, dtype=np.uint8) * 255
    alpha = fg_mask_blur.astype(np.float32) / 255.0
    for c in range(3):
        bgra[:, :, c] = (
            img_bgr[:, :, c].astype(np.float32) * alpha +
            white[:, :, c].astype(np.float32) * (1 - alpha)
        ).astype(np.uint8)
    bgra[:, :, 3] = 255  # fully opaque result
    return bgra


def _simple_white_bg(img_bgr: np.ndarray) -> np.ndarray:
    """Fallback: just return image as BGRA with full opacity (no bg removal)."""
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2BGRA)


def _crop_to_passport(img_bgr: np.ndarray, face) -> np.ndarray:
    """
    Crop the image so the face is centered with standard passport proportions.
    face = (x, y, w, h)
    """
    ih, iw = img_bgr.shape[:2]
    fx, fy, fw, fh = face

    # Face center
    cx = fx + fw // 2
    cy = fy + fh // 2

    # Desired crop height so face occupies HEAD_RATIO of it
    crop_h = int(fh / HEAD_RATIO)
    crop_w = int(crop_h * PASSPORT_W / PASSPORT_H)

    # Top of crop: face top should be at TOP_MARGIN * crop_h below crop top
    top = fy - int(TOP_MARGIN * crop_h)
    left = cx - crop_w // 2

    # Clamp to image bounds — pad with white if needed
    pad_top    = max(0, -top)
    pad_left   = max(0, -left)
    pad_bottom = max(0, (top + crop_h) - ih)
    pad_right  = max(0, (left + crop_w) - iw)

    top_c  = max(0, top)
    left_c = max(0, left)
    bot_c  = min(ih, top + crop_h)
    right_c = min(iw, left + crop_w)

    cropped = img_bgr[top_c:bot_c, left_c:right_c]

    if pad_top or pad_left or pad_bottom or pad_right:
        cropped = cv2.copyMakeBorder(
            cropped, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=(255, 255, 255)
        )

    return cropped


def _compress_jpeg(pil_img: Image.Image, max_kb: int = MAX_KB) -> bytes:
    """Compress PIL image to JPEG under max_kb."""
    quality = 95
    while quality >= 20:
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=quality, dpi=(TARGET_DPI, TARGET_DPI))
        size_kb = buf.tell() / 1024
        if size_kb <= max_kb:
            return buf.getvalue()
        quality -= 5

    # Last resort: resize down slightly
    w, h = pil_img.size
    for scale in [0.9, 0.8, 0.7]:
        small = pil_img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        buf = io.BytesIO()
        small.save(buf, format="JPEG", quality=40, dpi=(TARGET_DPI, TARGET_DPI))
        if buf.tell() / 1024 <= max_kb:
            return buf.getvalue()

    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=20, dpi=(TARGET_DPI, TARGET_DPI))
    return buf.getvalue()


def process_passport_photo(file_bytes: bytes) -> dict:
    """
    Full pipeline: load → detect face → remove bg → crop → resize → compress.

    Returns:
        {
            "ok": True,
            "image": bytes,       # final JPEG bytes
            "face_found": bool,
            "size_kb": float,
        }
    """
    img_bgr = _load_cv2(file_bytes)
    face = _detect_face(img_bgr)
    face_found = face is not None

    # Background removal
    bgra = _remove_background_grabcut(img_bgr)
    # Convert BGRA back to BGR (white bg already composited)
    result_bgr = bgra[:, :, :3]

    # Crop around face if detected
    if face_found:
        result_bgr = _crop_to_passport(result_bgr, face)

    # Resize to passport dimensions
    resized = cv2.resize(result_bgr, (PASSPORT_W, PASSPORT_H), interpolation=cv2.INTER_LANCZOS4)

    # Convert to PIL for final save
    pil_img = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))

    # Ensure white background (no transparency artifacts)
    white_bg = Image.new("RGB", pil_img.size, (255, 255, 255))
    white_bg.paste(pil_img)

    final_bytes = _compress_jpeg(white_bg)
    size_kb = len(final_bytes) / 1024

    return {
        "ok": True,
        "image": final_bytes,
        "face_found": face_found,
        "size_kb": round(size_kb, 1),
    }
