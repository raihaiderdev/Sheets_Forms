"""
Passport photo processing service.
Pipeline: load → orient → detect face → crop → resize → white bg → compress 10-18 KB
Background removal uses a simple edge-aware approach (no GrabCut — unreliable on headless servers).
"""
import io
import cv2
import numpy as np
from PIL import Image, ImageOps, ExifTags


PASSPORT_W = 413
PASSPORT_H = 531
TARGET_DPI = 300
MIN_KB = 10
MAX_KB = 18

HEAD_RATIO  = 0.70   # face height / total crop height
TOP_MARGIN  = 0.13   # space above face top / total crop height


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_pil(file_bytes: bytes) -> Image.Image:
    """Load image, auto-rotate by EXIF orientation."""
    img = Image.open(io.BytesIO(file_bytes))
    img = ImageOps.exif_transpose(img)   # fix phone portrait orientation
    img = img.convert("RGB")
    return img


def _pil_to_cv2(pil_img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def _cv2_to_pil(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


# ---------------------------------------------------------------------------
# Face detection
# ---------------------------------------------------------------------------

def _detect_face(bgr: np.ndarray):
    """Return (x, y, w, h) or None. Tries Haar cascade only — reliable everywhere."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    # Try to load Haar cascade
    try:
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(path)
        if not cascade.empty():
            for (sf, mn) in [(1.1, 5), (1.05, 3), (1.03, 2)]:
                faces = cascade.detectMultiScale(
                    gray, scaleFactor=sf, minNeighbors=mn, minSize=(40, 40)
                )
                if len(faces) > 0:
                    return tuple(map(int, max(faces, key=lambda f: f[2] * f[3])))
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Background removal — simple threshold-based approach (no GrabCut)
# ---------------------------------------------------------------------------

def _white_background(pil_img: Image.Image) -> Image.Image:
    """
    Replace background with white using a simple grab-cut-free approach:
    1. Convert to LAB colour space
    2. Use k-means (k=2) to separate fg/bg
    3. Composite subject onto white canvas

    Falls back to returning the image on white canvas untouched if anything fails.
    """
    try:
        bgr = _pil_to_cv2(pil_img)
        h, w = bgr.shape[:2]

        # Downscale for speed if large
        scale = 1.0
        if w > 600:
            scale = 600 / w
            small = cv2.resize(bgr, (int(w * scale), int(h * scale)))
        else:
            small = bgr.copy()

        # Convert to float for k-means
        data = small.reshape((-1, 3)).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(
            data, 2, None, criteria, 3, cv2.KMEANS_PP_CENTERS
        )

        # The background cluster is the one whose centroid is closest to the corners
        labels_img = labels.reshape(small.shape[:2])
        sh, sw = labels_img.shape

        # Sample corners
        corner_pixels = [
            labels_img[0, 0], labels_img[0, sw-1],
            labels_img[sh-1, 0], labels_img[sh-1, sw-1]
        ]
        from collections import Counter
        bg_label = Counter(corner_pixels).most_common(1)[0][0]
        fg_label = 1 - bg_label

        # Build mask at original size
        if scale != 1.0:
            mask_small = (labels_img == fg_label).astype(np.uint8) * 255
            mask = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            mask = (labels_img == fg_label).astype(np.uint8) * 255

        # Morphological cleanup
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)

        # Composite on white
        alpha = mask.astype(np.float32) / 255.0
        white = np.ones_like(bgr, dtype=np.float32) * 255
        orig  = bgr.astype(np.float32)
        composite = (orig * alpha[:, :, None] + white * (1 - alpha[:, :, None])).astype(np.uint8)

        return _cv2_to_pil(composite)

    except Exception:
        # Fallback: just put on white canvas
        canvas = Image.new("RGB", pil_img.size, (255, 255, 255))
        canvas.paste(pil_img)
        return canvas


# ---------------------------------------------------------------------------
# Face-centred crop
# ---------------------------------------------------------------------------

def _crop_portrait(pil_img: Image.Image, face) -> Image.Image:
    """Crop image around detected face using passport proportions."""
    iw, ih = pil_img.size
    fx, fy, fw, fh = face

    crop_h = int(fh / HEAD_RATIO)
    crop_w = int(crop_h * PASSPORT_W / PASSPORT_H)

    cx = fx + fw // 2
    top  = fy - int(TOP_MARGIN * crop_h)
    left = cx - crop_w // 2

    # Clamp
    pad_top    = max(0, -top)
    pad_left   = max(0, -left)
    pad_bottom = max(0, (top + crop_h) - ih)
    pad_right  = max(0, (left + crop_w) - iw)

    crop = pil_img.crop((
        max(0, left), max(0, top),
        min(iw, left + crop_w), min(ih, top + crop_h)
    ))

    if pad_top or pad_left or pad_bottom or pad_right:
        canvas = Image.new("RGB",
            (crop.width + pad_left + pad_right, crop.height + pad_top + pad_bottom),
            (255, 255, 255)
        )
        canvas.paste(crop, (pad_left, pad_top))
        crop = canvas

    return crop


# ---------------------------------------------------------------------------
# Compression
# ---------------------------------------------------------------------------

def _compress_to_range(pil_img: Image.Image,
                       min_kb: int = MIN_KB, max_kb: int = MAX_KB) -> bytes:
    ow, oh = pil_img.size

    def _enc(img, q):
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=q,
                 dpi=(TARGET_DPI, TARGET_DPI), optimize=True)
        return buf.getvalue()

    # Phase 1: quality sweep at full size
    for q in range(10, 1, -1):
        data = _enc(pil_img, q)
        kb = len(data) / 1024
        if min_kb <= kb <= max_kb:
            return data

    # Phase 2: shrink dimensions
    for scale in [0.85, 0.70, 0.58, 0.48, 0.40, 0.34, 0.28, 0.24, 0.20, 0.16]:
        nw, nh = max(int(ow * scale), 20), max(int(oh * scale), 26)
        small = pil_img.resize((nw, nh), Image.LANCZOS)
        for q in range(85, 2, -5):
            data = _enc(small, q)
            kb = len(data) / 1024
            if min_kb <= kb <= max_kb:
                return data
            if kb < min_kb:
                prev = _enc(small, min(q + 5, 95))
                if len(prev) / 1024 <= max_kb:
                    return prev
                break

    return _enc(pil_img.resize((max(int(ow * 0.18), 20), max(int(oh * 0.18), 26)), Image.LANCZOS), 50)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def process_passport_photo(file_bytes: bytes) -> dict:
    # 1. Load + fix orientation
    pil_img = _load_pil(file_bytes)

    # 2. Detect face
    bgr = _pil_to_cv2(pil_img)
    face = _detect_face(bgr)
    face_found = face is not None

    # 3. Background → white
    pil_white = _white_background(pil_img)

    # 4. Crop around face
    if face_found:
        pil_white = _crop_portrait(pil_white, face)

    # 5. Resize to passport dimensions
    pil_resized = pil_white.resize((PASSPORT_W, PASSPORT_H), Image.LANCZOS)

    # 6. Ensure pure white canvas
    canvas = Image.new("RGB", (PASSPORT_W, PASSPORT_H), (255, 255, 255))
    canvas.paste(pil_resized)

    # 7. Compress to 10–18 KB
    final_bytes = _compress_to_range(canvas, MIN_KB, MAX_KB)
    size_kb = round(len(final_bytes) / 1024, 1)

    return {
        "ok": True,
        "image": final_bytes,
        "face_found": face_found,
        "size_kb": size_kb,
    }
