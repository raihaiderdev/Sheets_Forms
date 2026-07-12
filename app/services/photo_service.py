"""
Passport photo processing service.
Pipeline:
  1. Load + fix EXIF orientation
  2. Detect face (Haar cascade)
  3. Remove background via remove.bg API → falls back to GrabCut
  4. Crop to face with passport proportions
  5. Resize to 413×531 px (35×45 mm @ 300 DPI)
  6. Paste on white canvas
  7. Compress to 10–18 KB
"""
import io
import os
import cv2
import numpy as np
import urllib.request
import urllib.parse
from PIL import Image, ImageOps


PASSPORT_W = 413
PASSPORT_H = 531
TARGET_DPI = 300
MIN_KB     = 10
MAX_KB     = 18
HEAD_RATIO = 0.68
TOP_MARGIN = 0.13

REMOVEBG_URL = "https://api.remove.bg/v1.0/removebg"


# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------

def _load_pil(file_bytes: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(file_bytes))
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def _pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(pil_img, dtype=np.uint8), cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------
# 2. Face detection
# ---------------------------------------------------------------------------

def _detect_face(bgr: np.ndarray):
    """Return (x, y, w, h) of largest face, or None."""
    try:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        cv2.equalizeHist(gray, gray)
        path    = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(path)
        if cascade.empty():
            return None
        for sf, mn in [(1.1, 5), (1.05, 3), (1.03, 2)]:
            faces = cascade.detectMultiScale(
                gray, scaleFactor=sf, minNeighbors=mn, minSize=(40, 40)
            )
            if len(faces) > 0:
                return tuple(int(v) for v in max(faces, key=lambda f: f[2] * f[3]))
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# 3a. Background removal via remove.bg API
# ---------------------------------------------------------------------------

def _removebg_api(file_bytes: bytes) -> Image.Image | None:
    """
    Call remove.bg API. Returns RGBA PIL image on success, None on failure.
    """
    api_key = os.environ.get("REMOVEBG_API_KEY", "").strip()
    if not api_key:
        return None

    try:
        import urllib.request, urllib.error

        # Multipart form-data POST
        boundary = b"----FormBoundary7MA4YWxkTrZu0gW"
        body  = b"--" + boundary + b"\r\n"
        body += b'Content-Disposition: form-data; name="image_file"; filename="image.jpg"\r\n'
        body += b'Content-Type: image/jpeg\r\n\r\n'
        body += file_bytes + b"\r\n"
        body += b"--" + boundary + b"\r\n"
        body += b'Content-Disposition: form-data; name="size"\r\n\r\nfull\r\n'
        body += b"--" + boundary + b"--\r\n"

        req = urllib.request.Request(
            REMOVEBG_URL,
            data=body,
            headers={
                "X-Api-Key": api_key,
                "Content-Type": f"multipart/form-data; boundary={boundary.decode()}",
            },
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=30) as resp:
            result_bytes = resp.read()

        img = Image.open(io.BytesIO(result_bytes)).convert("RGBA")
        return img

    except Exception:
        return None


# ---------------------------------------------------------------------------
# 3b. Fallback: face-seeded GrabCut
# ---------------------------------------------------------------------------

def _grabcut_remove_bg(pil_img: Image.Image, face=None) -> Image.Image:
    """GrabCut with face-region seeding. Returns image on white canvas."""
    try:
        orig_w, orig_h = pil_img.size
        scale  = min(1.0, 600 / max(orig_w, orig_h))
        work_w = max(int(orig_w * scale), 10)
        work_h = max(int(orig_h * scale), 10)
        small  = pil_img.resize((work_w, work_h), Image.LANCZOS)
        bgr    = cv2.cvtColor(np.array(small, dtype=np.uint8), cv2.COLOR_RGB2BGR)
        h, w   = bgr.shape[:2]

        mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)

        # Corners = definite background
        c = int(min(w, h) * 0.12)
        mask[:c, :c] = mask[:c, -c:] = mask[-c:, :c] = mask[-c:, -c:] = cv2.GC_BGD

        # Centre strip = probable foreground
        mask[int(h*.10):int(h*.90), int(w*.25):int(w*.75)] = cv2.GC_PR_FGD

        if face is not None:
            fx, fy, fw, fh = face
            sfx, sfy = max(0, int(fx*scale)), max(0, int(fy*scale))
            sfw, sfh = max(1, int(fw*scale)), max(1, int(fh*scale))
            mask[sfy:sfy+sfh, sfx:sfx+sfw] = cv2.GC_FGD
            mask[sfy+sfh:min(h, sfy+sfh*4),
                 max(0, sfx-sfh//2):min(w, sfx+sfw+sfh//2)] = cv2.GC_PR_FGD

        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)
        cv2.grabCut(bgr, mask, None, bgd, fgd, 5, cv2.GC_INIT_WITH_MASK)

        fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        k  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k, iterations=3)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  k, iterations=1)
        fg = cv2.GaussianBlur(fg, (7, 7), 0)

        if scale < 1.0:
            fg = cv2.resize(fg, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        orig_f  = np.array(pil_img, dtype=np.float32)
        alpha_f = fg.astype(np.float32) / 255.0
        white_f = np.ones_like(orig_f) * 255.0
        result  = (orig_f * alpha_f[:,:,None] + white_f*(1-alpha_f[:,:,None])).astype(np.uint8)
        return Image.fromarray(result, "RGB")

    except Exception:
        canvas = Image.new("RGB", pil_img.size, (255, 255, 255))
        canvas.paste(pil_img)
        return canvas


# ---------------------------------------------------------------------------
# 3. Remove background (try API first, fallback to GrabCut)
# ---------------------------------------------------------------------------

def _remove_background(pil_img: Image.Image, face=None) -> Image.Image:
    # Try remove.bg API first
    rgba = _removebg_api(_pil_to_jpeg_bytes(pil_img))
    if rgba is not None:
        # Composite RGBA onto white
        canvas = Image.new("RGB", rgba.size, (255, 255, 255))
        canvas.paste(rgba, mask=rgba.split()[3])
        # Resize to match original if API returns different size
        if canvas.size != pil_img.size:
            canvas = canvas.resize(pil_img.size, Image.LANCZOS)
        return canvas

    # Fallback: GrabCut
    return _grabcut_remove_bg(pil_img, face)


def _pil_to_jpeg_bytes(pil_img: Image.Image, quality: int = 92) -> bytes:
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# 4. Face-centred crop
# ---------------------------------------------------------------------------

def _crop_portrait(pil_img: Image.Image, face) -> Image.Image:
    iw, ih = pil_img.size
    fx, fy, fw, fh = face

    crop_h = int(fh / HEAD_RATIO)
    crop_w = int(crop_h * PASSPORT_W / PASSPORT_H)
    cx     = fx + fw // 2
    top    = fy - int(TOP_MARGIN * crop_h)
    left   = cx - crop_w // 2

    pad_t = max(0, -top)
    pad_l = max(0, -left)
    pad_b = max(0, (top + crop_h) - ih)
    pad_r = max(0, (left + crop_w) - iw)

    crop = pil_img.crop((
        max(0, left), max(0, top),
        min(iw, left + crop_w), min(ih, top + crop_h)
    ))

    if any([pad_t, pad_l, pad_b, pad_r]):
        canvas = Image.new("RGB",
            (crop.width + pad_l + pad_r, crop.height + pad_t + pad_b),
            (255, 255, 255)
        )
        canvas.paste(crop, (pad_l, pad_t))
        return canvas
    return crop


# ---------------------------------------------------------------------------
# 5. Compression
# ---------------------------------------------------------------------------

def _compress_to_range(pil_img: Image.Image) -> bytes:
    ow, oh = pil_img.size

    def enc(img, q):
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=q,
                 dpi=(TARGET_DPI, TARGET_DPI), optimize=True)
        return buf.getvalue()

    for q in range(10, 1, -1):
        data = enc(pil_img, q)
        if MIN_KB <= len(data)/1024 <= MAX_KB:
            return data

    for scale in [0.85, 0.70, 0.58, 0.48, 0.40, 0.33, 0.27, 0.22, 0.18]:
        nw, nh = max(int(ow*scale), 20), max(int(oh*scale), 26)
        small  = pil_img.resize((nw, nh), Image.LANCZOS)
        for q in range(85, 2, -5):
            data = enc(small, q)
            kb   = len(data) / 1024
            if MIN_KB <= kb <= MAX_KB:
                return data
            if kb < MIN_KB:
                prev = enc(small, min(q+5, 95))
                if len(prev)/1024 <= MAX_KB:
                    return prev
                break

    return enc(pil_img.resize(
        (max(int(ow*0.18), 20), max(int(oh*0.18), 26)), Image.LANCZOS
    ), 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_passport_photo(file_bytes: bytes) -> dict:
    pil  = _load_pil(file_bytes)
    bgr  = _pil_to_bgr(pil)
    face = _detect_face(bgr)

    pil  = _remove_background(pil, face=face)

    if face is not None:
        pil = _crop_portrait(pil, face)

    pil = pil.resize((PASSPORT_W, PASSPORT_H), Image.LANCZOS)

    canvas = Image.new("RGB", (PASSPORT_W, PASSPORT_H), (255, 255, 255))
    canvas.paste(pil)

    final_bytes = _compress_to_range(canvas)

    return {
        "ok":         True,
        "image":      final_bytes,
        "face_found": face is not None,
        "size_kb":    round(len(final_bytes) / 1024, 1),
    }
