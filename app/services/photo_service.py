"""
Passport photo processing service.
Pipeline: load → fix EXIF orientation → detect face → remove background
          → crop to face → resize 413×531 px → compress 10-18 KB @ 300 DPI
"""
import io
import cv2
import numpy as np
from PIL import Image, ImageOps


PASSPORT_W = 413
PASSPORT_H = 531
TARGET_DPI = 300
MIN_KB     = 10
MAX_KB     = 18
HEAD_RATIO = 0.68   # face height / total crop height
TOP_MARGIN = 0.13   # gap above face as fraction of crop height


# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------

def _load_pil(file_bytes: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(file_bytes))
    img = ImageOps.exif_transpose(img)   # fix phone portrait rotation
    return img.convert("RGB")


def _pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(pil_img, dtype=np.uint8), cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------
# 2. Face detection
# ---------------------------------------------------------------------------

def _detect_face(bgr: np.ndarray):
    """Return (x, y, w, h) of the largest face or None."""
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
# 3. Background removal
# ---------------------------------------------------------------------------

def _remove_background(pil_img: Image.Image, face=None) -> Image.Image:
    """
    Improved background removal using face-seeded GrabCut.
    - Marks face region as definite foreground
    - Marks image corners as definite background
    - Uses GrabCut on downscaled image for speed
    Falls back to plain white canvas if anything fails.
    """
    try:
        orig_w, orig_h = pil_img.size

        # Downscale to ≤600px for speed
        max_dim = 600
        scale   = min(1.0, max_dim / max(orig_w, orig_h))
        work_w  = max(int(orig_w * scale), 10)
        work_h  = max(int(orig_h * scale), 10)
        small   = pil_img.resize((work_w, work_h), Image.LANCZOS)

        bgr  = cv2.cvtColor(np.array(small, dtype=np.uint8), cv2.COLOR_RGB2BGR)
        h, w = bgr.shape[:2]

        # --- Build seeded mask ---
        # Start with GrabCut probable background everywhere
        mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)

        # Mark corners (15% from each edge) as definite background
        corner = int(min(w, h) * 0.12)
        mask[:corner, :corner]   = cv2.GC_BGD  # TL
        mask[:corner, -corner:]  = cv2.GC_BGD  # TR
        mask[-corner:, :corner]  = cv2.GC_BGD  # BL
        mask[-corner:, -corner:] = cv2.GC_BGD  # BR

        # The vertical strip in the centre of the image is probable foreground
        cx_start = int(w * 0.25)
        cx_end   = int(w * 0.75)
        cy_start = int(h * 0.10)
        cy_end   = int(h * 0.90)
        mask[cy_start:cy_end, cx_start:cx_end] = cv2.GC_PR_FGD

        # If face was detected, mark face bounding box as definite foreground
        if face is not None:
            fx, fy, fw, fh = face
            # Scale face coords to work size
            sfx = max(0, int(fx * scale))
            sfy = max(0, int(fy * scale))
            sfw = max(1, int(fw * scale))
            sfh = max(1, int(fh * scale))
            efx = min(w, sfx + sfw)
            efy = min(h, sfy + sfh)
            mask[sfy:efy, sfx:efx] = cv2.GC_FGD
            # Also mark a body region below face as probable foreground
            body_top = min(h, efy)
            body_bot = min(h, body_top + sfh * 3)
            body_l   = max(0, sfx - sfh // 2)
            body_r   = min(w, efx + sfh // 2)
            mask[body_top:body_bot, body_l:body_r] = cv2.GC_PR_FGD

        bgd = np.zeros((1, 65), dtype=np.float64)
        fgd = np.zeros((1, 65), dtype=np.float64)

        cv2.grabCut(bgr, mask, None, bgd, fgd, 5, cv2.GC_INIT_WITH_MASK)

        fg = np.where(
            (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0
        ).astype(np.uint8)

        # Morphological cleanup
        k  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k, iterations=3)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  k, iterations=1)
        fg = cv2.GaussianBlur(fg, (7, 7), 0)

        # Upscale mask to original size
        if scale < 1.0:
            fg = cv2.resize(fg, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        # Composite on white
        orig_f  = np.array(pil_img, dtype=np.float32)
        alpha_f = fg.astype(np.float32) / 255.0
        white_f = np.ones_like(orig_f) * 255.0
        result  = (orig_f * alpha_f[:, :, None] + white_f * (1 - alpha_f[:, :, None])).astype(np.uint8)
        return Image.fromarray(result, "RGB")

    except Exception:
        canvas = Image.new("RGB", pil_img.size, (255, 255, 255))
        canvas.paste(pil_img)
        return canvas


# ---------------------------------------------------------------------------
# 4. Face-centred crop
# ---------------------------------------------------------------------------

def _crop_portrait(pil_img: Image.Image, face) -> Image.Image:
    """Crop image centred on face with passport proportions."""
    iw, ih = pil_img.size
    fx, fy, fw, fh = face

    crop_h = int(fh / HEAD_RATIO)
    crop_w = int(crop_h * PASSPORT_W / PASSPORT_H)

    cx   = fx + fw // 2
    top  = fy - int(TOP_MARGIN * crop_h)
    left = cx - crop_w // 2

    pad_top    = max(0, -top)
    pad_left   = max(0, -left)
    pad_bottom = max(0, (top + crop_h) - ih)
    pad_right  = max(0, (left + crop_w) - iw)

    crop = pil_img.crop((
        max(0, left),       max(0, top),
        min(iw, left + crop_w), min(ih, top + crop_h)
    ))

    if any([pad_top, pad_left, pad_bottom, pad_right]):
        canvas = Image.new("RGB",
            (crop.width + pad_left + pad_right,
             crop.height + pad_top + pad_bottom),
            (255, 255, 255)
        )
        canvas.paste(crop, (pad_left, pad_top))
        return canvas

    return crop


# ---------------------------------------------------------------------------
# 5. Compression
# ---------------------------------------------------------------------------

def _compress_to_range(pil_img: Image.Image) -> bytes:
    """Compress to JPEG between MIN_KB and MAX_KB with 300 DPI tag."""
    ow, oh = pil_img.size

    def enc(img, q):
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=q,
                 dpi=(TARGET_DPI, TARGET_DPI), optimize=True)
        return buf.getvalue()

    # Phase 1: quality sweep at full resolution
    for q in range(10, 1, -1):
        data = enc(pil_img, q)
        kb   = len(data) / 1024
        if MIN_KB <= kb <= MAX_KB:
            return data

    # Phase 2: shrink dimensions
    for scale in [0.85, 0.70, 0.58, 0.48, 0.40, 0.33, 0.27, 0.22, 0.18]:
        nw, nh = max(int(ow * scale), 20), max(int(oh * scale), 26)
        small  = pil_img.resize((nw, nh), Image.LANCZOS)
        for q in range(85, 2, -5):
            data = enc(small, q)
            kb   = len(data) / 1024
            if MIN_KB <= kb <= MAX_KB:
                return data
            if kb < MIN_KB:
                prev = enc(small, min(q + 5, 95))
                if len(prev) / 1024 <= MAX_KB:
                    return prev
                break

    return enc(pil_img.resize(
        (max(int(ow * 0.18), 20), max(int(oh * 0.18), 26)), Image.LANCZOS
    ), 60)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def process_passport_photo(file_bytes: bytes) -> dict:
    # 1. Load + fix orientation
    pil = _load_pil(file_bytes)

    # 2. Detect face (on original before bg removal — more accurate)
    bgr       = _pil_to_bgr(pil)
    face      = _detect_face(bgr)
    face_found = face is not None

    # 3. Remove background → white (pass face coords for better seeding)
    pil = _remove_background(pil, face=face)

    # 4. Crop around face
    if face_found:
        pil = _crop_portrait(pil, face)

    # 5. Resize to passport dimensions
    pil = pil.resize((PASSPORT_W, PASSPORT_H), Image.LANCZOS)

    # 6. Final white canvas (removes any remaining colour artifacts)
    canvas = Image.new("RGB", (PASSPORT_W, PASSPORT_H), (255, 255, 255))
    canvas.paste(pil)

    # 7. Compress to 10–18 KB
    final_bytes = _compress_to_range(canvas)

    return {
        "ok":         True,
        "image":      final_bytes,
        "face_found": face_found,
        "size_kb":    round(len(final_bytes) / 1024, 1),
    }
