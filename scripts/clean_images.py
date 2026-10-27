import sys
from pathlib import Path
import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageOps

# Change this if your images are stored elsewhere.
IMAGE_DIR = Path("images")

SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp"
}

# Tune these settings if necessary.
OCR_CONFIDENCE = 30
MASK_DILATION = 5
INPAINT_RADIUS = 5


def clean_image(path):
    try:
        # Correct camera orientation before processing.
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            original = np.array(image)

        height, width = original.shape[:2]

        # OCR works on a resized copy for speed.
        scale = min(1.0, 1800 / max(width, height))
        if scale < 1.0:
            small = cv2.resize(
                original, None, fx=scale, fy=scale,
                interpolation=cv2.INTER_AREA
            )
        else:
            small = original

        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)

        # Detect text boxes.
        data = pytesseract.image_to_data(
            gray,
            config="--psm 11",
            output_type=pytesseract.Output.DICT
        )

        mask = np.zeros(small.shape[:2], dtype=np.uint8)
        found = 0

        for i, text in enumerate(data["text"]):
            text = text.strip()

            try:
                confidence = float(data["conf"][i])
            except (ValueError, TypeError):
                continue

            if not text or confidence < OCR_CONFIDENCE:
                continue

            x = data["left"][i]
            y = data["top"][i]
            w = data["width"][i]
            h = data["height"][i]

            # Ignore extremely small noise detections.
            if w < 3 or h < 3:
                continue

            # Expand the box slightly to cover text edges.
            pad_x = max(3, int(w * 0.12))
            pad_y = max(3, int(h * 0.25))

            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(mask.shape[1], x + w + pad_x)
            y2 = min(mask.shape[0], y + h + pad_y)

            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
            found += 1

        if found == 0:
            print(f"No text detected: {path}")
            return False

        # Enlarge the mask to cover outlines and shadows.
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (MASK_DILATION, MASK_DILATION)
        )
        mask = cv2.dilate(mask, kernel, iterations=2)

        # Restore the mask to original image dimensions.
        if scale < 1.0:
            mask = cv2.resize(
                mask, (width, height),
                interpolation=cv2.INTER_NEAREST
            )

        # Reconstruct the background around detected text.
        bgr = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
        cleaned = cv2.inpaint(
            bgr, mask, INPAINT_RADIUS, cv2.INPAINT_TELEA
        )
        cleaned = cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB)

        # Preserve the filename and image format.
        with Image.fromarray(cleaned) as output:
            suffix = path.suffix.lower()

            if suffix in {".jpg", ".jpeg"}:
                output.save(path, quality=94, optimize=True)
            elif suffix == ".png":
                output.save(path, optimize=True)
            elif suffix == ".webp":
                output.save(path, quality=94, method=6)

        print(f"Cleaned: {path} ({found} text regions)")
        return True

    except Exception as error:
        print(f"ERROR processing {path}: {error}")
        return False


def main():
    if not IMAGE_DIR.exists():
        raise SystemExit(f"Image directory not found: {IMAGE_DIR}")

    if len(sys.argv) > 1:
        files = [
            Path(arg) for arg in sys.argv[1:]
            if Path(arg).is_file()
            and Path(arg).suffix.lower() in SUPPORTED_EXTENSIONS
        ]
    else:
        files = [
            p for p in IMAGE_DIR.rglob("*")
            if p.is_file()
            and p.suffix.lower() in SUPPORTED_EXTENSIONS
        ]

    print(f"Found {len(files)} images")

    for path in files:
        clean_image(path)


if __name__ == "__main__":
    main()
