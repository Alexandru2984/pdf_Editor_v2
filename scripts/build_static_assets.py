#!/usr/bin/env python3
"""Build deterministic, bandwidth-conscious brand assets.

The original 1024 px logo remains the editable source. Templates consume only
the generated, purpose-sized derivatives. Run this script from any directory;
output paths are resolved from the repo.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "static"


def build_brand_images() -> None:
    source = STATIC / "images" / "logo.jpg"
    with Image.open(source) as original:
        image = original.convert("RGB")
        for size in (64, 192, 512):
            resized = image.resize((size, size), Image.Resampling.LANCZOS)
            resized.save(
                STATIC / "images" / f"logo-{size}.webp",
                "WEBP",
                quality=82,
                method=6,
            )

        apple_icon = image.resize((180, 180), Image.Resampling.LANCZOS)
        apple_icon.quantize(
            colors=256,
            method=Image.Quantize.MEDIANCUT,
            dither=Image.Dither.FLOYDSTEINBERG,
        ).save(STATIC / "images" / "apple-touch-icon.png", "PNG", optimize=True)


def main() -> None:
    build_brand_images()


if __name__ == "__main__":
    main()
