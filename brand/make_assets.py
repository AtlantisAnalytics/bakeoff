"""Regenerate the site's logo assets from the source logo.

    uv run --no-project --with pillow python brand/make_assets.py brand/bakeoff-logo.png site

Writes logo.webp (lockup, 2x for 440 css px), favicon-32.png and apple-touch-icon.png (the
cupcake mark alone) and og.jpg (1200x630 link preview). The source has an opaque white field,
so the page is light-only; a dark variant needs a transparent-background source logo.
"""

import sys
from pathlib import Path

from PIL import Image

src, out = Path(sys.argv[1]), Path(sys.argv[2])
im = Image.open(src).convert("RGB")
W, H = im.size
px = im.load()
bg = px[5, 5]


def ink(p, tol=18):
    return any(abs(a - b) > tol for a, b in zip(p, bg, strict=True))


rows = [any(ink(px[x, y]) for x in range(0, W, 2)) for y in range(H)]
cols = [any(ink(px[x, y]) for y in range(0, H, 2)) for x in range(W)]
top, bottom = rows.index(True), H - 1 - rows[::-1].index(True)
left, right = cols.index(True), W - 1 - cols[::-1].index(True)

# the widest blank band of rows separates the cupcake mark from the wordmark
gaps, y = [], top
while y < bottom:
    if not rows[y]:
        start = y
        while y < bottom and not rows[y]:
            y += 1
        gaps.append((start, y))
    y += 1
mark_bottom, _ = max(gaps, key=lambda g: g[1] - g[0])

lockup = im.crop((left, top, right + 1, bottom + 1))
lw = 880
lockup.resize((lw, round(lockup.height * lw / lockup.width)), Image.Resampling.LANCZOS).save(
    out / "logo.webp", "WEBP", quality=88, method=6)

mark = im.crop((left, top, right + 1, mark_bottom))
mcols = [any(ink(mark.getpixel((x, y))) for y in range(0, mark.height, 2)) for x in range(mark.width)]
mark = mark.crop((mcols.index(True), 0, mark.width - mcols[::-1].index(True), mark.height))
side = int(max(mark.size) * 1.16)
square = Image.new("RGB", (side, side), bg)
square.paste(mark, ((side - mark.width) // 2, (side - mark.height) // 2))
for name, size in (("favicon-32.png", 32), ("apple-touch-icon.png", 180)):
    square.resize((size, size), Image.Resampling.LANCZOS).save(out / name, optimize=True)

og = Image.new("RGB", (1200, 630), bg)
oh = 470
ol = lockup.resize((round(lockup.width * oh / lockup.height), oh), Image.Resampling.LANCZOS)
og.paste(ol, ((1200 - ol.width) // 2, (630 - oh) // 2))
og.save(out / "og.jpg", "JPEG", quality=86, optimize=True, progressive=True)  # JPEG keeps the gradients

for f in ("logo.webp", "favicon-32.png", "apple-touch-icon.png", "og.jpg"):
    print(f, (out / f).stat().st_size // 1024, "KB")
