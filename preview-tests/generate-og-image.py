#!/usr/bin/env python3
"""Generate OG preview image + favicons for IDHS 2026.

Aesthetic: CMYK faux halftone (classic screen angles) + 2-bit ordered
(Bayer) dithering. Type: Kelmscott Mono (MIT, IbrahimAbdelhay1/kelmscott-mono).
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# Kelmscott Mono (MIT): https://github.com/IbrahimAbdelhay1/kelmscott-mono
# Clone it next to this script (or set KELMSCOTT_MONO) before running.
import os
FONT = os.environ.get("KELMSCOTT_MONO",
                      os.path.join(os.path.dirname(__file__),
                                   "kelmscott-mono", "KelmscottMono.otf"))
OUT = os.path.join(os.path.dirname(__file__), "..", "assets")

# ---------------------------------------------------------------- base art
def base_art(w, h):
    img = Image.new("RGB", (w, h), (246, 240, 226))  # warm paper
    d = ImageDraw.Draw(img, "RGBA")

    # big soft CMY discs, overlapping like misregistered ink
    r = int(w * 0.42)
    d.ellipse([int(-r*0.25), int(h*0.55), int(-r*0.25)+2*r, int(h*0.55)+2*r],
              fill=(0, 174, 239, 175))            # cyan
    d.ellipse([int(w*0.62), int(-r*0.45), int(w*0.62)+2*r, int(-r*0.45)+2*r],
              fill=(236, 0, 140, 165))            # magenta
    d.ellipse([int(w*0.70), int(h*0.60), int(w*0.70)+2*r, int(h*0.60)+2*r],
              fill=(255, 222, 23, 200))           # yellow
    d.ellipse([int(-r*0.35), int(-r*0.55), int(-r*0.35)+2*r, int(-r*0.55)+2*r],
              fill=(255, 222, 23, 140))           # yellow, top-left

    img = img.filter(ImageFilter.GaussianBlur(w * 0.02))
    d = ImageDraw.Draw(img, "RGBA")

    # rule frame, Kelmscott-page style
    m = int(w * 0.045)
    lw = max(2, w // 300)
    d.rectangle([m, m, w - m, h - m], outline=(20, 16, 14, 255), width=lw)
    m2 = m + lw * 3
    d.rectangle([m2, m2, w - m2, h - m2], outline=(20, 16, 14, 255),
                width=max(1, lw // 2))

    # cream printed plate behind the type, so it reads through the screen
    px0, py0, px1, py1 = int(w * 0.10), int(h * 0.115), int(w * 0.90), int(h * 0.72)
    d.rectangle([px0, py0, px1, py1], fill=(246, 240, 226, 245))
    d.rectangle([px0, py0, px1, py1], outline=(20, 16, 14, 255), width=lw)
    p2 = lw * 3
    d.rectangle([px0 + p2, py0 + p2, px1 - p2, py1 - p2],
                outline=(20, 16, 14, 255), width=max(1, lw // 2))

    # ---- type
    f_big = ImageFont.truetype(FONT, int(w * 0.150))
    f_mid = ImageFont.truetype(FONT, int(w * 0.062))
    f_sm = ImageFont.truetype(FONT, int(w * 0.055))

    cx = w // 2
    def center(txt, font, y, fill=(20, 16, 14)):
        bb = d.textbbox((0, 0), txt, font=font)
        tw = bb[2] - bb[0]
        # offset shadow pass in magenta for misregistered-ink feel
        off = max(2, w // 400)
        d.text((cx - tw / 2 - bb[0] + off, y + off), txt, font=font,
               fill=(236, 0, 140, 210))
        d.text((cx - tw / 2 - bb[0], y), txt, font=font, fill=fill)
        return bb[3] - bb[1]

    y = int(h * 0.155)
    hh = center("SVA IxD", f_big, y)
    y += int(hh * 1.38)
    hh = center("Interaction Design", f_mid, y)
    y += int(hh * 1.35)
    hh = center("History", f_mid, y)
    y += int(hh * 1.50)
    center("Spring 2026", f_sm, y)

    # ornament dots along bottom rule
    yy = int(h * 0.88)
    for i in range(9):
        xx = int(w * 0.30 + i * w * 0.05)
        rr = w // 160
        d.ellipse([xx - rr, yy - rr, xx + rr, yy + rr], fill=(20, 16, 14))
    return img


# ------------------------------------------------------- CMYK halftone
def rgb_to_cmyk(arr):
    rgb = arr.astype(np.float64) / 255.0
    k = 1.0 - rgb.max(axis=2)
    denom = np.clip(1.0 - k, 1e-6, None)
    c = (1.0 - rgb[..., 0] - k) / denom
    m = (1.0 - rgb[..., 1] - k) / denom
    y = (1.0 - rgb[..., 2] - k) / denom
    return c, m, y, k


def halftone_channel(chan, angle_deg, cell, blur=0.75, max_r=0.72):
    """Binary halftone screen: dot area proportional to ink coverage."""
    h, w = chan.shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    a = np.deg2rad(angle_deg)
    u = xs * np.cos(a) + ys * np.sin(a)
    v = -xs * np.sin(a) + ys * np.cos(a)
    cu, cv = np.floor(u / cell) + 0.5, np.floor(v / cell) + 0.5
    du, dv = u - cu * cell, v - cv * cell
    dist2 = du * du + dv * dv
    # smooth the channel so dot size follows local tone
    sm = np.asarray(Image.fromarray((chan * 255).astype(np.uint8))
                    .filter(ImageFilter.GaussianBlur(cell * blur)),
                    dtype=np.float64) / 255.0
    max_r2 = (cell * max_r) ** 2
    return dist2 <= sm * max_r2


def cmyk_halftone(img, cell):
    arr = np.asarray(img)
    c, m, y, k = rgb_to_cmyk(arr)
    angles = {"c": 15.0, "m": 75.0, "y": 0.0, "k": 45.0}
    C = halftone_channel(c, angles["c"], cell)
    M = halftone_channel(m, angles["m"], cell)
    Y = halftone_channel(y, angles["y"], cell)
    # tighter screen for K so blackletter strokes stay dense and legible
    K = halftone_channel(k, angles["k"], cell, blur=0.35, max_r=0.85)
    # multiplicative ink layering on paper white
    op = 0.92  # slight ink transparency keeps overlaps readable
    r = (1 - C * op) * (1 - K * op)
    g = (1 - M * op) * (1 - K * op)
    b = (1 - Y * op) * (1 - K * op)
    out = np.stack([r, g, b], axis=2)
    # warm paper tint
    out *= np.array([0.985, 0.965, 0.915])
    return Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8))


# ----------------------------------------------------- 2-bit dithering
BAYER8 = (1 / 64.0) * np.array(
    [[0, 32, 8, 40, 2, 34, 10, 42],
     [48, 16, 56, 24, 50, 18, 58, 26],
     [12, 44, 4, 36, 14, 46, 6, 38],
     [60, 28, 52, 20, 62, 30, 54, 22],
     [3, 35, 11, 43, 1, 33, 9, 41],
     [51, 19, 59, 27, 49, 17, 57, 25],
     [15, 47, 7, 39, 13, 45, 5, 37],
     [63, 31, 55, 23, 61, 29, 53, 21]]) - 0.5


def dither_2bit(img):
    """Ordered Bayer dither, 4 levels (2 bits) per RGB channel."""
    a = np.asarray(img).astype(np.float64) / 255.0
    h, w = a.shape[:2]
    t = np.tile(BAYER8, (h // 8 + 1, w // 8 + 1))[:h, :w][..., None]
    levels = 3  # 4 levels -> 2 bits
    q = np.floor(a * levels + 0.5 + t)
    return Image.fromarray((np.clip(q / levels, 0, 1) * 255).astype(np.uint8))


def process(img, cell, final_size):
    ht = cmyk_halftone(img, cell)
    ht = ht.resize(final_size, Image.LANCZOS)
    return dither_2bit(ht)


# ---------------------------------------------------------------- build
if __name__ == "__main__":
    import os
    os.makedirs(OUT, exist_ok=True)

    # OG image: 4:3, rendered 2x then downsampled
    W, H = 1200, 900
    art = base_art(W * 2, H * 2)
    og = process(art, cell=13, final_size=(W, H))
    og.save(f"{OUT}/og-image.png", optimize=True)
    print("og-image.png", og.size)

    # Favicon art: square monogram, chunkier screen
    S = 512
    fav = Image.new("RGB", (S, S), (246, 240, 226))
    d = ImageDraw.Draw(fav, "RGBA")
    d.ellipse([-S * 0.25, S * 0.35, S * 0.75, S * 1.35], fill=(0, 174, 239, 190))
    d.ellipse([S * 0.35, -S * 0.30, S * 1.35, S * 0.70], fill=(236, 0, 140, 180))
    fav = fav.filter(ImageFilter.GaussianBlur(S * 0.03))
    d = ImageDraw.Draw(fav)
    lw = S // 26
    d.rectangle([lw, lw, S - lw, S - lw], outline=(20, 16, 14), width=lw)
    f = ImageFont.truetype(FONT, int(S * 0.52))
    bb = d.textbbox((0, 0), "Ix", font=f)
    off = S // 90
    d.text((S / 2 - (bb[2] - bb[0]) / 2 - bb[0] + off,
            S / 2 - (bb[3] - bb[1]) / 2 - bb[1] + off), "Ix",
           font=f, fill=(236, 0, 140))
    d.text((S / 2 - (bb[2] - bb[0]) / 2 - bb[0],
            S / 2 - (bb[3] - bb[1]) / 2 - bb[1]), "Ix",
           font=f, fill=(20, 16, 14))

    fav_ht = process(fav, cell=10, final_size=(S, S))
    fav_ht.save(f"{OUT}/favicon-512.png", optimize=True)
    for size in (180, 32, 16):
        fav_ht.resize((size, size), Image.LANCZOS).save(
            f"{OUT}/favicon-{size}.png", optimize=True)
    os.rename(f"{OUT}/favicon-180.png", f"{OUT}/apple-touch-icon.png")
    fav_ht.resize((48, 48), Image.LANCZOS).save(
        os.path.join(os.path.dirname(__file__), "..", "favicon.ico"),
        sizes=[(16, 16), (32, 32), (48, 48)])
    print("favicons done")
