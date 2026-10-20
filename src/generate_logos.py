"""
Generator Logo Dummy — Trademark Logo Similarity Search
=======================================================

Membuat dataset logo sintetis untuk melatih dan mengevaluasi model kemiripan logo.

Konsep:
- Setiap "logo induk" (parent) dibuat dari kombinasi acak: bentuk, warna, ikon,
  tata letak, dan nama merek.
- Setiap logo induk punya beberapa "varian mirip" yang dibuat dengan mengubah
  sedikit logo induk (warna, typo nama, bentuk mirip, rotasi, noise scan, dll).
- Semua gambar dengan parent_id yang sama = MIRIP (positive pair).
  Gambar dengan parent_id berbeda = TIDAK MIRIP (negative pair).
- Split train/val/test dilakukan per parent_id agar tidak ada kebocoran data.

Output:
  <out>/images/p0001_v00.png    -> v00 = logo induk, v01..vNN = varian
  <out>/metadata.csv            -> label & atribut tiap gambar
  <out>/preview.png             -> grid contoh (kolom pertama = logo induk)

Pemakaian:
  pip install pillow numpy
  python generate_logos.py --n-parents 1000 --n-variants 8 --out data/dummy_logos

Membutuhkan Pillow >= 10.1.
"""

import argparse
import colorsys
import copy
import csv
import io
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

# ---------------------------------------------------------------------------
# Kosakata nama merek
# ---------------------------------------------------------------------------
SYLLABLES = ["ka", "ri", "nu", "sa", "ta", "ra", "ma", "ja", "ya", "wi", "lo",
             "ba", "di", "su", "go", "pe", "ku", "la", "na", "to", "me", "si",
             "an", "tu", "ga", "ni", "ro", "da", "ke", "bu"]
SUFFIXES = ["", "", " Tea", " Kopi", " Group", " Tech", " Food", " Mart",
            " Jaya", " Abadi", " Lab", " Studio"]

FONT_CANDIDATES = [
    "DejaVuSans-Bold.ttf", "arialbd.ttf", "Arial Bold.ttf",
    "LiberationSans-Bold.ttf", "DejaVuSans.ttf", "arial.ttf", "Arial.ttf",
]

SHAPES = ["circle", "square", "rounded", "triangle", "diamond", "hexagon",
          "star", "shield", "ring"]
# Pasangan bentuk yang "mirip" untuk transformasi shape_similar
SIMILAR_SHAPE = {
    "circle": ["ring", "hexagon"], "ring": ["circle"],
    "square": ["rounded", "diamond"], "rounded": ["square", "shield"],
    "triangle": ["diamond", "shield"], "diamond": ["square", "triangle"],
    "hexagon": ["circle", "rounded"], "star": ["hexagon", "circle"],
    "shield": ["rounded", "triangle"],
}
ICONS = ["none", "dot", "inner_ring", "stripes", "inner_star", "chevron"]
LAYOUTS = ["emblem", "badge", "inline", "wordmark"]

SPEC_TRANSFORMS = ["color_shift", "text_typo", "suffix_change",
                   "shape_similar", "layout_shift"]
POST_TRANSFORMS = ["rotate", "flip", "scan_noise", "crop_zoom", "grayscale"]

_FONT_CACHE = {}


# ---------------------------------------------------------------------------
# Utilitas
# ---------------------------------------------------------------------------
def load_font(size):
    size = max(int(size), 6)
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    font = None
    for name in FONT_CANDIDATES:
        try:
            font = ImageFont.truetype(name, size)
            break
        except OSError:
            continue
    if font is None:
        font = ImageFont.load_default(size=size)  # Pillow >= 10.1
    _FONT_CACHE[size] = font
    return font


def rand_color(rng, v_range=(0.35, 0.9)):
    h = rng.random()
    s = rng.uniform(0.5, 1.0)
    v = rng.uniform(*v_range)
    return tuple(int(c * 255) for c in colorsys.hsv_to_rgb(h, s, v))


def hue_shift(rgb, delta):
    r, g, b = [c / 255 for c in rgb]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    return tuple(int(c * 255) for c in colorsys.hsv_to_rgb((h + delta) % 1, s, v))


def regular_polygon(cx, cy, r, n, rot_deg=-90):
    return [(cx + r * math.cos(math.radians(rot_deg + i * 360 / n)),
             cy + r * math.sin(math.radians(rot_deg + i * 360 / n)))
            for i in range(n)]


def star_points(cx, cy, r_out, r_in, n=5):
    pts = []
    for i in range(n * 2):
        r = r_out if i % 2 == 0 else r_in
        a = math.radians(-90 + i * 180 / n)
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


def make_brand(rng):
    word = "".join(rng.choice(SYLLABLES) for _ in range(rng.randint(2, 4)))
    return word.capitalize(), rng.choice(SUFFIXES)


def make_initials(word, suffix):
    parts = [word] + ([suffix.strip()] if suffix.strip() else [])
    if len(parts) == 1:
        return word[:2].upper()
    return "".join(p[0] for p in parts).upper()


# ---------------------------------------------------------------------------
# Menggambar
# ---------------------------------------------------------------------------
def draw_shape(d, shape, cx, cy, r, fill, bg):
    if shape == "circle":
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill)
    elif shape == "ring":
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=bg,
                  outline=fill, width=max(int(r * 0.18), 2))
    elif shape == "square":
        d.rectangle([cx - r * 0.88, cy - r * 0.88, cx + r * 0.88, cy + r * 0.88], fill=fill)
    elif shape == "rounded":
        d.rounded_rectangle([cx - r * 0.9, cy - r * 0.9, cx + r * 0.9, cy + r * 0.9],
                            radius=r * 0.3, fill=fill)
    elif shape == "triangle":
        d.polygon(regular_polygon(cx, cy + r * 0.15, r * 1.1, 3), fill=fill)
    elif shape == "diamond":
        d.polygon(regular_polygon(cx, cy, r, 4, rot_deg=0), fill=fill)
    elif shape == "hexagon":
        d.polygon(regular_polygon(cx, cy, r, 6, rot_deg=0), fill=fill)
    elif shape == "star":
        d.polygon(star_points(cx, cy, r * 1.05, r * 0.5), fill=fill)
    elif shape == "shield":
        pts = [(cx - r * 0.85, cy - r * 0.85), (cx + r * 0.85, cy - r * 0.85),
               (cx + r * 0.85, cy + r * 0.1), (cx, cy + r),
               (cx - r * 0.85, cy + r * 0.1)]
        d.polygon(pts, fill=fill)


def draw_icon(d, icon, cx, cy, r, color):
    if icon == "dot":
        d.ellipse([cx - r * 0.4, cy - r * 0.4, cx + r * 0.4, cy + r * 0.4], fill=color)
    elif icon == "inner_ring":
        d.ellipse([cx - r * 0.7, cy - r * 0.7, cx + r * 0.7, cy + r * 0.7],
                  outline=color, width=max(int(r * 0.14), 2))
    elif icon == "stripes":
        h = r * 0.16
        for k in (-1, 0, 1):
            y = cy + k * r * 0.38
            d.rectangle([cx - r * 0.6, y - h / 2, cx + r * 0.6, y + h / 2], fill=color)
    elif icon == "inner_star":
        d.polygon(star_points(cx, cy, r * 0.75, r * 0.32), fill=color)
    elif icon == "chevron":
        w = r * 0.22
        d.line([(cx - r * 0.6, cy - r * 0.3), (cx, cy + r * 0.35),
                (cx + r * 0.6, cy - r * 0.3)], fill=color, width=int(w), joint="curve")


def draw_text_fit(d, text, center, max_w, max_h, fill):
    size = int(max_h)
    while size > 6:
        font = load_font(size)
        x0, y0, x1, y1 = d.textbbox((0, 0), text, font=font)
        if (x1 - x0) <= max_w and (y1 - y0) <= max_h:
            break
        size = int(size * 0.92)
    d.text(center, text, font=load_font(size), fill=fill, anchor="mm")


def render(spec, size):
    S = size * 2  # render 2x lalu downscale agar tepi halus
    img = Image.new("RGB", (S, S), spec["bg"])
    d = ImageDraw.Draw(img)
    sc = spec["scale"]
    ox, oy = spec["offset"]
    shape, icon, layout = spec["shape"], spec["icon"], spec["layout"]
    prim, sec, bg = spec["primary"], spec["secondary"], spec["bg"]
    inner = prim if shape == "ring" else sec  # warna isi di dalam bentuk
    text = spec["word"] + spec["suffix"]

    if layout == "emblem":
        cx, cy, r = S * (0.5 + ox), S * (0.5 + oy), S * 0.36 * sc
        draw_shape(d, shape, cx, cy, r, prim, bg)
        draw_text_fit(d, spec["initials"], (cx, cy), r * 1.0, r * 0.65, inner)
    elif layout == "badge":
        cx, cy, r = S * (0.5 + ox), S * (0.38 + oy), S * 0.26 * sc
        draw_shape(d, shape, cx, cy, r, prim, bg)
        draw_icon(d, icon, cx, cy, r * 0.6, inner)
        draw_text_fit(d, text, (S * 0.5, S * 0.83), S * 0.9, S * 0.13, spec["text_color"])
    elif layout == "inline":
        cx, cy, r = S * (0.19 + ox), S * (0.5 + oy), S * 0.14 * sc
        draw_shape(d, shape, cx, cy, r, prim, bg)
        draw_icon(d, icon, cx, cy, r * 0.6, inner)
        draw_text_fit(d, text, (S * 0.64, S * 0.5), S * 0.6, S * 0.16, spec["text_color"])
    else:  # wordmark
        draw_text_fit(d, text, (S * (0.5 + ox), S * (0.45 + oy)), S * 0.88 * sc,
                      S * 0.22, prim)
        d.rectangle([S * 0.15, S * 0.62, S * 0.85, S * 0.655], fill=spec["accent"])
        draw_shape(d, shape, S * 0.5, S * 0.76, S * 0.045, prim, bg)

    return img.resize((size, size), Image.LANCZOS)


# ---------------------------------------------------------------------------
# Logo induk
# ---------------------------------------------------------------------------
def make_parent_spec(rng):
    word, suffix = make_brand(rng)
    primary = rand_color(rng)
    roll = rng.random()
    if roll < 0.75:
        bg = (255, 255, 255)
    else:
        bg = tuple(int(c * 255) for c in colorsys.hsv_to_rgb(rng.random(), rng.uniform(0.04, 0.12), 0.97))
    secondary = (255, 255, 255) if rng.random() < 0.5 else rand_color(rng, (0.75, 1.0))
    return {
        "word": word, "suffix": suffix, "initials": make_initials(word, suffix),
        "shape": rng.choice(SHAPES), "icon": rng.choice(ICONS),
        "layout": rng.choice(LAYOUTS),
        "primary": primary, "secondary": secondary, "bg": bg,
        "text_color": primary if rng.random() < 0.5 else (40, 40, 40),
        "accent": rand_color(rng) if rng.random() < 0.5 else primary,
        "scale": 1.0, "offset": (0.0, 0.0),
    }


# ---------------------------------------------------------------------------
# Transformasi varian
# ---------------------------------------------------------------------------
SIMILAR_LETTERS = {"a": "e", "e": "a", "i": "l", "l": "i", "o": "u", "u": "o",
                   "m": "n", "n": "m", "r": "n", "k": "c", "c": "k", "s": "z",
                   "b": "d", "d": "b", "g": "q", "t": "f"}


def typo(word, rng):
    for _ in range(10):
        chars = list(word)
        i = rng.randrange(1, len(chars)) if len(chars) > 1 else 0
        op = rng.choice(["replace", "delete", "double", "swap"])
        if op == "replace":
            c = chars[i].lower()
            new = SIMILAR_LETTERS.get(c, rng.choice("aeiou"))
            chars[i] = new.upper() if chars[i].isupper() else new
        elif op == "delete" and len(chars) > 3:
            del chars[i]
        elif op == "double":
            chars.insert(i, chars[i].lower())
        elif op == "swap" and i < len(chars) - 1:
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
        new_word = "".join(chars)
        if new_word.lower() != word.lower():
            return new_word[0].upper() + new_word[1:]
    return word + "o"


def apply_spec_transform(spec, name, rng):
    if name == "color_shift":
        delta = rng.uniform(0.06, 0.18) * rng.choice([-1, 1])
        spec["primary"] = hue_shift(spec["primary"], delta)
        if spec["secondary"] != (255, 255, 255):
            spec["secondary"] = hue_shift(spec["secondary"], delta)
        spec["accent"] = hue_shift(spec["accent"], delta)
        if spec["text_color"] != (40, 40, 40):
            spec["text_color"] = spec["primary"]
    elif name == "text_typo":
        if spec["layout"] == "emblem":
            ini = list(spec["initials"])
            j = rng.randrange(len(ini))
            ini[j] = SIMILAR_LETTERS.get(ini[j].lower(), "o").upper()
            spec["initials"] = "".join(ini)
        else:
            spec["word"] = typo(spec["word"], rng)
    elif name == "suffix_change":
        options = [s for s in SUFFIXES if s != spec["suffix"]]
        spec["suffix"] = rng.choice(options)
    elif name == "shape_similar":
        spec["shape"] = rng.choice(SIMILAR_SHAPE[spec["shape"]])
    elif name == "layout_shift":
        spec["scale"] = rng.uniform(0.85, 1.12)
        spec["offset"] = (rng.uniform(-0.04, 0.04), rng.uniform(-0.04, 0.04))
    return spec


def apply_post_transform(img, name, rng, nrng, bg):
    if name == "rotate":
        angle = rng.uniform(6, 20) * rng.choice([-1, 1])
        img = img.rotate(angle, resample=Image.BICUBIC, fillcolor=bg)
    elif name == "flip":
        img = ImageOps.mirror(img)
    elif name == "scan_noise":
        arr = np.asarray(img).astype(np.float32)
        arr += nrng.normal(0, rng.uniform(6, 16), arr.shape)
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.4, 1.1)))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=rng.randint(25, 60))
        img = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    elif name == "crop_zoom":
        w, h = img.size
        frac = rng.uniform(0.8, 0.92)
        cw, ch = int(w * frac), int(h * frac)
        x0, y0 = rng.randint(0, w - cw), rng.randint(0, h - ch)
        img = img.crop((x0, y0, x0 + cw, y0 + ch)).resize((w, h), Image.LANCZOS)
    elif name == "grayscale":
        img = ImageOps.grayscale(img).convert("RGB")
    return img


def make_variant(parent_spec, size, rng, nrng):
    options = SPEC_TRANSFORMS + POST_TRANSFORMS
    if parent_spec["layout"] == "emblem":
        options = [o for o in options if o != "suffix_change"]  # suffix tak terlihat
    k = rng.choice([1, 1, 2, 2, 3])
    chosen = rng.sample(options, k)

    spec = copy.deepcopy(parent_spec)
    for t in chosen:
        if t in SPEC_TRANSFORMS:
            spec = apply_spec_transform(spec, t, rng)
    img = render(spec, size)
    for t in chosen:
        if t in POST_TRANSFORMS:
            img = apply_post_transform(img, t, rng, nrng, spec["bg"])
    return img, spec, chosen


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def assign_splits(n_parents, rng, ratios=(0.7, 0.15, 0.15)):
    ids = list(range(1, n_parents + 1))
    rng.shuffle(ids)
    n_train = int(n_parents * ratios[0])
    n_val = int(n_parents * ratios[1])
    split = {}
    for i, pid in enumerate(ids):
        split[pid] = "train" if i < n_train else ("val" if i < n_train + n_val else "test")
    return split


def save_preview(out_dir, rows, size, n_rows=8):
    rows = rows[:n_rows]
    if not rows:
        return
    n_cols = max(len(r) for r in rows)
    cell, pad = size, 6
    grid = Image.new("RGB", (n_cols * (cell + pad) + pad, len(rows) * (cell + pad) + pad),
                     (200, 200, 200))
    for ri, imgs in enumerate(rows):
        for ci, im in enumerate(imgs):
            grid.paste(im, (pad + ci * (cell + pad), pad + ri * (cell + pad)))
    grid.save(out_dir / "preview.png")


def main():
    ap = argparse.ArgumentParser(description="Generator logo dummy untuk similarity search")
    ap.add_argument("--n-parents", type=int, default=1000, help="jumlah logo induk")
    ap.add_argument("--n-variants", type=int, default=8, help="jumlah varian per logo induk")
    ap.add_argument("--size", type=int, default=224, help="ukuran gambar (piksel)")
    ap.add_argument("--out", type=str, default="data/dummy_logos", help="folder output")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    nrng = np.random.default_rng(args.seed)
    out_dir = Path(args.out)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    splits = assign_splits(args.n_parents, rng)
    records, preview_rows = [], []

    for pid in range(1, args.n_parents + 1):
        spec = make_parent_spec(rng)
        parent_img = render(spec, args.size)
        fname = f"p{pid:04d}_v00.png"
        parent_img.save(img_dir / fname)
        records.append({
            "filename": fname, "parent_id": pid, "is_parent": 1, "variant_types": "",
            "brand_text": spec["word"] + spec["suffix"], "initials": spec["initials"],
            "shape": spec["shape"], "layout": spec["layout"], "split": splits[pid],
        })
        row = [parent_img]

        for v in range(1, args.n_variants + 1):
            img, vspec, chosen = make_variant(spec, args.size, rng, nrng)
            vname = f"p{pid:04d}_v{v:02d}.png"
            img.save(img_dir / vname)
            records.append({
                "filename": vname, "parent_id": pid, "is_parent": 0,
                "variant_types": "+".join(chosen),
                "brand_text": vspec["word"] + vspec["suffix"], "initials": vspec["initials"],
                "shape": vspec["shape"], "layout": vspec["layout"], "split": splits[pid],
            })
            if len(row) < 6:
                row.append(img)
        preview_rows.append(row)

        if pid % 100 == 0 or pid == args.n_parents:
            print(f"  {pid}/{args.n_parents} logo induk selesai")

    with open(out_dir / "metadata.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    save_preview(out_dir, preview_rows, args.size)
    print(f"\nSelesai: {len(records)} gambar di {img_dir}")
    print(f"Metadata: {out_dir / 'metadata.csv'}")
    print(f"Preview : {out_dir / 'preview.png'}")


if __name__ == "__main__":
    main()
