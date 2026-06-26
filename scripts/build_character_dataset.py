#!/usr/bin/env python3
"""
build_character_dataset.py — turnkey crawler + Claude-vision auto-labeler for a
single-character anime dataset (Gate-1 data for the learned-deformation-manifold PoC).

Pipeline:  Danbooru (rating:general) → download → Claude-vision label (viewpoint /
expression / exaggeration / crop / quality) → filtered images/ + metadata.jsonl.

No GPU needed (crawl + API only) — runs locally. Resumable (skips already-labeled md5s).

USAGE
  python scripts/build_character_dataset.py --character "hatsune_miku" --count 800 \
      --out data/char_miku --model claude-haiku-4-5-20251001 --concurrency 6

  # then for the real PoC character (find its Danbooru tag first), e.g.:
  python scripts/build_character_dataset.py --character "<danbooru_character_tag>" --count 2000 \
      --out data/char_poc

NOTES / CAVEATS (read once)
  * Source images are fan-art/screenshots that are often copyrighted; this is for PERSONAL
    RESEARCH. Respect Danbooru's ToS + per-image licensing before any redistribution/training-at-scale.
  * rating:general only + a Claude NSFW double-check; still spot-check the output.
  * Be polite: the script rate-limits API pages and downloads. Don't crank concurrency high.
  * ANTHROPIC_API_KEY is read from the environment or the repo .env (never printed).
"""
import argparse, base64, hashlib, io, json, os, re, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from PIL import Image

DANBOORU = "https://danbooru.donmai.us/posts.json"
UA = "LenaLab-research-dataset-builder/1.0 (personal research; contact via repo)"
IMG_EXTS = {"jpg", "jpeg", "png", "webp"}

VIEWPOINTS = ["front", "three_quarter_left", "three_quarter_right",
              "profile_left", "profile_right", "from_above", "from_below", "back", "other"]

LABEL_PROMPT = """You are annotating a single frame for an anime-character dataset that will train a
deformation/expression/viewpoint model. Look at the image and return STRICT JSON only (no prose, no code
fences) with EXACTLY these keys:
{
  "character_present": bool,            // is there a clearly drawn anime character/face?
  "num_characters": int,               // how many distinct characters
  "single_character_clear": bool,      // exactly one character, clearly visible, not tiny
  "viewpoint": one of %s,              // camera/drawing angle of the head/face
  "expression": str,                   // short label e.g. "neutral","smile","surprised","angry","crying","laughing","smug"
  "exaggeration": float,               // 0.0 realistic .. 1.0 extremely exaggerated/stylized (squash-stretch, huge eyes, etc.)
  "mouth_open": float,                 // 0.0 closed .. 1.0 wide open
  "eyes": str,                         // "open","half","closed","wide","wink"
  "face_bbox": [x0,y0,x1,y1] or null,  // normalized 0..1 box around the FACE (null if unclear)
  "quality_ok": bool,                  // clear, in-focus, face not heavily occluded/cropped, decent resolution
  "nsfw": bool,                        // any nudity/suggestive content
  "notes": str                         // <=12 words, optional
}
Return ONLY the JSON object.""" % VIEWPOINTS


def load_api_key() -> str:
    k = os.environ.get("ANTHROPIC_API_KEY")
    if k:
        return k.strip()
    env = Path(__file__).resolve().parents[1] / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line.startswith("ANTHROPIC_API_KEY"):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    sys.exit("ANTHROPIC_API_KEY not found in env or .env")


def crawl_danbooru(character: str, target: int, rating: str, min_dim: int, log):
    """Yield post dicts (paginated), filtered to usable images."""
    posts, page, seen = [], 1, set()
    sess = requests.Session(); sess.headers["User-Agent"] = UA
    while len(posts) < target and page <= 1000:
        params = {"tags": f"{character} rating:{rating}", "limit": 200, "page": page}
        try:
            r = sess.get(DANBOORU, params=params, timeout=30)
            if r.status_code == 429:
                log(f"  rate-limited on page {page}; sleeping 10s"); time.sleep(10); continue
            r.raise_for_status()
            batch = r.json()
        except Exception as e:
            log(f"  crawl error page {page}: {e}"); break
        if not batch:
            break
        for p in batch:
            md5 = p.get("md5"); url = p.get("large_file_url") or p.get("file_url")
            ext = (p.get("file_ext") or "").lower()
            if not (md5 and url) or md5 in seen:
                continue
            if ext not in IMG_EXTS:
                continue
            if min(p.get("image_width", 0), p.get("image_height", 0)) < min_dim:
                continue
            seen.add(md5)
            posts.append({"id": p.get("id"), "md5": md5, "url": url, "ext": ext,
                          "tags": p.get("tag_string", ""),
                          "char_tags": p.get("tag_string_character", ""),
                          "w": p.get("image_width"), "h": p.get("image_height")})
            if len(posts) >= target:
                break
        log(f"  page {page}: collected {len(posts)}/{target}")
        page += 1
        time.sleep(1.0)  # politeness
    return posts


def download(post, img_dir: Path, sess) -> Path | None:
    dst = img_dir / f"{post['md5']}.{post['ext']}"
    if dst.exists():
        return dst
    try:
        r = sess.get(post["url"], timeout=60); r.raise_for_status()
        dst.write_bytes(r.content)
        return dst
    except Exception:
        return None


def to_b64_jpeg(path: Path, max_side=512) -> tuple[str, str]:
    im = Image.open(path).convert("RGB")
    im.thumbnail((max_side, max_side))
    buf = io.BytesIO(); im.save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode(), "image/jpeg"


def label_image(client, model: str, path: Path, retries=4):
    b64, media = to_b64_jpeg(path)
    for attempt in range(retries):
        try:
            msg = client.messages.create(
                model=model, max_tokens=400,
                messages=[{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": media, "data": b64}},
                    {"type": "text", "text": LABEL_PROMPT}]}])
            txt = msg.content[0].text.strip()
            txt = re.sub(r"^```(?:json)?|```$", "", txt.strip(), flags=re.MULTILINE).strip()
            m = re.search(r"\{.*\}", txt, re.DOTALL)
            return json.loads(m.group(0) if m else txt)
        except Exception as e:
            if attempt == retries - 1:
                return {"_error": str(e)}
            time.sleep(2 ** attempt)


def main():
    ap = argparse.ArgumentParser(description="Crawl + Claude-label a single-character anime dataset.")
    ap.add_argument("--character", required=True, help="Danbooru character tag, e.g. hatsune_miku")
    ap.add_argument("--count", type=int, default=800, help="target raw images to crawl")
    ap.add_argument("--out", required=True, help="output dataset dir")
    ap.add_argument("--model", default="claude-haiku-4-5-20251001", help="Claude vision model for labeling")
    ap.add_argument("--rating", default="general", choices=["general", "sensitive"], help="Danbooru rating tier")
    ap.add_argument("--min-dim", type=int, default=400, help="min image short side (px)")
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--max-label", type=int, default=0, help="cap images to label (0 = all downloaded)")
    ap.add_argument("--allow-multi", action="store_true", help="keep multi-character images too")
    args = ap.parse_args()

    import anthropic
    client = anthropic.Anthropic(api_key=load_api_key())

    out = Path(args.out); img_dir = out / "images"; img_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out / "metadata.jsonl"
    lock = threading.Lock()
    log = lambda m: print(m, flush=True)

    done = set()
    if meta_path.exists():
        for line in meta_path.read_text().splitlines():
            try: done.add(json.loads(line)["md5"])
            except Exception: pass
    log(f"[resume] {len(done)} already labeled")

    log(f"[crawl] character='{args.character}' target={args.count} rating={args.rating}")
    posts = crawl_danbooru(args.character, args.count, args.rating, args.min_dim, log)
    posts = [p for p in posts if p["md5"] not in done]
    log(f"[crawl] {len(posts)} new posts to fetch")

    sess = requests.Session(); sess.headers["User-Agent"] = UA
    if args.max_label > 0:
        posts = posts[:args.max_label]

    kept = skipped = errored = 0
    def work(post):
        nonlocal kept, skipped, errored
        path = download(post, img_dir, sess)
        if not path:
            return ("dl_fail", post)
        lab = label_image(client, args.model, path)
        if "_error" in lab:
            return ("label_err", post, lab)
        keep = (lab.get("character_present") and lab.get("quality_ok") and not lab.get("nsfw")
                and (args.allow_multi or lab.get("single_character_clear")))
        row = {"md5": post["md5"], "file": f"images/{path.name}", "danbooru_id": post["id"],
               "url": post["url"], "w": post["w"], "h": post["h"],
               "char_tags": post["char_tags"], "booru_tags": post["tags"],
               "label": lab, "kept": bool(keep)}
        with lock:
            with meta_path.open("a") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if keep:
            return ("kept", post)
        try: path.unlink()  # drop rejected image to save space
        except Exception: pass
        return ("skip", post)

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(work, p) for p in posts]
        for i, fut in enumerate(as_completed(futs), 1):
            res = fut.result(); tag = res[0]
            if tag == "kept": kept += 1
            elif tag == "skip": skipped += 1
            else: errored += 1
            if i % 25 == 0 or i == len(futs):
                log(f"  labeled {i}/{len(futs)} | kept={kept} skip={skipped} err={errored}")

    # summary
    by_view, by_expr = {}, {}
    if meta_path.exists():
        for line in meta_path.read_text().splitlines():
            try:
                r = json.loads(line)
                if not r.get("kept"): continue
                lv = r["label"]
                by_view[lv.get("viewpoint", "?")] = by_view.get(lv.get("viewpoint", "?"), 0) + 1
                by_expr[lv.get("expression", "?")] = by_expr.get(lv.get("expression", "?"), 0) + 1
            except Exception: pass
    total_kept = sum(by_view.values())
    log(f"\n[done] kept {total_kept} images → {out}")
    log(f"  viewpoints: {dict(sorted(by_view.items(), key=lambda x:-x[1]))}")
    log(f"  expressions: {dict(sorted(by_expr.items(), key=lambda x:-x[1])[:12])}")
    log(f"  metadata: {meta_path}")


if __name__ == "__main__":
    main()
