"""
Interactive HTML gallery for reviewing and re-categorizing presentation images.
- View all images grouped by age bracket with thumbnails
- Click images to select, then move them to another bracket
- Export changes as a JSON manifest for applying moves
- Color-coded source detection (iPhone, Android, Facebook, etc.)

Usage:
    python gallery_report.py
    python gallery_report.py --thumbsize 200
    python gallery_report.py --no-open
    python gallery_report.py --apply changes.json   # apply saved moves
"""

import os
import sys
import argparse
import base64
import json
import shutil
import html
import webbrowser
from io import BytesIO
from collections import defaultdict
from PIL import Image

sys.stdout.reconfigure(line_buffering=True)

PRESENTATION = os.path.join(
    os.path.expanduser("~"),
    "OneDrive - Pen-Link Ltd", "Desktop",
    "\u05d0\u05d9\u05e9\u05d9",
    "\u05e8\u05d9\u05e3",
    "\u05d1\u05e8 \u05de\u05e6\u05d5\u05d5\u05d4",
    "the presentation",
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".heic", ".webp"}
OUTPUT_HTML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gallery_report.html")


def make_thumbnail_b64(filepath, size):
    try:
        img = Image.open(filepath)
        img.thumbnail((size, size))
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=60)
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return None


def guess_source(filename):
    name = filename.lower()
    if "fb_img" in name:
        return "facebook"
    if "_n.jpg" in name and any(c.isdigit() for c in name):
        return "facebook"
    if name.startswith("img_"):
        return "iphone"
    if len(name) >= 15 and name[:8].isdigit() and name[8] == "_":
        return "android"
    if "collage" in name:
        return "collage"
    if name.startswith("screenshot"):
        return "screenshot"
    return "other"


def collect_images(presentation_dir):
    brackets = defaultdict(list)

    for entry in sorted(os.listdir(presentation_dir)):
        full = os.path.join(presentation_dir, entry)

        if os.path.isdir(full):
            bracket = entry
            for fname in sorted(os.listdir(full)):
                fpath = os.path.join(full, fname)
                if not os.path.isfile(fpath):
                    continue
                ext = os.path.splitext(fname)[1].lower()
                if ext not in IMAGE_EXTS:
                    continue
                orig = fname.split("__", 1)[1] if "__" in fname else fname
                brackets[bracket].append({
                    "path": fpath.replace("\\", "/"),
                    "filename": fname,
                    "display_name": orig,
                    "source": guess_source(orig),
                })
        elif os.path.isfile(full):
            ext = os.path.splitext(entry)[1].lower()
            if ext not in IMAGE_EXTS:
                continue
            bracket = entry.split("__")[0] if "__" in entry else "unknown"
            orig = entry.split("__", 1)[1] if "__" in entry else entry
            brackets[bracket].append({
                "path": full.replace("\\", "/"),
                "filename": entry,
                "display_name": orig,
                "source": guess_source(orig),
            })

    return brackets


def build_interactive_html(brackets, thumb_size):
    total = sum(len(imgs) for imgs in brackets.values())
    sorted_brackets = sorted(brackets.items())
    bracket_names = [b for b, _ in sorted_brackets]

    print(f"  Generating thumbnails for {total} images...")
    count = 0
    for bracket, imgs in sorted_brackets:
        for img in imgs:
            img["thumb_b64"] = make_thumbnail_b64(img["path"], thumb_size)
            count += 1
            if count % 50 == 0:
                print(f"    {count}/{total}...")
    print(f"  All {total} thumbnails done.")

    # Build JSON data for the JS app
    data = {}
    for bracket, imgs in sorted_brackets:
        data[bracket] = []
        for img in imgs:
            data[bracket].append({
                "path": img["path"],
                "filename": img["filename"],
                "name": img["display_name"],
                "source": img["source"],
                "thumb": img.get("thumb_b64", ""),
            })

    bracket_options = "\n".join(f'<option value="{html.escape(b)}">{html.escape(b)}</option>' for b in bracket_names)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Reef Gallery - Interactive Review</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #1a1a2e; color: #e0e0e0; }}

.toolbar {{
    background: #16213e; padding: 12px 20px; position: sticky; top: 0; z-index: 100;
    box-shadow: 0 2px 10px rgba(0,0,0,0.5); display: flex; align-items: center; gap: 15px; flex-wrap: wrap;
}}
.toolbar h1 {{ font-size: 1.3em; color: #e94560; white-space: nowrap; }}
.toolbar .stats {{ color: #888; font-size: 0.85em; }}
.toolbar .stats b {{ color: #e94560; }}
.toolbar button {{
    padding: 6px 14px; border: none; border-radius: 4px; cursor: pointer; font-size: 0.85em;
}}
.btn-move {{ background: #e94560; color: white; }}
.btn-move:disabled {{ background: #555; cursor: not-allowed; }}
.btn-export {{ background: #0f3460; color: #5bc0eb; }}
.btn-undo {{ background: #6d4c41; color: #d7ccc8; }}
.toolbar select {{ padding: 5px; border-radius: 4px; background: #0f3460; color: #5bc0eb; border: 1px solid #333; }}
.sel-count {{ color: #e94560; font-weight: bold; font-size: 0.9em; min-width: 80px; }}

.nav {{
    background: #0f3460; padding: 8px 20px; position: sticky; top: 55px; z-index: 99;
    overflow-x: auto; white-space: nowrap; display: flex; gap: 4px;
}}
.nav a {{
    color: #5bc0eb; text-decoration: none; font-size: 0.8em; padding: 4px 10px;
    border-radius: 4px; cursor: pointer;
}}
.nav a:hover, .nav a.active {{ background: #16213e; }}
.nav a .badge {{ background: #e94560; color: white; border-radius: 8px; padding: 1px 5px; font-size: 0.75em; margin-left: 4px; }}

.bracket {{ padding: 15px 20px; border-bottom: 2px solid #222; }}
.bracket-header {{
    display: flex; align-items: center; gap: 12px; margin-bottom: 10px; flex-wrap: wrap;
}}
.bracket-header h2 {{ color: #e94560; font-size: 1.2em; }}
.bracket-header .count {{ color: #666; font-size: 0.85em; }}
.select-all {{ font-size: 0.75em; color: #5bc0eb; cursor: pointer; text-decoration: underline; }}

.bar {{ display: flex; height: 5px; border-radius: 3px; overflow: hidden; margin-bottom: 10px; }}

.grid {{ display: flex; flex-wrap: wrap; gap: 6px; }}
.card {{
    position: relative; border-radius: 5px; overflow: hidden; background: #222;
    cursor: pointer; transition: transform 0.1s;
}}
.card:hover {{ transform: scale(1.03); }}
.card.selected {{ outline: 3px solid #e94560; outline-offset: -3px; }}
.card.moved {{ opacity: 0.5; }}
.card img {{ display: block; width: {thumb_size}px; height: {thumb_size}px; object-fit: cover; }}
.card .label {{
    position: absolute; bottom: 0; left: 0; right: 0; background: rgba(0,0,0,0.75);
    color: #ddd; font-size: 0.6em; padding: 2px 4px; white-space: nowrap; overflow: hidden;
    text-overflow: ellipsis; opacity: 0; transition: opacity 0.15s;
}}
.card:hover .label {{ opacity: 1; }}
.card .dot {{
    position: absolute; top: 3px; right: 3px; width: 8px; height: 8px; border-radius: 50%;
    border: 1px solid rgba(255,255,255,0.4);
}}
.card .check {{
    position: absolute; top: 3px; left: 3px; width: 18px; height: 18px; border-radius: 50%;
    background: rgba(233,69,96,0.9); color: white; font-size: 12px; line-height: 18px;
    text-align: center; display: none;
}}
.card.selected .check {{ display: block; }}
.card .move-badge {{
    position: absolute; bottom: 20px; left: 0; right: 0; background: rgba(233,69,96,0.9);
    color: white; font-size: 0.6em; padding: 2px; text-align: center; display: none;
}}
.card.moved .move-badge {{ display: block; }}

.dot-iphone {{ background: #2d6a4f; }}
.dot-android {{ background: #5a189a; }}
.dot-facebook {{ background: #1d3557; }}
.dot-collage {{ background: #6d4c41; }}
.dot-other {{ background: #424242; }}

.legend {{
    display: flex; gap: 12px; padding: 8px 20px; background: #16213e; flex-wrap: wrap;
    border-bottom: 1px solid #333;
}}
.legend-item {{ display: flex; align-items: center; gap: 4px; font-size: 0.75em; }}
.legend-dot {{ width: 10px; height: 10px; border-radius: 50%; }}

.changes-log {{
    background: #111; padding: 15px 20px; font-family: monospace; font-size: 0.8em;
    max-height: 200px; overflow-y: auto; display: none;
}}
.changes-log.visible {{ display: block; }}
.changes-log div {{ padding: 2px 0; }}
.changes-log .move-entry {{ color: #5bc0eb; }}

.lightbox {{
    display: none; position: fixed; top: 0; left: 0; right: 0; bottom: 0;
    background: rgba(0,0,0,0.95); z-index: 1000; justify-content: center; align-items: center;
}}
.lightbox.active {{ display: flex; }}
.lightbox img {{ max-width: 90vw; max-height: 85vh; object-fit: contain; }}
.lightbox .close {{ position: absolute; top: 15px; right: 25px; color: white; font-size: 2em; cursor: pointer; }}
.lightbox .info {{ position: absolute; bottom: 15px; color: #aaa; font-size: 0.85em; text-align: center; width: 100%; }}
</style>
</head>
<body>

<div class="toolbar">
    <h1>Reef Gallery Review</h1>
    <div class="stats"><b id="total-count">{total}</b> images, <b id="bracket-count">{len(sorted_brackets)}</b> brackets</div>
    <span class="sel-count"><span id="sel-count">0</span> selected</span>
    <label>Move to:</label>
    <select id="target-bracket">{bracket_options}</select>
    <button class="btn-move" id="btn-move" disabled onclick="moveSelected()">Move</button>
    <button class="btn-undo" onclick="undoLast()">Undo</button>
    <button class="btn-export" onclick="toggleLog()">Changes Log</button>
    <button class="btn-export" onclick="exportChanges()">Export JSON</button>
</div>

<div class="legend">
    <div class="legend-item"><div class="legend-dot" style="background:#2d6a4f"></div> iPhone</div>
    <div class="legend-item"><div class="legend-dot" style="background:#5a189a"></div> Android</div>
    <div class="legend-item"><div class="legend-dot" style="background:#1d3557"></div> Facebook</div>
    <div class="legend-item"><div class="legend-dot" style="background:#6d4c41"></div> Collage</div>
    <div class="legend-item"><div class="legend-dot" style="background:#424242"></div> Other</div>
</div>

<div class="nav" id="nav"></div>
<div class="changes-log" id="changes-log"><div style="color:#666">No changes yet.</div></div>
<div id="gallery"></div>

<div class="lightbox" id="lightbox">
    <span class="close" onclick="closeLightbox()">&times;</span>
    <img id="lb-img" src="">
    <div class="info" id="lb-info"></div>
</div>

<script>
const DATA = {json.dumps(data)};
const BRACKETS = {json.dumps(bracket_names)};
const SOURCE_COLORS = {{iphone:'#2d6a4f', android:'#5a189a', facebook:'#1d3557', collage:'#6d4c41', other:'#424242'}};

let selected = new Set(); // "bracket::index"
let changes = []; // {{from, to, filename, path}}
let undoStack = [];

function render() {{
    const nav = document.getElementById('nav');
    const gallery = document.getElementById('gallery');
    nav.innerHTML = '';
    gallery.innerHTML = '';

    let totalCount = 0;
    for (const b of BRACKETS) {{
        const imgs = DATA[b] || [];
        totalCount += imgs.length;

        // Nav link
        const a = document.createElement('a');
        a.href = '#sect-' + b;
        a.innerHTML = b + ' <span class="badge">' + imgs.length + '</span>';
        nav.appendChild(a);

        // Section
        const sect = document.createElement('div');
        sect.className = 'bracket';
        sect.id = 'sect-' + b;

        // Header
        const hdr = document.createElement('div');
        hdr.className = 'bracket-header';
        hdr.innerHTML = '<h2>' + esc(b) + '</h2><span class="count">' + imgs.length +
            ' images</span><span class="select-all" onclick="selectAllIn(\'' + esc(b) + '\')">select all</span>' +
            '<span class="select-all" onclick="deselectAllIn(\'' + esc(b) + '\')">deselect all</span>';
        sect.appendChild(hdr);

        // Source bar
        const srcCounts = {{}};
        imgs.forEach(im => srcCounts[im.source] = (srcCounts[im.source]||0) + 1);
        const bar = document.createElement('div');
        bar.className = 'bar';
        for (const [src, cnt] of Object.entries(srcCounts).sort((a,b) => b[1]-a[1])) {{
            const d = document.createElement('div');
            d.style.width = (cnt/imgs.length*100) + '%';
            d.style.background = SOURCE_COLORS[src] || '#424242';
            d.title = src + ': ' + cnt;
            bar.appendChild(d);
        }}
        sect.appendChild(bar);

        // Grid
        const grid = document.createElement('div');
        grid.className = 'grid';
        imgs.forEach((img, idx) => {{
            const key = b + '::' + idx;
            const card = document.createElement('div');
            card.className = 'card' + (selected.has(key) ? ' selected' : '');
            card.dataset.key = key;
            card.dataset.bracket = b;
            card.dataset.idx = idx;

            const thumbSrc = img.thumb ? 'data:image/jpeg;base64,' + img.thumb : '';
            card.innerHTML =
                '<img src="' + thumbSrc + '">' +
                '<div class="dot dot-' + img.source + '"></div>' +
                '<div class="check">&#10003;</div>' +
                '<div class="label">' + esc(img.name) + '</div>';

            card.onclick = (e) => {{
                if (e.shiftKey || e.ctrlKey) {{
                    // Multi-select
                    toggleSelect(key, card);
                }} else if (e.detail === 2) {{
                    // Double-click = lightbox
                    openLightbox(img, b);
                }} else {{
                    toggleSelect(key, card);
                }}
            }};
            grid.appendChild(card);
        }});
        sect.appendChild(grid);
        gallery.appendChild(sect);
    }}

    document.getElementById('total-count').textContent = totalCount;
    updateSelCount();
}}

function esc(s) {{ const d = document.createElement('div'); d.textContent = s; return d.innerHTML; }}

function toggleSelect(key, card) {{
    if (selected.has(key)) {{
        selected.delete(key);
        card.classList.remove('selected');
    }} else {{
        selected.add(key);
        card.classList.add('selected');
    }}
    updateSelCount();
}}

function selectAllIn(bracket) {{
    const imgs = DATA[bracket] || [];
    imgs.forEach((_, idx) => selected.add(bracket + '::' + idx));
    render();
}}

function deselectAllIn(bracket) {{
    const imgs = DATA[bracket] || [];
    imgs.forEach((_, idx) => selected.delete(bracket + '::' + idx));
    render();
}}

function updateSelCount() {{
    const n = selected.size;
    document.getElementById('sel-count').textContent = n;
    document.getElementById('btn-move').disabled = n === 0;
}}

function moveSelected() {{
    const target = document.getElementById('target-bracket').value;
    if (!target) return;
    const batch = [];

    // Sort by bracket to process correctly
    const byBracket = {{}};
    for (const key of selected) {{
        const [b, idx] = key.split('::');
        if (b === target) continue;
        if (!byBracket[b]) byBracket[b] = [];
        byBracket[b].push(parseInt(idx));
    }}

    for (const [fromBracket, indices] of Object.entries(byBracket)) {{
        // Sort descending so splicing doesn't shift indices
        indices.sort((a,b) => b - a);
        for (const idx of indices) {{
            const img = DATA[fromBracket][idx];
            // Update filename prefix
            const newFilename = target + '__' + img.name;
            const entry = {{from: fromBracket, to: target, filename: img.filename, newFilename, path: img.path, name: img.name}};
            batch.push(entry);

            // Move in data
            DATA[fromBracket].splice(idx, 1);
            img.filename = newFilename;
            DATA[target].push(img);
        }}
    }}

    if (batch.length > 0) {{
        changes.push(...batch);
        undoStack.push(batch);
        updateLog();
    }}

    selected.clear();
    render();
}}

function undoLast() {{
    const batch = undoStack.pop();
    if (!batch) return;

    for (const entry of batch) {{
        // Find the image in the target bracket and move it back
        const targetImgs = DATA[entry.to];
        const idx = targetImgs.findIndex(im => im.path === entry.path);
        if (idx >= 0) {{
            const img = targetImgs.splice(idx, 1)[0];
            img.filename = entry.filename;
            DATA[entry.from].push(img);
        }}
        // Remove from changes
        const ci = changes.findIndex(c => c.path === entry.path && c.to === entry.to);
        if (ci >= 0) changes.splice(ci, 1);
    }}

    updateLog();
    render();
}}

function updateLog() {{
    const log = document.getElementById('changes-log');
    if (changes.length === 0) {{
        log.innerHTML = '<div style="color:#666">No changes yet.</div>';
        return;
    }}
    log.innerHTML = changes.map(c =>
        '<div class="move-entry">' + esc(c.name) + ': ' + esc(c.from) + ' &rarr; ' + esc(c.to) + '</div>'
    ).join('');
    log.scrollTop = log.scrollHeight;
}}

function toggleLog() {{
    document.getElementById('changes-log').classList.toggle('visible');
}}

function exportChanges() {{
    if (changes.length === 0) {{ alert('No changes to export.'); return; }}
    const blob = new Blob([JSON.stringify(changes, null, 2)], {{type: 'application/json'}});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'gallery_changes.json';
    a.click();
    URL.revokeObjectURL(url);
}}

function openLightbox(img, bracket) {{
    const lb = document.getElementById('lightbox');
    const lbImg = document.getElementById('lb-img');
    const info = document.getElementById('lb-info');
    lbImg.src = 'file:///' + img.path;
    info.textContent = bracket + '  |  ' + img.name + '  |  ' + img.source;
    lb.classList.add('active');
}}

function closeLightbox() {{
    document.getElementById('lightbox').classList.remove('active');
}}

document.addEventListener('keydown', e => {{ if (e.key === 'Escape') closeLightbox(); }});

render();
</script>
</body>
</html>"""


def apply_changes(changes_json_path, presentation_dir):
    """Apply moves from an exported JSON file."""
    with open(changes_json_path, "r", encoding="utf-8") as f:
        changes = json.load(f)

    print(f"Applying {len(changes)} moves...")
    for entry in changes:
        src_path = entry["path"].replace("/", os.sep)
        from_bracket = entry["from"]
        to_bracket = entry["to"]
        new_filename = entry["newFilename"]

        # Determine destination directory
        dest_dir = os.path.join(presentation_dir, to_bracket)
        if not os.path.isdir(dest_dir):
            os.makedirs(dest_dir)

        dest_path = os.path.join(dest_dir, new_filename)

        if not os.path.isfile(src_path):
            print(f"  SKIP (not found): {src_path}")
            continue

        if os.path.exists(dest_path):
            print(f"  SKIP (exists): {dest_path}")
            continue

        shutil.move(src_path, dest_path)
        print(f"  Moved: {entry['name']}  {from_bracket} -> {to_bracket}")

    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Interactive gallery report")
    parser.add_argument("--thumbsize", type=int, default=150)
    parser.add_argument("--no-open", action="store_true")
    parser.add_argument("--output", type=str, default=OUTPUT_HTML)
    parser.add_argument("--apply", type=str, help="Apply changes from exported JSON")
    args = parser.parse_args()

    if args.apply:
        apply_changes(args.apply, PRESENTATION)
        return

    print("Scanning presentation folder...")
    if not os.path.isdir(PRESENTATION):
        print("  ERROR: directory not found")
        sys.exit(1)

    brackets = collect_images(PRESENTATION)
    total = sum(len(imgs) for imgs in brackets.values())
    print(f"  Found {total} images in {len(brackets)} brackets")
    for b in sorted(brackets):
        print(f"    {b}: {len(brackets[b])}")

    html_content = build_interactive_html(brackets, args.thumbsize)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"\nGallery: {args.output}")

    if not args.no_open:
        webbrowser.open(args.output)


if __name__ == "__main__":
    main()
