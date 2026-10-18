"""Check whether the upstream datasets changed since the team's last download.
Uses only the standard library and curl. Run from the repo root:
    python scripts/check_updates.py                  # compare with data/manifest.json: exit 1 changed, 2 source unreachable
    python scripts/check_updates.py --accept         # record today's fingerprints as the new baseline
    python scripts/check_updates.py --report r.md    # also write a Markdown report (used by the GitHub Action)

Run --accept right after scripts/download_data.py, so the manifest says which version the model was built on.

Fingerprints:
- INEGI zips and the IMIP growth file: size, read with a 1-byte ranged GET (HEAD answers without it on INEGI).
  ETag and Last-Modified differ between INEGI replicas, so they are recorded but not compared.
- IMIP layers: record count and a hash of every record (attributes and geometry), because the server
  does not publish a last-edit date.
Source URLs are read from download_data.py, so there is a single list to maintain.
"""
import argparse
import ast
import hashlib
import json
import subprocess
import sys
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "manifest.json"
PAGE = 2000  # the IMIP server returns at most 2000 records per request


def source_constants():
    """Read the URL constants from download_data.py without importing it (it needs geopandas)."""
    tree = ast.parse((ROOT / "scripts" / "download_data.py").read_text())
    found = {}
    for n in tree.body:
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and isinstance(n.value, (ast.Constant, ast.Dict)):
            found[n.targets[0].id] = ast.literal_eval(n.value)  # skips computed values such as RAW = Path(...) / ...
    return found


def curl(url, *extra):
    cmd = ["curl", "-sSL", "--fail", "--max-time", "120", "--retry", "3", "--retry-all-errors",
           "--retry-delay", "5", "-A", "Mozilla/5.0", *extra, url]
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def file_fingerprint(url):
    out = curl(url, "-r", "0-0", "-D", "-", "-o", "/dev/null")
    headers = {}
    for line in out.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            headers[k.strip().lower()] = v.strip()
    size = headers.get("content-range", "").rsplit("/", 1)[-1] or headers.get("content-length", "")
    # INEGI answers from replicas with different ETag and Last-Modified for the same file, so only the
    # size is compared; the headers are kept under "info" for reference
    return {"size": size, "info": {"etag": headers.get("etag", ""), "last_modified": headers.get("last-modified", "")}}


def layer_fingerprint(base, path):
    sha, count, offset = hashlib.sha256(), 0, 0
    while True:
        q = urllib.parse.urlencode(dict(where="1=1", outFields="*", outSR=4326, f="json", orderByFields="OBJECTID",
                                        resultOffset=offset, resultRecordCount=PAGE))
        page = json.loads(curl(f"{base}{path}/query?{q}")).get("features", [])
        sha.update(json.dumps(page, sort_keys=True).encode())
        count += len(page)
        if len(page) < PAGE:
            break
        offset += PAGE
    return {"records": count, "sha256": sha.hexdigest()[:16]}


def current_fingerprints():
    c = source_constants()
    sources = {
        "inegi_censo_2020": c["INEGI_CENSO"],
        "inegi_marco_geo": c["INEGI_MG"],
        "imip_crecimiento_historico": c["IMIP_CRECIMIENTO"],
    }
    if "INEGI_CENSO_2010" in c:
        sources["inegi_censo_2010"] = c["INEGI_CENSO_2010"]
    result = {}
    for name, url in sources.items():
        result[name] = _safe(lambda url=url: file_fingerprint(url))
    for name, path in c["IMIP_LAYERS"].items():
        result[f"imip_{name}"] = _safe(lambda path=path: layer_fingerprint(c["IMIP"], path))
    return result


def _safe(fn):
    # an unreachable source is reported, not fatal: the other sources still get checked
    try:
        return fn()
    except Exception as e:
        return {"error": f"{type(e).__name__}: {str(e)[:120]}"}


def compare(old, new):
    changed, errors, added = {}, {}, []
    for name, fp in new.items():
        if "error" in fp:
            errors[name] = fp["error"]
        elif name not in old:
            added.append(name)
        else:
            diff = {k: (old[name].get(k), fp[k]) for k in fp if k != "info" and old[name].get(k) != fp[k]}
            if diff:
                changed[name] = diff
    removed = [n for n in old if n not in new]
    return changed, errors, added, removed


def markdown(changed, errors, added, removed, checked_at, baseline_at):
    lines = [f"Checked {checked_at}; baseline from {baseline_at}.", ""]
    if changed:
        lines += ["## Changed", ""]
        for name, diff in changed.items():
            lines.append(f"- **{name}**: " + "; ".join(f"{k}: `{a}` → `{b}`" for k, (a, b) in diff.items()))
        lines.append("")
    if errors:
        lines += ["## Could not check", ""] + [f"- **{n}**: {e}" for n, e in errors.items()] + [""]
    if added or removed:
        lines += ["## Source list changed", ""] + [f"- new: {n}" for n in added] + [f"- gone: {n}" for n in removed] + [""]
    if not (changed or errors or added or removed):
        lines.append("Nothing changed.")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--accept", action="store_true", help="write today's fingerprints as the baseline")
    ap.add_argument("--report", help="write a Markdown report to this path")
    args = ap.parse_args()

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    new = current_fingerprints()
    if args.accept:
        failed = [n for n, fp in new.items() if "error" in fp]
        if failed:
            sys.exit(f"Not accepting a baseline with unreachable sources: {failed}")
        MANIFEST.write_text(json.dumps({"checked_at": now, "sources": new}, indent=2, sort_keys=True) + "\n")
        print(f"Baseline written to {MANIFEST} ({len(new)} sources)")
        sys.exit(0)

    if not MANIFEST.exists():
        sys.exit("No data/manifest.json yet. Run with --accept after downloading the data.")
    base = json.loads(MANIFEST.read_text())
    changed, errors, added, removed = compare(base["sources"], new)
    text = markdown(changed, errors, added, removed, now, base["checked_at"])
    print(text)
    if args.report:
        Path(args.report).write_text(text + "\n")
    # 1 = data changed, 2 = nothing changed but some sources could not be checked, 0 = all clear
    sys.exit(1 if (changed or added or removed) else 2 if errors else 0)
