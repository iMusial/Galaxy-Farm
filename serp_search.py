"""
SerpApi search helper for Galaxy-Farm.

The API key is never stored in the code. It is looked up in this order:
  1. Streamlit secrets:  .streamlit/secrets.toml  ->  SERPAPI_KEY = "..."   (git-ignored)
     (the repo-root .streamlit/secrets.toml is also read directly, so it works when the app starts from a subfolder)
  2. Environment variable SERPAPI_KEY
  3. A git-ignored secrets.env / .env file next to this script:  SERPAPI_KEY=...

Usage:
    from serp_search import search, have_key
    if have_key():
        results = search("Iowa cereal rye termination soybean site:iastate.edu")
"""
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

import requests

ENDPOINT = "https://serpapi.com/search.json"
ROOT = Path(__file__).parent
# Results are kept in memory and in a git-ignored file, so restarts and reruns don't spend the same search twice.
DISK_CACHE = ROOT / ".serp_cache.json"
MAX_AGE_DAYS = 30
_cache = None  # json(params) -> {"t": unix time, "results": [...]}
_lock = threading.Lock()


def _load():
    global _cache
    if _cache is None:
        try:
            _cache = json.loads(DISK_CACHE.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            _cache = {}
    return _cache


def _save():
    try:
        tmp = DISK_CACHE.with_suffix(".tmp")
        tmp.write_text(json.dumps(_cache, ensure_ascii=False), encoding="utf-8")
        tmp.replace(DISK_CACHE)
    except OSError:
        pass  # read-only disk: the in-memory copy still works


def _resolve_links(items):
    """Google now returns google.com/goto redirect links. Follow each redirect once (no page download) so results
    carry the real URL and its domain, which callers use to judge the source. Leaves the link as-is on failure."""
    def one(it):
        link = it.get("link") or ""
        if urlparse(link).netloc.endswith("google.com"):
            try:
                # GET, not HEAD: Google answers HEAD with a plain 200. stream=True reads only the redirect response.
                with requests.get(link, allow_redirects=False, timeout=8, stream=True) as r:
                    if r.headers.get("location", "").startswith("http"):
                        link = it["link"] = r.headers["location"]
            except requests.RequestException:
                pass
        it["domain"] = urlparse(link).netloc.lower().removeprefix("www.")

    with ThreadPoolExecutor(6) as ex:
        list(ex.map(one, items))


def get_key():
    try:
        import streamlit as st
        key = st.secrets.get("SERPAPI_KEY")
        if key:
            return key
    except Exception:  # Streamlit not installed / no secrets.toml
        pass
    try:
        import tomllib
        key = tomllib.loads((ROOT / ".streamlit" / "secrets.toml").read_text(encoding="utf-8-sig")).get("SERPAPI_KEY")
        if key:
            return key
    except Exception:  # no file, bad TOML, or Python < 3.11
        pass
    if os.environ.get("SERPAPI_KEY"):
        return os.environ["SERPAPI_KEY"]
    for name in ("secrets.env", ".env"):
        try:
            for line in (ROOT / name).read_text(encoding="utf-8-sig").splitlines():
                k, _, v = line.partition("=")
                if k.strip() == "SERPAPI_KEY" and v.strip().strip('"').strip("'"):
                    return v.strip().strip('"').strip("'")
        except OSError:
            continue
    return None


def have_key():
    return bool(get_key())


def search(query, num=5, location=None, engine="google", **extra):
    """Organic results via SerpApi: list of {title, link, snippet, date, source}.

    engine: "google" (default) or another SerpApi engine with organic_results, e.g. "google_scholar".
    extra:  any other SerpApi parameters (e.g. tbs="qdr:m" for the past month).
    """
    params = {"engine": engine, "q": query, "num": num, **extra}
    if location:
        params["location"] = location
    ck = json.dumps(params, sort_keys=True)
    with _lock:
        hit = _load().get(ck)
    if hit and time.time() - hit["t"] < MAX_AGE_DAYS * 86400:
        return hit["results"]
    key = get_key()
    if not key:
        raise RuntimeError("No SerpApi key. Put SERPAPI_KEY in .streamlit/secrets.toml (see README).")
    r = requests.get(ENDPOINT, params={**params, "api_key": key}, timeout=45)
    r.raise_for_status()
    data = r.json()
    if "error" in data:
        if "hasn't returned any results" in data["error"]:
            data = {}
        else:
            raise RuntimeError(f"SerpApi: {data['error']}")
    out = [{"title": o.get("title"), "link": o.get("link"), "snippet": o.get("snippet"), "date": o.get("date"),
            "site": o.get("source") if isinstance(o.get("source"), str) else None,  # e.g. "Mississippi Soybean Promotion Board"
            "source": (o.get("publication_info") or {}).get("summary")}  # authors/journal/year on Scholar
           for o in data.get("organic_results", [])[:num]]
    _resolve_links(out)
    with _lock:
        _load()[ck] = {"t": time.time(), "results": out}
        _save()
    return out


if __name__ == "__main__":
    print("SerpApi key found" if have_key() else "No SerpApi key found (see README: SerpApi key setup)")
