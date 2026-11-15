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
import os
from pathlib import Path

import requests

ENDPOINT = "https://serpapi.com/search.json"
ROOT = Path(__file__).parent
_cache = {}  # search params -> results, so reruns don't spend extra searches


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
    ck = tuple(sorted(params.items()))
    if ck in _cache:
        return _cache[ck]
    key = get_key()
    if not key:
        raise RuntimeError("No SerpApi key. Put SERPAPI_KEY in .streamlit/secrets.toml (see README).")
    r = requests.get(ENDPOINT, params={**params, "api_key": key}, timeout=20)
    r.raise_for_status()
    data = r.json()
    if "error" in data:
        if "hasn't returned any results" in data["error"]:
            data = {}
        else:
            raise RuntimeError(f"SerpApi: {data['error']}")
    out = [{"title": o.get("title"), "link": o.get("link"), "snippet": o.get("snippet"), "date": o.get("date"),
            "source": (o.get("publication_info") or {}).get("summary")}  # authors/journal/year on Scholar
           for o in data.get("organic_results", [])[:num]]
    _cache[ck] = out
    return out


if __name__ == "__main__":
    print("SerpApi key found" if have_key() else "No SerpApi key found (see README: SerpApi key setup)")
