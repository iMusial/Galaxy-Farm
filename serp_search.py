"""
SerpApi search helper for Galaxy-Farm.

The API key is never stored in the code. It is looked up in this order:
  1. Streamlit secrets:  .streamlit/secrets.toml  ->  SERPAPI_KEY = "..."   (git-ignored)
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
_cache = {}  # query -> results, so reruns don't spend extra searches


def get_key():
    try:
        import streamlit as st
        key = st.secrets.get("SERPAPI_KEY")
        if key:
            return key
    except Exception:  # Streamlit not installed / no secrets.toml
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


def search(query, num=5, location=None):
    """Google organic results via SerpApi: list of {title, link, snippet, date}."""
    if query in _cache:
        return _cache[query]
    key = get_key()
    if not key:
        raise RuntimeError("No SerpApi key. Put SERPAPI_KEY in .streamlit/secrets.toml (see README).")
    params = {"engine": "google", "q": query, "num": num, "api_key": key}
    if location:
        params["location"] = location
    r = requests.get(ENDPOINT, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    if "error" in data:
        raise RuntimeError(f"SerpApi: {data['error']}")
    out = [{"title": o.get("title"), "link": o.get("link"), "snippet": o.get("snippet"), "date": o.get("date")}
           for o in data.get("organic_results", [])[:num]]
    _cache[query] = out
    return out


if __name__ == "__main__":
    print("SerpApi key found" if have_key() else "No SerpApi key found (see README: SerpApi key setup)")
