"""
PRISM Climate Explorer + USDA NRCS SSURGO / STATSGO2 Soil Data
--------------------------------------------------------------
Search for a place, draw a rectangle/polygon (or drop a pin) on a satellite map,
and pull PRISM climate data, growing degree days (GDD), and USDA NRCS SSURGO/STATSGO2 soil properties.

Install:   pip install streamlit streamlit-folium folium requests numpy pandas plotly
Run:       streamlit run prism_explorer.py

Data Sources:
- PRISM Climate Group, Oregon State University (https://prism.oregonstate.edu)
- USDA NRCS Soil Data Access (SDA) REST Web Service for SSURGO/STATSGO2
- NASA POWER Soil Wetness (optional)
"""
import datetime as dt
import hashlib
import base64
import io
import json
import re
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import branca.colormap as bcm
import folium
from branca.element import MacroElement, Template
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st
from folium.plugins import Draw
from folium.raster_layers import ImageOverlay
from PIL import Image  # installed with streamlit
from streamlit_folium import st_folium

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # serp_search.py lives in the repo root
try:
    import serp_search
except ImportError:
    serp_search = None

BASE = "https://services.nacse.org/prism/data/get/us/4km"
ERRORS = {}  # url -> last failure reason
CACHE = Path(__file__).resolve().parent / "prism_cache"  # same cache no matter where the app is started from
CACHE.mkdir(exist_ok=True)
LABEL = {
    "ppt": "Precipitation",
    "tmin": "Min temp",
    "tmax": "Max temp",
    "tmean": "Mean temp",
    "gdd": "Growing Degree Days",
}
COLOR = {
    "ppt": "#2b7bba",
    "tmin": "#4a90d9",
    "tmax": "#d9534f",
    "tmean": "#e6a23c",
    "gdd": "#27ae60",
}
MAX_FILES = 1500
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# ----------------------------------------------------------------- PRISM access
_local = threading.local()


def _session():
    # One keep-alive connection per download thread instead of a new TLS handshake per file.
    if not hasattr(_local, "s"):
        _local.s = requests.Session()
    return _local.s


def fetch_zip(url):
    fp = CACHE / (re.sub(r"[^A-Za-z0-9]+", "_", url.replace(BASE, "")) + ".zip")
    if fp.exists():
        return fp.read_bytes()
    for attempt in range(3):
        try:
            r = _session().get(url, timeout=90)
        except requests.RequestException as e:
            ERRORS[url] = f"Network error: {e}"
            time.sleep(2 * (attempt + 1))
            continue
        if r.status_code == 200 and r.content[:2] == b"PK":
            tmp = fp.with_suffix(".part")
            tmp.write_bytes(r.content)
            tmp.replace(fp)  # never leave a half-written zip in the cache
            return r.content
        text = r.text[:300]
        if "more than twice" in text:
            # PRISM allows each file twice per day; retrying risks an IP block.
            ERRORS[url] = "LIMIT: " + text
            return None
        ERRORS[url] = f"HTTP {r.status_code}: {text!r}"
        if r.status_code != 429 and r.status_code < 500:
            return None  # not available (e.g. not yet released); retrying won't help
        time.sleep(2 * (attempt + 1))
    return None


def read_bil(zip_bytes):
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        names = z.namelist()
        bil = next(n for n in names if n.lower().endswith(".bil"))
        hdr = next((n for n in names if n.lower().endswith(".hdr")), None)
        h = {}
        if hdr:
            h = dict(l.split(None, 1) for l in z.read(hdr).decode().splitlines() if len(l.split()) > 1)
            h = {k.upper(): v.strip() for k, v in h.items()}
        dtype = "<f4" if h.get("BYTEORDER", "I") == "I" else ">f4"
        arr = np.frombuffer(z.read(bil), dtype=dtype).reshape(int(h.get("NROWS", 621)), int(h.get("NCOLS", 1405))).astype(float)
    arr[arr <= float(h.get("NODATA", -9999)) + 1] = np.nan
    meta = dict(ulx=float(h.get("ULXMAP", -125.0208333333333)), uly=float(h.get("ULYMAP", 49.9375)),
                xdim=float(h.get("XDIM", 0.0416666666666667)), ydim=float(h.get("YDIM", 0.0416666666666667)))
    return arr, meta


def points_in_polygon(px, py, poly):
    inside = np.zeros(px.shape, bool)
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        cond = ((yi > py) != (yj > py)) & (px < (xj - xi) * (py - yi) / (yj - yi + 1e-300) + xi)
        inside ^= cond
        j = i
    return inside


def cells_for(geom, meta, shape):
    kind, coords = geom
    nrows, ncols = shape
    to_rc = lambda lon, lat: ((meta["uly"] - lat) / meta["ydim"], (lon - meta["ulx"]) / meta["xdim"])

    def nearest(lon, lat):
        r, c = to_rc(lon, lat)
        r, c = int(round(r)), int(round(c))
        if 0 <= r < nrows and 0 <= c < ncols:
            return np.array([r]), np.array([c])
        return np.array([], int), np.array([], int)

    if kind == "point":
        return nearest(*coords)
    ring = np.array(coords)
    rs, cs = zip(*[to_rc(x, y) for x, y in ring])
    r0, r1 = max(int(np.floor(min(rs))), 0), min(int(np.ceil(max(rs))), nrows - 1)
    c0, c1 = max(int(np.floor(min(cs))), 0), min(int(np.ceil(max(cs))), ncols - 1)
    if r0 > r1 or c0 > c1:
        return np.array([], int), np.array([], int)
    rr, cc = np.meshgrid(np.arange(r0, r1 + 1), np.arange(c0, c1 + 1), indexing="ij")
    lon = meta["ulx"] + cc * meta["xdim"]
    lat = meta["uly"] - rr * meta["ydim"]
    m = points_in_polygon(lon, lat, ring)
    if not m.any():
        return nearest(*ring.mean(axis=0))
    return rr[m], cc[m]


def area_value(zip_bytes, geom):
    arr, meta = read_bil(zip_bytes)
    r, c = cells_for(geom, meta, arr.shape)
    vals = arr[r, c] if len(r) else np.array([])
    vals = vals[np.isfinite(vals)]
    return (float(vals.mean()) if len(vals) else np.nan), len(vals)


def month_span(d0, d1):
    y, m = d0.year, d0.month
    while (y, m) <= (d1.year, d1.month):
        yield y, m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def build_tasks(elements, mode, d0, d1):
    today = dt.date.today()
    tasks = []
    fetch_els = set(elements)
    if "gdd" in fetch_els:
        fetch_els.add("tmin")
        fetch_els.add("tmax")

    for el in fetch_els:
        if mode == "Daily":
            for k in range((d1 - d0).days + 1):
                d = d0 + dt.timedelta(days=k)
                if d < today:
                    tasks.append(("daily", el, d.year, d.month, d.day, f"{BASE}/{el}/{d:%Y%m%d}?format=bil"))
        else:
            for y, m in month_span(d0, d1):
                if dt.date(y, m, 1) < today.replace(day=1):
                    tasks.append(("monthly", el, y, m, 1, f"{BASE}/{el}/{y}{m:02d}?format=bil"))
    return tasks


def compute_gdd_rows(df, base_c, daily):
    piv = df.pivot_table(index=["kind", "year", "month", "day"], columns="element", values="value").reset_index()
    if "tmin" not in piv.columns or "tmax" not in piv.columns:
        return pd.DataFrame(columns=df.columns)
    tmin = piv["tmin"]
    tmax = piv["tmax"]
    lo = tmin.clip(lower=base_c)
    hi = tmax.clip(upper=30.0)
    gdd_vals = ((hi + lo) / 2 - base_c).clip(lower=0)
    if not daily:
        dates = pd.to_datetime(dict(year=piv.year, month=piv.month, day=1))
        gdd_vals = gdd_vals * dates.dt.days_in_month
    gdd_df = piv[["kind", "year", "month", "day"]].copy()
    gdd_df["element"] = "gdd"
    gdd_df["value"] = gdd_vals
    return gdd_df.dropna(subset=["value"])


def run_fetch(geom, tasks):
    ERRORS.clear()
    rows, failed, ncells = [], [], 0
    bar = st.progress(0.0, "Downloading PRISM grids…")

    def handle(t, data):
        nonlocal ncells
        if data is None:
            return False
        v, n = area_value(data, geom)
        ncells = max(ncells, n)
        rows.append(dict(kind=t[0], element=t[1], year=t[2], month=t[3], day=t[4], value=v))
        return True

    # Grids already on disk need no network: read them first, then download only what's new.
    to_get = []
    for t in tasks:
        fp = CACHE / (re.sub(r"[^A-Za-z0-9]+", "_", t[5].replace(BASE, "")) + ".zip")
        if not (fp.exists() and handle(t, fp.read_bytes())):
            to_get.append(t)
    if to_get:
        with ThreadPoolExecutor(4) as ex:
            futs = {ex.submit(fetch_zip, t[5]): t for t in to_get}
            for i, fut in enumerate(as_completed(futs), 1):
                if not handle(futs[fut], fut.result()):
                    failed.append(futs[fut])
                bar.progress(i / len(to_get), f"Downloading PRISM grids… {i}/{len(to_get)} "
                                              f"({len(tasks) - len(to_get)} already saved)")
    # Second pass, one at a time, for network hiccups only. Files that hit PRISM's twice-a-day
    # limit are not retried: asking again doesn't work and can get the IP blocked.
    still_missing = [t for t in failed if ERRORS.get(t[5], "").startswith("LIMIT")]
    retry = [t for t in failed if t not in still_missing]
    for i, t in enumerate(retry, 1):
        bar.progress(i / len(retry), f"Retrying {len(retry)} grids that failed…")
        time.sleep(1)
        if not handle(t, fetch_zip(t[5])):
            still_missing.append(t)
    bar.empty()
    return pd.DataFrame(rows), len(still_missing), ncells


# ----------------------------------------------------------------- USDA SSURGO / SDA
SDA_URL = "https://sdmdataaccess.sc.egov.usda.gov/Tabular/post.rest"


def fetch_ssurgo_data(geom):
    """Soil Data Access (SDA): map units, components and horizon properties under the selected point/polygon."""
    kind, coords = geom
    if kind == "point":
        lon, lat = coords
        wkt = f"POINT({lon:.6f} {lat:.6f})"
    else:
        ring = [(float(x), float(y)) for x, y in coords]
        if ring[0] != ring[-1]:
            ring.append(ring[0])  # WKT rings must be closed
        wkt = "POLYGON((" + ", ".join(f"{x:.6f} {y:.6f}" for x, y in ring) + "))"

    # SDA's spatial lookup is the SDA_Get_Mukey_from_intersection_with_WktWgs84() function (longitude first, WGS84)
    sql = f"""
    SELECT TOP 5000
        mu.mukey, mu.musym, mu.muname,
        c.cokey, c.compname, c.comppct_r, c.taxclname, c.drainagecl,
        ch.hzname, ch.hzdept_r, ch.hzdepb_r, ch.awc_r, ch.om_r, ch.claytotal_r, ch.sandtotal_r
    FROM mapunit mu
    INNER JOIN component c ON c.mukey = mu.mukey
    LEFT JOIN chorizon ch ON ch.cokey = c.cokey
    WHERE mu.mukey IN (SELECT DISTINCT mukey FROM SDA_Get_Mukey_from_intersection_with_WktWgs84('{wkt}'))
    ORDER BY mu.mukey, c.comppct_r DESC, c.cokey, ch.hzdept_r
    """
    try:
        r = requests.post(SDA_URL, data={"query": sql, "format": "JSON+COLUMNNAME"}, timeout=90)
    except requests.RequestException as e:
        raise RuntimeError(f"Soil Data Access could not be reached: {e}")
    if r.status_code != 200:
        m = re.search(r"<ServiceException[^>]*>(.*?)</ServiceException>", r.text, re.S)
        raise RuntimeError(f"Soil Data Access error (HTTP {r.status_code}): {(m.group(1) if m else r.text).strip()[:300]}")
    if not r.text.strip():
        return pd.DataFrame()  # nothing mapped here
    js = r.json()
    table = js.get("Table") or []
    if len(table) < 2:
        return pd.DataFrame()
    return pd.DataFrame(table[1:], columns=table[0])


def _wavg(h, col, limit):
    """Average of `col` over 0..limit cm, weighted by component % and horizon thickness."""
    x = h.dropna(subset=[col])
    w = (x.hzdepb_r.clip(upper=limit) - x.hzdept_r.clip(lower=0)).clip(lower=0) * x.comppct_r
    return float((x[col] * w).sum() / w.sum()) if w.sum() > 0 else float("nan")


def ssurgo_tab(df):
    if df is None or df.empty:
        st.info("No soil survey data was returned for this spot (open water, some urban land and unmapped areas have none).")
        return
    d = df.copy()
    for c in ["comppct_r", "hzdept_r", "hzdepb_r", "awc_r", "om_r", "claytotal_r", "sandtotal_r"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    h = d.dropna(subset=["hzdept_r", "hzdepb_r", "comppct_r"])

    comps = d.drop_duplicates("cokey")
    tot = comps.comppct_r.sum()
    awc100 = float("nan")
    if tot > 0 and not h.dropna(subset=["awc_r"]).empty:
        x = h.dropna(subset=["awc_r"])
        ov = (x.hzdepb_r.clip(upper=100) - x.hzdept_r.clip(lower=0)).clip(lower=0)
        awc100 = float((x.awc_r * ov * x.comppct_r).sum() / tot)  # awc_r is cm water per cm soil -> cm of water
    c = st.columns(4)
    c[0].metric("Organic matter (0–30 cm)", f"{_wavg(h, 'om_r', 30):.1f} %")
    c[1].metric("Clay (0–30 cm)", f"{_wavg(h, 'claytotal_r', 30):.0f} %")
    c[2].metric("Sand (0–30 cm)", f"{_wavg(h, 'sandtotal_r', 30):.0f} %")
    c[3].metric("Plant-available water (top 1 m)", f"{awc100:.1f} cm" if awc100 == awc100 else "n/a")

    st.subheader("Map units and soil components")
    mu = d.drop_duplicates("mukey")[["mukey", "musym", "muname"]]
    st.dataframe(mu.rename(columns={"mukey": "Map unit key", "musym": "Symbol", "muname": "Name"}),
                 use_container_width=True, hide_index=True)
    comp = comps[["muname", "compname", "comppct_r", "taxclname", "drainagecl"]].sort_values("comppct_r", ascending=False)
    st.dataframe(comp.rename(columns={"muname": "Map unit", "compname": "Component", "comppct_r": "% of map unit",
                                      "taxclname": "Taxonomic class", "drainagecl": "Drainage class"}),
                 use_container_width=True, hide_index=True)

    top = comps.sort_values("comppct_r", ascending=False).iloc[0]
    prof = h[h.cokey == top.cokey].sort_values("hzdept_r")
    if not prof.empty:
        st.subheader(f"Profile of the largest component: {top.compname} ({int(top.comppct_r)}%)")
        mid = (prof.hzdept_r + prof.hzdepb_r) / 2
        f = go.Figure()
        for col, lab, colr in [("claytotal_r", "Clay %", "#a1887f"), ("sandtotal_r", "Sand %", "#e6a23c"), ("om_r", "Organic matter %", "#2e7d32")]:
            f.add_scatter(x=prof[col], y=mid, name=lab, mode="lines+markers", line_color=colr)
        f.update_layout(yaxis=dict(autorange="reversed", title="Depth (cm)"), xaxis_title="percent", height=380)
        st.plotly_chart(f, use_container_width=True)
        st.dataframe(prof[["hzname", "hzdept_r", "hzdepb_r", "claytotal_r", "sandtotal_r", "om_r", "awc_r"]].rename(columns={
            "hzname": "Horizon", "hzdept_r": "Top (cm)", "hzdepb_r": "Bottom (cm)", "claytotal_r": "Clay %",
            "sandtotal_r": "Sand %", "om_r": "Organic matter %", "awc_r": "Avail. water (cm/cm)"}),
            use_container_width=True, hide_index=True)
    st.caption("Source: USDA NRCS Soil Data Access (SSURGO). Values are representative figures for each component, weighted by "
               "component share and horizon thickness. Map units are mixtures, so treat them as a guide and confirm with a soil test.")


# --------------------------------------------------------------------- helpers
def geocode(q):
    r = requests.get(
        "https://nominatim.openstreetmap.org/search",
        params=dict(q=q, format="json", limit=6),
        headers={"User-Agent": "prism-explorer-demo/1.0"},
        timeout=15,
    )
    r.raise_for_status()
    return r.json()


@st.cache_data(show_spinner=False)
def place_name(lat, lon):
    """(state, county) for a point, via OpenStreetMap reverse geocoding."""
    r = requests.get(
        "https://nominatim.openstreetmap.org/reverse",
        params=dict(lat=lat, lon=lon, format="json", zoom=8),
        headers={"User-Agent": "prism-explorer-demo/1.0"},
        timeout=15,
    )
    r.raise_for_status()
    a = r.json().get("address", {})
    return a.get("state"), a.get("county")


def parse_geom(drawings):
    if not drawings:
        return None
    g = drawings[-1]["geometry"]
    if g["type"] == "Point":
        return ("point", tuple(g["coordinates"]))
    if g["type"] == "Polygon":
        return ("poly", [tuple(p) for p in g["coordinates"][0]])
    return None


def approx_area_km2(ring):
    a = np.array(ring)
    lat0 = np.radians(a[:, 1].mean())
    x = a[:, 0] * 111.32 * np.cos(lat0)
    y = a[:, 1] * 110.57
    return abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2


def convert(df, units):
    df = df.copy()
    if units == "US (in, °F)":
        is_p = df.element == "ppt"
        is_gdd = df.element == "gdd"
        df.loc[is_p, "value"] /= 25.4
        df.loc[is_gdd, "value"] *= 9 / 5
        df.loc[~is_p & ~is_gdd, "value"] = df.loc[~is_p & ~is_gdd, "value"] * 9 / 5 + 32
    return df


# --------------------------------------------------------------------- map overlays
SDM_WMS = "https://SDMDataAccess.sc.egov.usda.gov/Spatial/SDM.wms"
RAMPS = {"ppt": ["#f7fbff", "#c6dbef", "#6baed6", "#2171b5", "#08306b"],
         "temp": ["#313695", "#74add1", "#ffffbf", "#f46d43", "#a50026"]}


def add_reference_overlays(m):
    """Layers you can switch on in the map's layer control (all off by default)."""
    folium.WmsTileLayer(url=SDM_WMS, layers="mapunitpoly", fmt="image/png", transparent=True, version="1.1.1",
                        name="Soil map units (SSURGO, zoom in to see)", overlay=True, control=True, show=False,
                        attr="USDA NRCS Soil Data Access").add_to(m)
    nm = "https://basemap.nationalmap.gov/arcgis/rest/services"
    folium.TileLayer(f"{nm}/USGSShadedReliefOnly/MapServer/tile/{{z}}/{{y}}/{{x}}", attr="USGS The National Map",
                     name="Terrain shading", overlay=True, control=True, show=False, opacity=0.6).add_to(m)
    folium.TileLayer(f"{nm}/USGSTopo/MapServer/tile/{{z}}/{{y}}/{{x}}", attr="USGS The National Map",
                     name="USGS topo map", overlay=True, control=True, show=False, opacity=0.85).add_to(m)


def grid_url(el, kind, y, mth, day):
    if kind == "daily":
        return f"{BASE}/{el}/{y}{mth:02d}{day:02d}?format=bil"
    return f"{BASE}/{el}/{y}{mth:02d}?format=bil"


def prism_grid_layer(el, kind, y, mth, day, geom, us):
    """Colour-coded PRISM grid cells around the selection as a map overlay (uses the cached download)."""
    z = fetch_zip(grid_url(el, kind, y, mth, day))
    if z is None:
        return None
    arr, meta = read_bil(z)
    xd, yd = meta["xdim"], meta["ydim"]
    west, north = meta["ulx"] - xd / 2, meta["uly"] + yd / 2  # header gives the centre of the top-left cell
    pts = np.array([geom[1]]) if geom[0] == "point" else np.array(geom[1])
    w, e, so, n = pts[:, 0].min(), pts[:, 0].max(), pts[:, 1].min(), pts[:, 1].max()
    margin = max(0.4, 0.75 * max(e - w, n - so))  # small window keeps the map projection error negligible
    c0, c1 = max(int(np.floor((w - margin - west) / xd)), 0), min(int(np.ceil((e + margin - west) / xd)), arr.shape[1])
    r0, r1 = max(int(np.floor((north - (n + margin)) / yd)), 0), min(int(np.ceil((north - (so - margin)) / yd)), arr.shape[0])
    sub = arr[r0:r1, c0:c1].copy()
    if sub.size == 0 or not np.isfinite(sub).any():
        return None
    if el == "ppt":
        sub, unit = (sub / 25.4, "in") if us else (sub, "mm")
    else:
        sub, unit = (sub * 9 / 5 + 32, "°F") if us else (sub, "°C")
    ok = np.isfinite(sub)
    lo, hi = np.percentile(sub[ok], [2, 98])
    hi = hi if hi > lo else lo + 1
    t = np.clip((sub - lo) / (hi - lo), 0, 1)
    ramp = RAMPS["ppt" if el == "ppt" else "temp"]
    rgb = np.array([[int(c[i:i + 2], 16) for i in (1, 3, 5)] for c in ramp], float)
    stops = np.linspace(0, 1, len(ramp))
    rgba = np.zeros(sub.shape + (4,), np.uint8)
    for ch in range(3):
        rgba[..., ch] = np.interp(t, stops, rgb[:, ch]).astype(np.uint8)
    rgba[..., 3] = np.where(ok, 190, 0)
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, "PNG")
    uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    bounds = [[north - r1 * yd, west + c0 * xd], [north - r0 * yd, west + c1 * xd]]
    layer = ImageOverlay(image=uri, bounds=bounds, opacity=0.9, name=f"PRISM {LABEL.get(el, el)}",
                         overlay=True, control=True, show=True)
    cmap = bcm.LinearColormap(ramp, vmin=float(lo), vmax=float(hi), caption=f"{LABEL.get(el, el)} ({unit})")
    return layer, cmap, bounds


def map_tab(d, units):
    geom, raw = d["geom"], d["df"]
    avail = [e for e in ["ppt", "tmin", "tmax", "tmean"] if e in set(raw.element)]
    if not avail:
        st.info("Fetch at least one PRISM variable to see it on the map.")
        return
    c1, c2 = st.columns([1, 2])
    el = c1.selectbox("PRISM layer", avail, format_func=LABEL.get, key="ov_el")
    rows = (raw[(raw.element == el) & raw.kind.isin(["monthly", "daily"])][["kind", "year", "month", "day"]]
            .drop_duplicates().sort_values(["year", "month", "day"]).reset_index(drop=True))
    if rows.empty:
        st.info("No PRISM grids were downloaded for this variable.")
        return
    lab = lambda i: (f"{rows.year[i]}-{rows.month[i]:02d}-{rows.day[i]:02d}" if rows.kind[i] == "daily"
                     else f"{rows.year[i]}-{rows.month[i]:02d}")
    i = c2.select_slider("Date", options=list(range(len(rows))), value=len(rows) - 1, format_func=lab,
                         key=f"ov_date_{el}_{len(rows)}_{lab(0)}_{lab(len(rows) - 1)}")
    res = prism_grid_layer(el, rows.kind[i], int(rows.year[i]), int(rows.month[i]), int(rows.day[i]), geom,
                           units.startswith("US"))
    if res is None:
        st.warning("Could not load that PRISM grid. " + (str(next(iter(ERRORS.values()))) if ERRORS else ""))
        return
    layer, cmap, bounds = res
    m = folium.Map(tiles=None)
    folium.TileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                     attr="Esri World Imagery", name="Satellite").add_to(m)
    folium.TileLayer("OpenStreetMap", name="Street map").add_to(m)
    layer.add_to(m)
    add_reference_overlays(m)
    if geom[0] == "poly":
        folium.Polygon(locations=[(y, x) for x, y in geom[1]], color="#ff3b30", weight=3, fill=True,
                       fill_opacity=0.05, name="Your selection").add_to(m)
    else:
        folium.CircleMarker(location=(geom[1][1], geom[1][0]), radius=7, color="#ff3b30", fill=True,
                            name="Your selection").add_to(m)
    cmap.add_to(m)
    m.fit_bounds(bounds)
    folium.LayerControl().add_to(m)
    st_folium(m, height=520, use_container_width=True, key="data_map", returned_objects=[])
    st.caption("Each coloured block is one 4 km PRISM cell; red marks your selection. Use the layer control (top right) to "
               "add soil map units (zoom in), terrain shading, or a USGS topo map, or to hide the climate layer.")


# -------------------------------------------------------------------- heatmaps
HEAT_SCALE = {"ppt": "Blues", "gdd": "YlOrRd", "tmin": "RdYlBu_r", "tmax": "RdYlBu_r", "tmean": "RdYlBu_r", "wet": "YlGnBu"}
HEAT_ANOM = {"ppt": "BrBG", "wet": "BrBG"}  # wetter = green, drier = brown; temperatures use RdBu_r


def make_heat(df, title, unit, key, daily, anomaly):
    """df has year, month, day, value. Monthly: year x month grid. Daily: (year-month) x day-of-month grid."""
    if daily:
        d = df.assign(row=df.year.astype(int).astype(str) + "-" + df.month.astype(int).astype(str).str.zfill(2))
        z = d.pivot_table(index="row", columns="day", values="value", aggfunc="mean").reindex(columns=range(1, 32)).sort_index()
        x, xtitle = list(range(1, 32)), "Day of month"
    else:
        z = df.pivot_table(index="year", columns="month", values="value", aggfunc="mean").reindex(columns=range(1, 13)).sort_index()
        if anomaly:
            z = z - z.mean(axis=0)  # difference from that calendar month's average across the selected years
        z.index = [str(int(i)) for i in z.index]
        x, xtitle = MONTHS, None
    scale = (HEAT_ANOM.get(key, "RdBu_r") if anomaly else HEAT_SCALE.get(key, "Viridis"))
    sign = "+" if anomaly else ""
    fig = go.Figure(go.Heatmap(
        z=z.values, x=x, y=list(z.index), colorscale=scale, zmid=0 if anomaly else None, xgap=2, ygap=2,
        hoverongaps=False, colorbar=dict(title=unit),
        hovertemplate="%{y} · %{x}<br>%{z:" + sign + ".2f} " + unit + "<extra></extra>"))
    fig.update_layout(title=title + (" (difference from the usual for that month)" if anomaly else ""),
                      height=int(min(720, max(260, 90 + 32 * len(z)))), yaxis=dict(autorange="reversed", type="category"),
                      xaxis=dict(title=xtitle, dtick=1 if daily else None))
    return fig


def heatmap_tab(ts, units, power, p_unit, t_unit):
    us = units.startswith("US")
    daily = bool((ts.kind == "daily").any())
    anomaly = False
    if not daily:
        anomaly = st.radio("Show", ["Actual values", "Difference from the usual for that month"], horizontal=True,
                           key="heat_mode").startswith("Diff")
    unit_of = {"ppt": p_unit, "gdd": "°F-days" if us else "°C-days"}
    shown = 0
    for el in [e for e in LABEL if e in set(ts.element)]:
        sub = ts[ts.element == el].dropna(subset=["value"])
        if sub.empty or (anomaly and sub.year.nunique() < 2):
            continue
        st.plotly_chart(make_heat(sub, LABEL[el], unit_of.get(el, t_unit), el, daily, anomaly), use_container_width=True)
        shown += 1
    if anomaly and not shown:
        st.info("The difference view needs at least two years of monthly data. Pick a longer range or switch to actual values.")
    if power is not None:
        names = {"surface": "Surface soil wetness (top ~5 cm)", "root": "Root-zone soil wetness (top ~1 m)", "profile": "Full-profile soil wetness"}
        avail = [k for k in names if k in power]
        k = st.selectbox("Soil wetness depth (NASA POWER)", avail, format_func=names.get,
                         index=avail.index("root") if "root" in avail else 0, key="heat_wet")
        pdf = pd.DataFrame({"year": power.index.year, "month": power.index.month, "day": power.index.day,
                            "value": power[k].values}).dropna()
        if not pdf.empty and not (anomaly and pdf.year.nunique() < 2):
            st.plotly_chart(make_heat(pdf, names[k], "0 = dry, 1 = saturated", "wet", daily, anomaly), use_container_width=True)
    st.caption("Each cell is one month (or one day in Daily mode). Darker or redder means more. In the difference view, "
               "red is warmer and brown is drier than that month's average across your selected years, and blue/green is the opposite.")


# ------------------------------------------------------------ rotation advisor
# Starter crop traits for the contiguous US. APPROXIMATE screening values: edit this table for your region and crops.
#   gdd   = minimum "°C-days above 10 °C, Apr-Oct" (30 °C cap) the site should reach in a cool year
#   water = rough seasonal water need (mm)            ffd = minimum frost-free days (0 = overwinters/not limiting)
#   drain = acceptable SSURGO drainage ranks (0 excessively ... 6 very poorly drained)
#   n     = nitrogen role: fix / light / heavy        res = residue: low / med / high      root = deep / shallow
#   brk   = years to wait before the same crop returns   hot = Apr-Oct GDD above which this cool-season crop struggles
CROPS = [
    dict(name="Corn", fam="Poaceae (grasses)", gdd=1350, water=550, ffd=120, drain=(1, 4), n="heavy", res="high", root="deep", brk=1),
    dict(name="Soybeans", fam="Fabaceae (legumes)", gdd=1200, water=480, ffd=110, drain=(1, 5), n="fix", res="low", root="shallow", brk=2),
    dict(name="Winter wheat", fam="Poaceae (grasses)", gdd=600, water=400, ffd=0, drain=(1, 4), n="light", res="high", root="shallow", brk=1),
    dict(name="Spring wheat", fam="Poaceae (grasses)", gdd=650, water=350, ffd=90, drain=(1, 4), n="light", res="high", root="shallow", brk=1, hot=2200),
    dict(name="Barley", fam="Poaceae (grasses)", gdd=600, water=330, ffd=80, drain=(1, 3), n="light", res="med", root="shallow", brk=1, hot=2300),
    dict(name="Oats", fam="Poaceae (grasses)", gdd=650, water=330, ffd=80, drain=(1, 4), n="light", res="med", root="shallow", brk=1, hot=2300),
    dict(name="Sorghum", fam="Poaceae (grasses)", gdd=1350, water=400, ffd=115, drain=(0, 4), n="heavy", res="high", root="deep", brk=1),
    dict(name="Rice", fam="Poaceae (grasses)", gdd=1450, water=900, ffd=130, drain=(3, 6), n="heavy", res="high", root="shallow", brk=1),
    dict(name="Canola", fam="Brassicaceae (brassicas)", gdd=700, water=380, ffd=85, drain=(1, 3), n="heavy", res="low", root="deep", brk=3, hot=2000),
    dict(name="Sunflower", fam="Asteraceae", gdd=1200, water=450, ffd=105, drain=(1, 3), n="light", res="high", root="deep", brk=3),
    dict(name="Cotton", fam="Malvaceae", gdd=1400, water=650, ffd=160, drain=(1, 4), n="heavy", res="med", root="deep", brk=1, clay_max=55),
    dict(name="Dry beans", fam="Fabaceae (legumes)", gdd=1000, water=380, ffd=90, drain=(1, 3), n="fix", res="low", root="shallow", brk=2, clay_max=40),
    dict(name="Field peas", fam="Fabaceae (legumes)", gdd=550, water=300, ffd=70, drain=(1, 3), n="fix", res="low", root="shallow", brk=3, hot=2000),
    dict(name="Peanuts", fam="Fabaceae (legumes)", gdd=1500, water=500, ffd=140, drain=(0, 3), n="fix", res="low", root="shallow", brk=2, clay_max=25, sand_max=95),
    dict(name="Potatoes", fam="Solanaceae (nightshades)", gdd=900, water=500, ffd=100, drain=(1, 3), n="heavy", res="low", root="shallow", brk=3, clay_max=35, sand_max=90, hot=2600),
    dict(name="Tomatoes", fam="Solanaceae (nightshades)", gdd=1100, water=550, ffd=120, drain=(1, 3), n="heavy", res="low", root="shallow", brk=3, clay_max=45),
    dict(name="Sugar beets", fam="Amaranthaceae", gdd=1050, water=650, ffd=130, drain=(1, 3), n="heavy", res="low", root="deep", brk=3, clay_max=45),
]
CROP = {c["name"]: c for c in CROPS}
# Broadly grown crops used as the default pool; the advisor cannot see markets or local practice, so pick your own crops.
CORE = ["Corn", "Soybeans", "Winter wheat", "Spring wheat", "Barley", "Oats", "Sorghum", "Canola", "Sunflower", "Cotton", "Dry beans"]
DRAIN_RANK = {"excessively drained": 0, "somewhat excessively drained": 1, "well drained": 2, "moderately well drained": 3,
              "somewhat poorly drained": 4, "poorly drained": 5, "very poorly drained": 6}


def season_totals(rows, daily, months=range(4, 11)):
    """Apr-Oct totals per year, only for years whose season is (nearly) fully covered."""
    r = rows[rows.month.isin(list(months)) & rows.kind.isin(["monthly", "daily"])].dropna(subset=["value"])
    need = 200 if daily else len(list(months))
    return [float(g.value.sum()) for _, g in r.groupby("year") if len(g) >= need]


def frost_free_days(raw):
    t = raw[(raw.element == "tmin") & (raw.kind == "daily")].dropna(subset=["value"]).copy()
    if t.empty:
        return None
    t["date"] = pd.to_datetime(dict(year=t.year, month=t.month, day=t.day))
    res = []
    for y, g in t.groupby("year"):
        if not (g.date.min() <= pd.Timestamp(y, 2, 15) and g.date.max() >= pd.Timestamp(y, 11, 30)):
            continue  # needs a (nearly) full year of daily data
        fr = g[g.value <= 0]
        sp, fa = fr[fr.date.dt.month <= 6].date.max(), fr[fr.date.dt.month >= 8].date.min()
        sp = sp if pd.notna(sp) else pd.Timestamp(y, 1, 1)
        fa = fa if pd.notna(fa) else pd.Timestamp(y, 12, 31)
        res.append((fa - sp).days)
    return min(res) if res else None


def soil_summary(df):
    if df is None or df.empty or "cokey" not in df:
        return {}
    x = df.copy()
    for c in ["comppct_r", "hzdept_r", "hzdepb_r", "awc_r", "om_r", "claytotal_r", "sandtotal_r"]:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    comps, h = x.drop_duplicates("cokey"), x.dropna(subset=["hzdept_r", "hzdepb_r", "comppct_r"])
    out = {}
    dc = comps.dropna(subset=["drainagecl"]).groupby("drainagecl").comppct_r.sum()
    if not dc.empty:
        out["drain_name"] = dc.idxmax()
        out["drain_rank"] = DRAIN_RANK.get(str(dc.idxmax()).strip().lower())
    for key, col in [("clay", "claytotal_r"), ("sand", "sandtotal_r"), ("om", "om_r")]:
        v = _wavg(h, col, 30)
        out[key] = None if v != v else v
    a, tot = h.dropna(subset=["awc_r"]), comps.comppct_r.sum()
    if tot > 0 and not a.empty:
        ov = (a.hzdepb_r.clip(upper=100) - a.hzdept_r.clip(lower=0)).clip(lower=0)
        out["awc100"] = float((a.awc_r * ov * a.comppct_r).sum() / tot)
    return out


def build_site(raw, ssurgo_df, daily):
    site = {}
    gdd = compute_gdd_rows(raw, 10.0, daily)  # standard 10 °C base so it matches the crop table
    for key, vals in [("gdd", season_totals(gdd, daily)), ("rain", season_totals(raw[raw.element == "ppt"], daily))]:
        if vals:
            site[key + "_n"] = len(vals)
            site[key + "_cool"] = float(np.percentile(vals, 20)) if len(vals) >= 3 else float(min(vals))
            site[key + "_typ"] = float(np.median(vals))
    ffd = frost_free_days(raw)
    if ffd is not None:
        site["ffd"] = ffd
    site.update({k: v for k, v in soil_summary(ssurgo_df).items() if v is not None})
    return site


def apply_tile_drainage(site):
    """Field has drain tile: score wet soils as moderately well drained, which is what tiling achieves."""
    if site.get("drain_rank") is not None and site["drain_rank"] > DRAIN_RANK["moderately well drained"]:
        site["drain_rank"] = DRAIN_RANK["moderately well drained"]
        site["drain_name"] = f"{site['drain_name']} (tile-drained)"
    return site


def crop_fit(c, site, irrigated):
    """-> (score 0-1, hard-fail reason or None, notes)"""
    if site.get("drain_rank") is not None and not (c["drain"][0] <= site["drain_rank"] <= c["drain"][1]):
        return 0.0, f"{site['drain_name'].lower()} soil doesn't suit it", []
    if site.get("gdd_cool") is not None and site["gdd_cool"] < c["gdd"]:
        return 0.0, f"cool years give ~{site['gdd_cool']:.0f} °C-days, it needs {c['gdd']}", []
    if site.get("ffd") is not None and site["ffd"] < c["ffd"]:
        return 0.0, f"shortest frost-free season ({site['ffd']} d) is under its {c['ffd']} d", []
    score, notes = 1.0, []
    if site.get("gdd_cool") is not None and site["gdd_cool"] < (1.35 if c["gdd"] >= 1000 else 1.1) * c["gdd"]:
        score -= 0.15
        notes.append("heat is marginal in cool years")
    if c.get("hot") and site.get("gdd_typ") and site["gdd_typ"] > c["hot"]:
        score -= 0.35
        notes.append("site runs hot for this cool-season crop")
    if not irrigated and site.get("rain_cool") is not None and site["rain_cool"] < c["water"]:
        score -= min(0.5, (1 - site["rain_cool"] / c["water"]) * 0.8)
        notes.append(f"dry-year rain (~{site['rain_cool']:.0f} mm) is below its ~{c['water']} mm need")
    if site.get("clay") is not None and site["clay"] > c.get("clay_max", 60):
        score -= 0.2
        notes.append("soil is heavier than it prefers")
    if site.get("sand") is not None and site["sand"] > c.get("sand_max", 85):
        score -= 0.2
        notes.append("soil is sandier than it prefers")
    if site.get("awc100") is not None and site["awc100"] < 8 and c["water"] >= 500 and not irrigated:
        score -= 0.15
        notes.append("low water-holding soil for a thirsty crop")
    return max(score, 0.05), None, notes or ["comfortable fit"]


def trans_score(a, b):
    """How good is crop b after crop a?"""
    s = 0.0
    if a["name"] == b["name"]:
        s -= 1.5
    if a["fam"] == b["fam"]:
        s -= 1.0
    if a["n"] == "heavy" and b["n"] == "heavy":
        s -= 0.6
    if a["n"] == "fix" and b["n"] == "heavy":
        s += 0.6
    if a["n"] == "fix" and b["n"] == "light":
        s += 0.2
    if a["n"] == "heavy" and b["n"] == "fix":
        s += 0.5
    if a["n"] == "fix" and b["n"] == "fix":
        s -= 0.4
    if a["res"] == "high" and b["res"] == "low":
        s += 0.2
    if a["res"] == "low" and b["res"] == "low":
        s -= 0.2
    if a["root"] != b["root"]:
        s += 0.2
    return s


def search_rotations(pool, fit, L, last=None, top=5, beam=400):
    """Beam search for the best repeating L-year cycles. Returns [(score, [crop names])]."""
    by = {c["name"]: c for c in pool}

    def spacing_ok(hist, b):
        return all(not (prev == b and dist < by[b]["brk"] + 1) for dist, prev in enumerate(reversed(hist), start=1))

    start = [last["name"]] if last else []
    states = [(2.0 * fit[b] + (trans_score(last, by[b]) if last else 0.0), [b]) for b in by if spacing_ok(start, b)]
    for _ in range(L - 1):
        new = [(s + 2.0 * fit[b] + trans_score(by[seq[-1]], by[b]), seq + [b])
               for s, seq in states for b in by if spacing_ok(start + seq, b)]
        states = sorted(new, key=lambda t: -t[0])[:beam]
    done = {}
    for s, seq in states:
        if any(L - [i for i, x in enumerate(seq) if x == n][-1] + seq.index(n) < by[n]["brk"] + 1 for n in set(seq)):
            continue  # a crop would come back too soon when the cycle repeats
        s += trans_score(by[seq[-1]], by[seq[0]])
        s += len({by[x]["fam"] for x in seq}) / L + len(set(seq)) / L  # reward variety of families and of crops
        s += 1.0 if any(by[x]["n"] == "fix" for x in seq) else -0.5
        key = frozenset(seq)  # keep the best ordering of each crop set so the list stays varied
        if key not in done or s > done[key][0]:
            done[key] = (s, seq)
    return sorted(done.values(), key=lambda t: -t[0])[:top]


def explain_year(i, seq, last=None):
    name = seq[i]
    c = CROP[name]
    prev_name = seq[i - 1] if i > 0 else (last["name"] if last else seq[-1])
    p = CROP[prev_name]
    why = []
    if c["fam"] != p["fam"]:
        why.append(f"switches plant family ({p['fam'].split(' (')[0]} → {c['fam'].split(' (')[0]}), breaking pest and disease cycles")
    else:
        why.append("⚠️ same plant family as the previous crop")
    if c["n"] == "fix":
        why.append("fixes nitrogen for what follows")
    if p["n"] == "fix" and c["n"] in ("heavy", "light"):
        why.append(f"uses the nitrogen {prev_name.lower()} left behind")
    if p["n"] == "heavy" and c["n"] == "fix":
        why.append(f"rests the soil after heavy-feeding {prev_name.lower()}")
    if p["n"] == "heavy" and c["n"] == "heavy":
        why.append("⚠️ two heavy nitrogen users in a row; plan extra fertility")
    if p["res"] == "high" and c["res"] == "low":
        why.append("follows a high-residue crop (watch residue handling at planting)")
    if p["root"] != c["root"]:
        why.append(f"{c['root']}-rooted after {p['root']}-rooted, using different soil layers")
    return "; ".join(why)


def rotation_tab(d):
    raw, daily = d["df"], d["mode"] == "Daily"
    site = build_site(raw, d.get("ssurgo_df"), daily)

    st.subheader("Site profile from your data")
    if site.get("drain_rank") is not None and site["drain_rank"] > DRAIN_RANK["moderately well drained"]:
        if st.checkbox(f"The field has tile drainage (soil maps call it {site['drain_name'].lower()})", key="rot_tile",
                       help="Drain tile lets most crops grow on wet soils. When ticked, the soil is scored as moderately well drained."):
            apply_tile_drainage(site)
    c = st.columns(4)
    c[0].metric("Heat, cool year (Apr–Oct)", f"{site['gdd_cool']:.0f} °C-days" if "gdd_cool" in site else "n/a")
    c[1].metric("Rain, dry year (Apr–Oct)", f"{site['rain_cool']:.0f} mm" if "rain_cool" in site else "n/a")
    c[2].metric("Frost-free days (shortest year)", f"{site['ffd']}" if "ffd" in site else "n/a")
    c[3].metric("Soil drainage", site.get("drain_name", "n/a"))
    c = st.columns(4)
    c[0].metric("Clay (0–30 cm)", f"{site['clay']:.0f} %" if "clay" in site else "n/a")
    c[1].metric("Sand (0–30 cm)", f"{site['sand']:.0f} %" if "sand" in site else "n/a")
    c[2].metric("Organic matter", f"{site['om']:.1f} %" if "om" in site else "n/a")
    c[3].metric("Water held, top 1 m", f"{site['awc100']:.1f} cm" if "awc100" in site else "n/a")

    notes = []
    if "gdd_cool" not in site:
        notes.append("Heat: select min and max temperature and fetch **monthly** data covering at least one full April–October season (3–5 years is best).")
    elif min(site["gdd_n"], site.get("rain_n", 9)) < 3:
        notes.append(f"Only {site['gdd_n']} complete season(s) found, so the lowest year stands in for a 'cool/dry year'. Fetch 5+ years for a sturdier estimate.")
    if "rain_cool" not in site:
        notes.append("Rain: select precipitation to include water in the scoring.")
    if "ffd" not in site:
        notes.append("Frost-free days: only measured with Daily mode, min temperature, and a full calendar year; skipped here.")
    if "drain_name" not in site:
        notes.append("Soil: tick the SSURGO option so drainage and texture can be scored.")
    for n in notes:
        st.caption("ℹ️ " + n)

    irrigated = st.checkbox("Irrigation is available (waives the rainfall penalty)", key="rot_irrig")
    fits = {cr["name"]: crop_fit(cr, site, irrigated) for cr in CROPS}
    verdict = lambda f: "Not suited" if f == 0 else "Good fit" if f >= 0.8 else "Workable" if f >= 0.5 else "Marginal"
    tbl = pd.DataFrame([{"Crop": n, "Plant family": CROP[n]["fam"], "Fit": round(f * 100), "Verdict": verdict(f),
                         "Why": reason or "; ".join(nt)} for n, (f, reason, nt) in fits.items()]).sort_values("Fit", ascending=False)
    st.subheader("How well each crop suits the site")
    st.dataframe(tbl, hide_index=True, use_container_width=True,
                 column_config={"Fit": st.column_config.ProgressColumn("Fit", min_value=0, max_value=100, format="%d%%")})

    suited = [n for n, (f, _, _) in fits.items() if f > 0]
    if len(suited) < 2:
        st.warning("Fewer than two crops pass the hard limits (heat, frost-free days, drainage), so no rotation can be built. "
                   "Check the site numbers above, or edit the CROPS table at the top of the script for your region.")
        return
    sig = abs(hash((tuple(suited), round(site.get("gdd_cool", 0)), irrigated))) % 10**8
    st.subheader("Suggested rotations")
    a, b, e = st.columns([1, 2, 1])
    L = a.slider("Cycle length (years)", 2, 6, 4, key="rot_len")
    pool_names = b.multiselect("Crops to consider", suited, default=([n for n in tbl.Crop if n in suited and n in CORE] or [n for n in tbl.Crop if n in suited])[:8],
                              key=f"rot_pool_{sig}", help="Choose crops you can actually grow and sell; the advisor can't see markets.")
    last_name = e.selectbox("Last crop grown", ["(unknown)"] + [c["name"] for c in CROPS], key="rot_last")
    last = CROP.get(last_name)
    pool = [CROP[n] for n in pool_names]
    if len(pool) < 2:
        st.info("Pick at least two crops.")
        return
    res = search_rotations(pool, {n: fits[n][0] for n in pool_names}, L, last)
    if not res:
        st.info("No valid sequence with these crops and this cycle length (some crops need long breaks before returning). "
                "Add crops or lengthen the cycle.")
        return
    summary = pd.DataFrame([{"Rank": i + 1, "Rotation": " → ".join(seq), "Score": round(s, 1),
                             "Average fit": f"{round(100 * np.mean([fits[x][0] for x in seq]))}%",
                             "Includes a legume": "yes" if any(CROP[x]["n"] == "fix" for x in seq) else "no"}
                            for i, (s, seq) in enumerate(res)])
    st.dataframe(summary, hide_index=True, use_container_width=True)
    for i, (s, seq) in enumerate(res):
        with st.expander(f"#{i + 1}: {' → '.join(seq)} (then repeats)", expanded=(i == 0)):
            st.dataframe(pd.DataFrame([{"Year": j + 1, "Crop": x, "Fit": f"{round(100 * fits[x][0])}%",
                                        "Why it works here": explain_year(j, seq, last)} for j, x in enumerate(seq)]),
                         hide_index=True, use_container_width=True)
    with st.expander("Crop traits used by the advisor"):
        st.dataframe(pd.DataFrame(CROPS).drop(columns=["drain"]).assign(drain=[f"{c['drain'][0]}–{c['drain'][1]}" for c in CROPS]),
                     hide_index=True, use_container_width=True)
    info = sources_section(res, d)
    plan_section(res, fits, last, irrigated, info)
    st.caption("Screening tool, not agronomic advice. Crop traits are approximate starter values (editable in the CROPS table at the top "
               "of this section). The advisor does not see field history, pests and disease, soil pH and nutrients, prices, or equipment, "
               "so check any rotation with a local extension agent.")


# ------------------------------------------------------- outside sources (SerpApi)
def _md(text):
    """Make search text safe for st.markdown (no broken links, no stray LaTeX)."""
    return (text or "").replace("[", "(").replace("]", ")").replace("$", "\\$").replace("\n", " ")


def _links(items):
    if not items:
        st.caption("Nothing found.")
        return
    for it in items:
        meta = " · ".join(_md(x) for x in (it.get("site") or it.get("domain"), it.get("source"), it.get("date")) if x)
        st.markdown(f"- [{_md(it['title'])}]({it['link']})" + (f"  \n  *{meta}*" if meta else "")
                    + (f"  \n  {_md(it['snippet'])}" if it.get("snippet") else ""))


def cycle_pairs(seq):
    """Each (previous, next) crop change in a repeating cycle, in order, without repeats or same-crop pairs."""
    return [p for p in dict.fromkeys(zip(seq, seq[1:] + seq[:1])) if p[0] != p[1]]


def sources_section(res, d):
    st.subheader("Check it against outside sources")
    if serp_search is None or not serp_search.have_key():
        st.caption("ℹ️ Add a SerpApi key (see the project README) to look up extension guides, research, and "
                   "cover-crop advice for these rotations.")
        return None
    labels = [" → ".join(seq) for _, seq in res]
    pick = st.selectbox("Rotation to look up", range(len(res)), format_func=lambda i: f"#{i + 1}: {labels[i]}", key="src_pick")
    seq = res[pick][1]

    kind, c = d["geom"]
    lon, lat = c if kind == "point" else np.array(c).mean(axis=0)
    try:
        state, county = place_name(round(float(lat), 3), round(float(lon), 3))
    except Exception:
        state = county = None
    region = state or ""
    pairs = cycle_pairs(seq)
    cover_pairs = [p for p in pairs if p[1] != "Winter wheat"]  # wheat is sown right after harvest: no cover-crop window

    crops = list(dict.fromkeys(seq))
    cols = st.columns(4)
    want_plant = cols[3].checkbox("Planting dates", True, key="src_plant",
                                  help="Local planting window for each crop, used in the plan below (1 search per crop).")
    want_ext = cols[0].checkbox("Extension guides", True, key="src_ext",
                                help="University extension pages about this crop mix in your state (1 search).")
    want_sch = cols[1].checkbox("Research on each crop change", True, key="src_sch",
                                help="Google Scholar studies on how each crop does after the one before it (1 search per change).")
    want_cov = cols[2].checkbox("Cover crops between crops", True, key="src_cov",
                                help="Extension advice on a cover crop for each gap (1 search per gap; skipped before winter wheat).")
    cost = want_ext + want_sch * len(pairs) + want_cov * len(cover_pairs) + want_plant * len(crops)
    st.caption(f"Searching for: **{', '.join(x for x in (county, state) if x) or 'location unknown, so searching nationally'}**. "
               "Results are saved for this session, so looking again costs nothing.")

    key = (tuple(seq), region)
    store = st.session_state.setdefault("src_results", {})
    if st.button(f"Search the web ({cost} SerpApi search{'es' if cost != 1 else ''})", disabled=cost == 0, key="src_go"):
        found = {}
        with st.spinner("Searching…"):
            try:
                if want_plant:  # first, since the plan depends on it most
                    found["plant"] = {x: serp_search.search(f"when to plant {x.lower()} {region} planting date extension "
                                                            "-garden -vegetable -almanac", num=8)
                                      for x in crops}
                if want_ext:
                    found["ext"] = serp_search.search(f"{' '.join(x.lower() for x in crops)} crop rotation {region} extension site:edu", num=5)
                if want_sch:
                    found["sch"] = {p: serp_search.search(f"{p[1].lower()} following {p[0].lower()} rotation yield",
                                                          num=3, engine="google_scholar") for p in pairs}
                if want_cov:
                    found["cov"] = {p: serp_search.search(f"cover crop after {p[0].lower()} before {p[1].lower()} {region} site:edu",
                                                          num=3) for p in cover_pairs}
            except Exception as e:
                st.error(f"Search stopped: {e}")
        old = store.get(key, {})
        for k, v in found.items():  # keep anything found, even if a later search failed
            old[k] = {**old.get(k, {}), **v} if isinstance(v, dict) else v
        store[key] = old

    got = store.get(key)
    info = dict(seq=seq, got=got or {}, state=state, county=county)
    if not got:
        return info
    if "plant" in got:
        st.markdown(f"**Planting dates{' in ' + state if state else ''}**")
        for x, items in got["plant"].items():
            w = planting_window(items, CROP[x])
            with st.expander(f"{x}: {w[0] if w else 'no clear date found'}"):
                _links(items)
    if "ext" in got:
        st.markdown(f"**Extension guides{' for ' + state if state else ''}**")
        _links(got["ext"])
    if "sch" in got:
        st.markdown("**Research on each crop change**")
        for (a, b), items in got["sch"].items():
            with st.expander(f"{b} after {a.lower()}"):
                _links(items)
    if "cov" in got:
        st.markdown("**Cover crops between crops**")
        for (a, b), items in got["cov"].items():
            with st.expander(f"Between {a.lower()} and {b.lower()}"):
                _links(items)
    st.caption("These are web search results, not checked by the advisor. Read the source before acting on it.")
    return info


# ------------------------------------------------------------ final plan for the farmer
COVER_NAMES = ["cereal rye", "winter rye", "annual ryegrass", "triticale", "oats", "barley", "hairy vetch", "crimson clover",
               "red clover", "white clover", "berseem clover", "field peas", "austrian winter pea", "radish", "turnip",
               "sorghum-sudangrass", "cowpea", "sunn hemp", "buckwheat", "mustard"]
EARLY_HARVEST = {"Winter wheat", "Spring wheat", "Barley", "Oats", "Field peas", "Canola"}  # leaves a long late-summer window


def default_cover(a, b):
    """Common starting-point cover crop for the gap between crop a's harvest and crop b's planting."""
    if b == "Winter wheat":
        return None, "skip the cover crop. Winter wheat goes in soon after harvest and covers the ground over winter"
    if a in EARLY_HARVEST and CROP[b]["n"] == "heavy":
        return "red clover or hairy vetch", f"a legume has time to grow after the early {a.lower()} harvest and adds nitrogen for {b.lower()}"
    if CROP[b]["n"] == "fix":
        return "cereal rye", f"a grass soaks up leftover nitrogen; end it 2-3 weeks before planting {b.lower()}"
    if CROP[b]["n"] == "heavy":
        return "cereal rye, or a rye + hairy vetch mix", f"protects the soil over winter; the vetch adds some nitrogen for {b.lower()}"
    return "cereal rye or oats", "protects the soil over winter and builds organic matter"


_SOCIAL = ("facebook.com", "reddit.com", "quora.com", "youtube.com", "tiktok.com", "instagram.com", "pinterest.com",
           "x.com", "twitter.com", "almanac.com")
_US_TLDS = (".edu", ".gov", ".org", ".com", ".net", ".us", ".coop", ".info")


def trusted_source(item):
    """Is this search result fit to base farm advice on? Rules out social media, home-garden/almanac pages, blogs
    and foreign-country sites (often unrelated or spam for a US farm). Results without a resolved domain are allowed."""
    domain = (item.get("domain") or "").lower()
    text = f"{item.get('title') or ''} {item.get('site') or ''} {domain}".lower()
    if any(domain == s or domain.endswith("." + s) for s in _SOCIAL):
        return False
    if re.search(r"\b(facebook|reddit|quora|youtube|tiktok|instagram|pinterest)\b", (item.get("site") or "").lower()):
        return False  # also catches links that couldn't be resolved to a real domain
    if re.search(r"garden|vegetable|almanac|backyard|homestead", text) or "blog" in domain:
        return False
    return not domain or domain.endswith(_US_TLDS)


def official_source(item):
    """University, government, extension or USDA site: preferred over other trusted sources."""
    domain = (item.get("domain") or "").lower()
    return trusted_source(item) and (domain.endswith((".edu", ".gov")) or bool(re.search(r"extension|usda", domain)))


def prefer_official(items, pick):
    """pick(official results) if that finds something, else pick(other trusted results)."""
    items = [i for i in items or [] if trusted_source(i)]
    return pick([i for i in items if official_source(i)]) or pick(items)


def source_name(item):
    return item.get("site") or item.get("domain") or "web search result"


def mentioned_cover(items):
    """Cover crop named most often in the search results for one gap, or None."""
    text = " ".join(f"{i.get('title') or ''} {i.get('snippet') or ''}" for i in items or [] if trusted_source(i)).lower()
    counts = {n: text.count(n) for n in COVER_NAMES if text.count(n)}
    return max(counts, key=counts.get) if counts else None


# Month names in any case, except a bare lowercase "may" (almost always the verb) unless a day number follows.
_MONTH = (r"(?:(?i:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?"
          r"|nov(?:ember)?|dec(?:ember)?)|May|(?i:may)(?=\s+\d))\b\.?")
_POINT = (rf"(?:(?i:early|mid|late)[- ]?)?{_MONTH}(?!\s+\d{{1,2}},?\s*\d{{4}})"  # skip 'June 5, 2009' and 'June 2009'
          rf"(?:\s+\d{{1,2}}(?:st|nd|rd|th)?\b)?(?!,?\s*\d{{4}})")                  # (report dates, not advice)
_RANGE = re.compile(rf"\b({_POINT})\s*(?:to|through|until|and|or|-|–|—)\s*({_POINT})")
_MNUM = {m: i + 1 for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}
_MNAME = ["", "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]


def _month_of(point):
    return _MNUM[re.search(_MONTH, point).group(0)[:3].lower()]


def _tidy(point):
    p = re.sub(r"\s+", " ", point.strip()).replace("mid ", "mid-")
    return re.sub(_MONTH, lambda m: _MNAME[_MNUM[m.group(0)[:3].lower()]], p)  # 'sept.' -> 'September'


def planting_window(items, crop):
    """Planting window quoted in search snippets -> (text, source item) or None.

    Uses university/government/extension results when they give a date, other trusted sources otherwise.
    Prefers an explicit range ('late April to mid-May'); otherwise falls back to the months most often named
    ('around April to May'). Only sentences about planting/seeding count, and months must fit the crop's season
    (Aug-Dec for overwintering crops, Feb-Jul otherwise), so harvest dates and report dates are ignored.
    """
    return prefer_official(items, lambda its: _window(its, crop))


def _window(items, crop):
    fall = crop["ffd"] == 0
    in_season = (lambda m: 8 <= m <= 12) if fall else (lambda m: 2 <= m <= 7)
    votes, first, months = {}, {}, {}
    for it in items:
        for sent in re.split(r"(?<=[.!?])\s+|\s?\.\.\.\s?", it.get("snippet") or ""):
            if not re.search(r"plant|seed|sow", sent, re.I):
                continue
            for a, b in _RANGE.findall(sent):
                ma, mb = _month_of(a), _month_of(b)
                if in_season(ma) and in_season(mb) and ma <= mb:
                    text = f"{_tidy(a)} to {_tidy(b)}"
                    votes[text.lower()] = votes.get(text.lower(), 0) + 1
                    first.setdefault(text.lower(), (text, it))
            for m in {_month_of(p) for p in re.findall(_POINT, sent)}:
                if in_season(m):
                    n, src = months.get(m, (0, it))
                    months[m] = (n + 1, src)
    if votes:
        return first[max(votes, key=votes.get)]
    if not months:
        return None
    top = max(n for n, _ in months.values())
    keep = sorted(m for m, (n, _) in months.items() if n * 2 >= top)  # months named at least half as often as the leader
    if keep[-1] - keep[0] > 2:  # sources disagree too much to give one window
        return None
    src = months[max(keep, key=lambda m: months[m][0])][1]
    text = _MNAME[keep[0]] + ("" if len(keep) == 1 else f" to {_MNAME[keep[-1]]}")
    return f"around {text}", src


def fertility_note(c, p):
    if c["n"] == "fix":
        return "Usually needs little or no nitrogen fertilizer."
    if c["n"] == "heavy" and p["n"] == "fix":
        return f"Count the nitrogen {p['name'].lower()} left behind; a soil test will likely show you can apply less."
    if c["n"] == "heavy" and p["n"] == "heavy":
        return f"Plan full nitrogen; {p['name'].lower()} used most of it up."
    if c["n"] == "heavy":
        return "Heavy nitrogen user: plan fertilizer from a soil test."
    return "Moderate fertility needs: fertilize to a soil test."


def plan_section(res, fits, last, irrigated, info):
    seq = info["seq"] if info else res[0][1]
    got = info["got"] if info else {}
    state, county = (info["state"], info["county"]) if info else (None, None)
    avg = round(100 * np.mean([fits[x][0] for x in seq]))
    legume = next((x for x in seq if CROP[x]["n"] == "fix"), None)

    lines = [f"**Grow {' → '.join(seq)}, then start over.** Average fit for your site: {avg}%."
             + (f" Growing {legume.lower()} feeds nitrogen to the next crop." if legume else " There is no legume, so plan on more nitrogen fertilizer.")]
    if last:
        lines.append(f"Coming off {last['name'].lower()}, start with **{seq[0]}**: " + explain_year(0, seq, last).split(";")[0] + ".")
    lines.append("")
    for i, crop in enumerate(seq):
        prev = CROP[seq[i - 1] if i > 0 else (last["name"] if last else seq[-1])]
        nxt = seq[(i + 1) % len(seq)]
        lines.append(f"**Year {i + 1}: plant {crop}**")
        if "plant" in got:
            w = planting_window(got["plant"].get(crop), CROP[crop])
            lines.append(f"- When to plant: **{w[0]}** (source: [{_md(source_name(w[1]))}]({w[1]['link']}))." if w else
                         "- When to plant: local sources didn't give a clear date; ask your extension office.")
        lines.append(f"- Fertility: {fertility_note(CROP[crop], prev)}")
        watch = [n for n in fits[crop][2] if n != "comfortable fit"]
        if watch:
            tip = " Choose a drought-tolerant variety or plan irrigation." if any("rain" in w for w in watch) and not irrigated else ""
            lines.append(f"- Watch: {'; '.join(watch)}.{tip}")
        if crop == nxt:
            continue
        cover, why = default_cover(crop, nxt)
        local = mentioned_cover(got.get("cov", {}).get((crop, nxt)))
        if cover is None:
            lines.append(f"- After harvest: {why}.")
        elif local and local not in cover:
            lines.append(f"- After harvest: sow a cover crop before {nxt.lower()}. Local sources mention **{local}**; "
                         f"{cover} is the usual default ({why}).")
        else:
            lines.append(f"- After harvest: sow **{local or cover}** as a cover crop before {nxt.lower()} ({why})."
                         + (" Local sources agree." if local else ""))
        lines.append("")

    where = ", ".join(x for x in (county, state) if x)
    ext = prefer_official(got.get("ext"), lambda its: its[0] if its else None)
    w1 = planting_window(got.get("plant", {}).get(seq[0]), CROP[seq[0]])
    steps = [
        f"Get a soil test (pH, phosphorus, potassium) before planting {seq[0].lower()}"
        + (f" ({w1[0]})" if w1 else "") + ". The advisor can't see soil nutrients.",
        f"Line up cover crop seed before the year 1 harvest so it can be sown right away.",
        f"Check the plan with your {'county ' if county else ''}extension office{' in ' + where if where else ''}"
        + (f", starting with [{_md(ext['title'])}]({ext['link']}) ({_md(source_name(ext))})." if ext else "."),
        "Before committing acres, check that you have a buyer and the equipment for every crop in the cycle.",
    ]
    if not info or not got:
        steps.append("For local planting dates, cover crops, and research backing, run the outside-source search above.")
    lines += ["**What to do next**"] + [f"{i + 1}. {x}" for i, x in enumerate(steps)]

    st.subheader("Your plan")
    with st.container(border=True):
        st.markdown("\n".join(lines))
    st.download_button("Download this plan", "# Crop rotation plan\n\n" + "\n".join(lines).replace("**", ""),
                       file_name="rotation_plan.md", mime="text/markdown", key="plan_dl")


class SingleSelection(MacroElement):
    _template = Template("""
        {% macro script(this, kwargs) %}
        {{ this._parent.get_name() }}.on('draw:created', function (e) {
            {{ this._parent.get_name() }}.eachLayer(function (g) {
                if (g instanceof L.FeatureGroup) {
                    var old = [];
                    g.eachLayer(function (c) { if (c !== e.layer) old.push(c); });
                    old.forEach(function (c) { g.removeLayer(c); });
                }
            });
        });
        {% endmacro %}
    """)


# ------------------------------------------------------------------ NASA POWER
def fetch_power(geom, d0, d1):
    kind, c = geom
    lon, lat = c if kind == "point" else tuple(np.array(c).mean(axis=0))
    sd = max(d0, dt.date(1981, 1, 1))
    ed = min(d1, dt.date.today() - dt.timedelta(days=2))
    if sd > ed:
        raise RuntimeError("NASA POWER has no data for this date range.")
    params = dict(
        parameters="GWETTOP,GWETROOT,GWETPROF",
        community="AG",
        longitude=round(lon, 4),
        latitude=round(lat, 4),
        start=sd.strftime("%Y%m%d"),
        end=ed.strftime("%Y%m%d"),
        format="JSON"
    )
    cache = CACHE / ("power_" + hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()[:16] + ".json")
    if cache.exists():
        js = json.loads(cache.read_text())
    else:
        r = requests.get("https://power.larc.nasa.gov/api/temporal/daily/point", params=params, timeout=120)
        if r.status_code != 200:
            raise RuntimeError(f"NASA POWER returned HTTP {r.status_code}: {r.text[:300]}")
        js = r.json()
        cache.write_text(json.dumps(js))
    par = js["properties"]["parameter"]
    cols = {name: pd.Series(par[code]) for name, code in
            [("surface", "GWETTOP"), ("root", "GWETROOT"), ("profile", "GWETPROF")] if code in par}
    df = pd.DataFrame(cols)
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df.index.name = "date"
    df = df.where(df > -900).dropna(how="all")
    if df.empty:
        raise RuntimeError("NASA POWER returned no valid values for this location/period.")
    return df.sort_index()


def power_tab(pw, ts, elements, daily, p_unit):
    names = [("surface", "Surface (top ~5 cm)", "#8d6e63"), ("root", "Root zone (top ~1 m)", "#2e7d32"),
             ("profile", "Full profile", "#1565c0")]
    names = [n for n in names if n[0] in pw]
    cols = st.columns(len(names))
    for col, (k, lab, _) in zip(cols, names):
        col.metric(f"{lab} · average", f"{pw[k].mean():.2f}")
    f = go.Figure()
    for k, lab, colr in names:
        f.add_scatter(x=pw.index, y=pw[k], name=lab, line_color=colr)
    f.update_layout(title="NASA POWER soil wetness (0 = dry, 1 = saturated)", yaxis_title="fraction of saturation",
                    yaxis_range=[0, 1], height=350)
    st.plotly_chart(f, use_container_width=True)


# ------------------------------------------------------------------------- UI
st.set_page_config(page_title="PRISM Climate & NRCS Soil Explorer", layout="wide")
st.title("🌎 PRISM Climate & NRCS Soil Explorer")
st.caption("Search a place → draw your land on the map → get PRISM climate, GDD, and USDA SSURGO/STATSGO2 soil properties.")

ss = st.session_state
ss.setdefault("center", [39.8, -98.6])
ss.setdefault("zoom", 4)
ss.setdefault("map_v", 0)
ss.setdefault("results", [])
ss.setdefault("data", None)

with st.sidebar:
    st.header("1 · Find your land")
    q = st.text_input("Search address, town, park, ranch…")
    if st.button("Search", use_container_width=True) and q:
        try:
            ss.results = geocode(q)
            if not ss.results:
                st.warning("No matches.")
        except Exception as e:
            st.error(f"Search failed: {e}")
    if ss.results:
        idx = st.selectbox("Matches", range(len(ss.results)), format_func=lambda i: ss.results[i]["display_name"][:70])
        if st.button("Go to location", use_container_width=True):
            r = ss.results[idx]
            ss.center = [float(r["lat"]), float(r["lon"])]
            ss.zoom = 14
            ss.map_v += 1
            st.rerun()

    st.header("2 · Data options")
    elements = st.multiselect("Variables", list(LABEL), default=["ppt", "tmin", "tmax", "gdd"], format_func=LABEL.get)
    mode = st.radio("Time step", ["Monthly", "Daily"], horizontal=True)
    today = dt.date.today()
    if mode == "Daily":
        default = (today - dt.timedelta(days=37), today - dt.timedelta(days=7))
    else:
        default = (dt.date(today.year - 5, 1, 1), dt.date(today.year - 1, 12, 31))
    rng = st.date_input("Date range (start → end)", value=default, min_value=dt.date(1981, 1, 1),
                        max_value=today, key=f"range_{mode}")
    valid = isinstance(rng, (tuple, list)) and len(rng) == 2
    d0, d1 = rng if valid else (None, None)
    units = st.radio("Units", ["Metric (mm, °C)", "US (in, °F)"])

    st.header("3 · GDD & Agricultural Settings")
    gdd_base = st.number_input("GDD Base Temperature (" + ("°F" if units.startswith("US") else "°C") + ")",
                               value=50.0 if units.startswith("US") else 10.0, step=1.0)

    st.header("4 · Extra layers")
    want_ssurgo = st.checkbox("Add USDA NRCS SSURGO/STATSGO2 Soil Data", value=True)
    want_power = st.checkbox("Add NASA POWER soil wetness")

    tasks = build_tasks(elements, mode, d0, d1) if valid else []
    too_many = len(tasks) > MAX_FILES
    if too_many:
        st.error(f"Too many files (max {MAX_FILES}). Shorten the range or pick fewer variables.")

# Map
m = folium.Map(location=ss.center, zoom_start=ss.zoom, tiles=None)
folium.TileLayer(
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri World Imagery", name="Satellite",
).add_to(m)
folium.TileLayer("OpenStreetMap", name="Street map").add_to(m)
Draw(
    export=False,
    draw_options=dict(polyline=False, circle=False, circlemarker=False, marker=True, polygon=True, rectangle=True),
    edit_options=dict(edit=True),
).add_to(m)
SingleSelection().add_to(m)
add_reference_overlays(m)  # off by default; switch on in the layer control
folium.LayerControl().add_to(m)

st.subheader("Draw a rectangle or polygon around your land (or drop a pin)")
out = st_folium(m, height=520, use_container_width=True, key=f"map-{ss.map_v}", returned_objects=["all_drawings"])
geom = parse_geom(out.get("all_drawings") if out else None)

if geom is None:
    st.info("Use the toolbar on the left of the map to draw your selection.")
else:
    if geom[0] == "poly":
        st.success(f"Selection: polygon, ≈ {approx_area_km2(geom[1]):.2f} km²")
    else:
        st.success(f"Selection: point at {geom[1][1]:.4f}, {geom[1][0]:.4f}")
    if st.button("📊 Fetch data & soil properties", type="primary", disabled=not elements or not tasks or too_many):
        df, missing, ncells = run_fetch(geom, tasks)
        n_limit = sum(v.startswith("LIMIT") for v in ERRORS.values())
        if n_limit:
            st.warning(f"{n_limit} PRISM grids were skipped because PRISM only lets each file be downloaded twice "
                       "a day. They will download normally tomorrow (Pacific time), and the app saves every "
                       "grid it gets, so this won't recur for the same months.")
        if df.empty or df.value.isna().all():
            st.error("No data returned. PRISM covers the continental US only.")
            ss.data = None
        else:
            if {"tmin", "tmax"} <= set(df.element):
                base_c = (gdd_base - 32) * 5 / 9 if units.startswith("US") else float(gdd_base)
                gdd_rows = compute_gdd_rows(df, base_c, mode == "Daily")
                if not gdd_rows.empty:
                    df = pd.concat([df, gdd_rows], ignore_index=True)

            ssurgo_df = None
            if want_ssurgo:
                with st.spinner("Querying USDA NRCS SSURGO / STATSGO2 Soil Data Access..."):
                    try:
                        ssurgo_df = fetch_ssurgo_data(geom)
                    except Exception as e:
                        st.warning(f"Could not load soil data: {e}")

            power = None
            if want_power:
                with st.spinner("Fetching NASA POWER soil wetness…"):
                    try:
                        power = fetch_power(geom, d0, d1)
                    except Exception as e:
                        st.warning(f"NASA POWER: {e}")

            ss.data = dict(df=df, missing=missing, ncells=ncells, mode=mode, d0=d0, d1=d1, geom=geom,
                           ssurgo_df=ssurgo_df, power=power)

if ss.data and geom is not None and ss.data.get("geom") != geom:
    ss.data = None
if ss.data:
    d = ss.data
    st.markdown(f"**Grid cells averaged:** {d['ncells']}" + (f"  ·  ⚠️ {d['missing']} grids unavailable" if d["missing"] else ""))
    df = convert(d["df"], units)
    p_unit, t_unit = ("mm", "°C") if units.startswith("Metric") else ("in", "°F")
    daily = d["mode"] == "Daily"
    step = "Daily" if daily else "Monthly"
    ts = df[df.kind != "normal"].copy()
    ts["date"] = pd.to_datetime(dict(year=ts.year, month=ts.month, day=ts.day))
    ts = ts.sort_values("date")

    labels = [f"{step} time series", "Summary" if daily else "By calendar month", "Annual summary", "Data"]
    extras = [("Data map", lambda: map_tab(d, units)),
              ("Heatmaps", lambda: heatmap_tab(ts, units, d.get("power"), p_unit, t_unit))]
    if d.get("ssurgo_df") is not None:
        extras.append(("Soil Survey (SSURGO)", lambda: ssurgo_tab(d["ssurgo_df"])))
    if d.get("power") is not None:
        extras.append(("NASA POWER soil wetness", lambda: power_tab(d["power"], ts, elements, daily, p_unit)))

    extras.append(("Rotation advisor", lambda: rotation_tab(d)))
    tabs = st.tabs(labels + [e[0] for e in extras])
    t1, t2, t3, t4 = tabs[:4]

    with t1:
        for el in elements:
            sub = ts[ts.element == el]
            if sub.empty:
                continue
            f = go.Figure(go.Scatter(x=sub.date, y=sub.value, mode="lines+markers", name=LABEL[el], line_color=COLOR.get(el, "#333")))
            f.update_layout(title=f"{step} {LABEL[el]}", yaxis_title=f"{p_unit if el=='ppt' else ('°F-days' if el=='gdd' and not units.startswith('Metric') else ('°C-days' if el=='gdd' else t_unit))}", height=320)
            st.plotly_chart(f, use_container_width=True)

    with t2:
        if daily:
            items = []
            if "ppt" in elements:
                s = ts[ts.element == "ppt"].dropna(subset=["value"]).set_index("date").value
                if len(s):
                    items += [("Total precipitation", f"{s.sum():.1f} {p_unit}"), ("Wettest day", f"{s.max():.1f} {p_unit} on {s.idxmax():%b %d}")]
            cols = st.columns(min(len(items), 3)) if items else []
            for k, (lab, val) in enumerate(items):
                cols[k % len(cols)].metric(lab, val)
        else:
            avg = ts.groupby(["element", "month"]).value.mean().reset_index()
            for el in elements:
                sub = avg[avg.element == el]
                if sub.empty:
                    continue
                f = go.Figure(go.Bar(x=[MONTHS[m - 1] for m in sub.month], y=sub.value, marker_color=COLOR.get(el, "#333")))
                f.update_layout(title=f"Average monthly {LABEL[el]}", height=320)
                st.plotly_chart(f, use_container_width=True)

    with t3:
        months_have = ts.dropna(subset=["value"]).groupby(["element", "year"]).month.nunique()
        full = ts.dropna(subset=["value"]).groupby(["element", "year"]).filter(lambda g: g.month.nunique() == 12)
        if daily:
            st.info("Annual summary is available in Monthly mode.")
        elif full.empty:
            gaps = ", ".join(f"{LABEL.get(el, el)} {y}: {n}/12 months" for (el, y), n in months_have.items() if n < 12)
            st.info("No complete calendar years in the selected range. An annual total needs all 12 months "
                    "(January–December) of a year. Pick a range that covers whole years, or click Fetch again "
                    "to retry grids that failed to download." + (f"\n\nMissing data: {gaps}" if gaps else ""))
        else:
            for el in elements:
                sub = full[full.element == el].groupby("year").value.sum() if el in ["ppt", "gdd"] else full[full.element == el].groupby("year").value.mean()
                sub = sub.reset_index()
                if sub.empty:
                    continue
                f = go.Figure(go.Bar(x=sub.year, y=sub.value, marker_color=COLOR.get(el, "#333")))
                f.update_layout(title=f"Annual {LABEL[el]}", height=300, xaxis=dict(dtick=1))
                st.plotly_chart(f, use_container_width=True)

    with t4:
        wide = df.pivot_table(index=["kind", "year", "month", "day"], columns="element", values="value").reset_index()
        st.dataframe(wide, use_container_width=True)
        st.download_button("Download CSV", wide.to_csv(index=False), "prism_selection.csv", "text/csv")

    for k, (_, render) in enumerate(extras, start=4):
        with tabs[k]:
            render()