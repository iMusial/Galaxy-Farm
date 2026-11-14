"""
PRISM Climate Explorer
----------------------
Search for a place, draw a rectangle/polygon (or drop a pin) on a satellite map,
and pull PRISM monthly climate data for that land. Results are averaged over the
4 km PRISM grid cells inside your selection and shown as interactive graphs.

Install:   pip install streamlit streamlit-folium folium requests numpy pandas plotly
Run:       streamlit run prism_explorer.py

Data: PRISM Climate Group, Oregon State University (https://prism.oregonstate.edu)
Web service: https://services.nacse.org/prism/data/get/us/4km/...   (continental US only)
Downloaded zips are cached in ./prism_cache so each file is only fetched once.
"""
import datetime as dt
import io
import re
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from folium.plugins import Draw
from streamlit_folium import st_folium

BASE = "https://services.nacse.org/prism/data/get/us/4km"
ERRORS = {}  # url -> last failure reason (shown in the UI if nothing downloads)
CACHE = Path("prism_cache")
CACHE.mkdir(exist_ok=True)
LABEL = {"ppt": "Precipitation", "tmin": "Min temp", "tmax": "Max temp", "tmean": "Mean temp"}
COLOR = {"ppt": "#2b7bba", "tmin": "#4a90d9", "tmax": "#d9534f", "tmean": "#e6a23c"}
MAX_FILES = 1500  # safety cap on downloads per request
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# ----------------------------------------------------------------- PRISM access
def fetch_zip(url):
    """Download a PRISM zip (with on-disk cache). Returns bytes, or None (reason stored in ERRORS)."""
    fp = CACHE / (re.sub(r"[^A-Za-z0-9]+", "_", url.replace(BASE, "")) + ".zip")
    if fp.exists():
        return fp.read_bytes()
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=90)
        except requests.RequestException as e:
            ERRORS[url] = f"Network error: {e}"
            time.sleep(2 * (attempt + 1))
            continue
        if r.status_code == 200 and r.content[:2] == b"PK":
            fp.write_bytes(r.content)
            return r.content
        ERRORS[url] = f"HTTP {r.status_code}: {r.text[:200]!r}"
        if r.status_code < 500:  # 4xx (not published / download limit) - retrying won't help
            return None
        time.sleep(2 * (attempt + 1))
    return None


def read_bil(zip_bytes):
    """Parse the .bil grid + .hdr inside a PRISM zip (no GDAL needed)."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        names = z.namelist()
        bil = next(n for n in names if n.lower().endswith(".bil"))
        hdr = next((n for n in names if n.lower().endswith(".hdr")), None)
        h = {}
        if hdr:
            h = dict(l.split(None, 1) for l in z.read(hdr).decode().splitlines() if len(l.split()) > 1)
            h = {k.upper(): v.strip() for k, v in h.items()}
        dtype = "<f4" if h.get("BYTEORDER", "I") == "I" else ">f4"
        # defaults = standard 4 km CONUS grid, used only if the header lacks a value
        arr = np.frombuffer(z.read(bil), dtype=dtype).reshape(int(h.get("NROWS", 621)), int(h.get("NCOLS", 1405))).astype(float)
    arr[arr <= float(h.get("NODATA", -9999)) + 1] = np.nan
    meta = dict(ulx=float(h.get("ULXMAP", -125.0208333333333)), uly=float(h.get("ULYMAP", 49.9375)),
                xdim=float(h.get("XDIM", 0.0416666666666667)), ydim=float(h.get("YDIM", 0.0416666666666667)))
    return arr, meta


def points_in_polygon(px, py, poly):
    """Vectorised ray-casting point-in-polygon test."""
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
    """Row/col indices of grid cells whose centres fall inside the selection."""
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
    if not m.any():  # selection smaller than one cell -> use the cell under its centroid
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
    """List of (kind, element, year, month, day, url) downloads for the chosen date range."""
    today = dt.date.today()
    tasks = []
    for el in elements:
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


def run_fetch(geom, tasks):
    ERRORS.clear()
    rows, missing, ncells = [], 0, 0
    bar = st.progress(0.0, "Downloading PRISM grids…")
    with ThreadPoolExecutor(2) as ex:
        futs = {ex.submit(fetch_zip, t[5]): t for t in tasks}
        for i, fut in enumerate(as_completed(futs), 1):
            kind, el, y, m, dd, _ = futs[fut]
            data = fut.result()
            if data is None:
                missing += 1
            else:
                v, n = area_value(data, geom)
                ncells = max(ncells, n)
                rows.append(dict(kind=kind, element=el, year=y, month=m, day=dd, value=v))
            bar.progress(i / len(tasks), f"Processed {i}/{len(tasks)} grids")
    bar.empty()
    return pd.DataFrame(rows), missing, ncells


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
        df.loc[is_p, "value"] /= 25.4
        df.loc[~is_p, "value"] = df.loc[~is_p, "value"] * 9 / 5 + 32
    return df


def daily_summary(ts, elements, p_unit, t_unit):
    items = []
    if "ppt" in elements:
        s = ts[ts.element == "ppt"].dropna(subset=["value"]).set_index("date").value
        if len(s):
            wet = 0.254 if p_unit == "mm" else 0.01
            items += [
                ("Total precipitation", f"{s.sum():.1f} {p_unit}"),
                ("Wettest day", f"{s.max():.1f} {p_unit} on {s.idxmax():%b %d}"),
                ("Days with measurable rain", f"{int((s >= wet).sum())} of {len(s)}"),
            ]
    for e, lab, fn in [("tmax", "Hottest daily max", "max"), ("tmin", "Coldest daily min", "min"), ("tmean", "Average mean temp", "mean")]:
        if e in elements:
            s = ts[ts.element == e].dropna(subset=["value"]).set_index("date").value
            if len(s):
                v = getattr(s, fn)()
                extra = f" on {(s.idxmax() if fn == 'max' else s.idxmin()):%b %d}" if fn != "mean" else ""
                items.append((lab, f"{v:.1f} {t_unit}{extra}"))
    if not items:
        st.info("No data in this range.")
        return
    cols = st.columns(min(len(items), 3))
    for k, (lab, val) in enumerate(items):
        cols[k % len(cols)].metric(lab, val)
    if "ppt" in elements:
        s = ts[ts.element == "ppt"].sort_values("date")
        f = go.Figure(go.Scatter(x=s.date, y=s.value.fillna(0).cumsum(), fill="tozeroy", line_color=COLOR["ppt"]))
        f.update_layout(title="Cumulative precipitation", yaxis_title=p_unit, height=320)
        st.plotly_chart(f, use_container_width=True)


# ------------------------------------------------------------------------- UI
st.set_page_config(page_title="PRISM Climate Explorer", layout="wide")
st.title("🌎 PRISM Climate Explorer")
st.caption("Search a place → draw your land on the map → get PRISM climate graphs (continental US).")

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
    elements = st.multiselect("Variables", list(LABEL), default=["ppt", "tmin", "tmax"], format_func=LABEL.get)
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
    if not valid:
        st.info("Pick an end date to complete the range.")
    units = st.radio("Units", ["Metric (mm, °C)", "US (in, °F)"])
    tasks = build_tasks(elements, mode, d0, d1) if valid else []
    too_many = len(tasks) > MAX_FILES
    st.caption(f"≈ {len(tasks)} small downloads (cached after the first run).")
    if too_many:
        st.error(f"Too many files (max {MAX_FILES}). Shorten the range or pick fewer variables.")
    if mode == "Daily":
        st.caption("Recent daily grids are provisional, and the newest days may not be published yet.")

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
folium.LayerControl().add_to(m)

st.subheader("Draw a rectangle or polygon around your land (or drop a pin)")
out = st_folium(m, height=520, use_container_width=True, key=f"map-{ss.map_v}", returned_objects=["all_drawings"])
geom = parse_geom(out.get("all_drawings") if out else None)

if geom is None:
    st.info("Use the toolbar on the left of the map to draw. The most recent shape is used.")
else:
    if geom[0] == "poly":
        st.success(f"Selection: polygon, ≈ {approx_area_km2(geom[1]):.2f} km²")
    else:
        st.success(f"Selection: point at {geom[1][1]:.4f}, {geom[1][0]:.4f} (nearest 4 km PRISM cell)")
    if st.button("📊 Fetch PRISM data", type="primary", disabled=not elements or not tasks or too_many):
        df, missing, ncells = run_fetch(geom, tasks)
        if df.empty or df.value.isna().all():
            st.error("No data returned. PRISM covers the continental US only, so check your selection is inside it.")
            if ERRORS:
                st.caption("First download problem reported:")
                st.code(next(iter(ERRORS.items())))
            ss.data = None
        else:
            ss.data = dict(df=df, missing=missing, ncells=ncells, mode=mode, d0=d0, d1=d1)

# Graphs
if ss.data:
    d = ss.data
    st.markdown(f"**Grid cells averaged:** {d['ncells']}" + (f"  ·  ⚠️ {d['missing']} grids unavailable (likely not yet published)" if d["missing"] else ""))
    df = convert(d["df"], units)
    p_unit, t_unit = ("mm", "°C") if units.startswith("Metric") else ("in", "°F")
    y0, y1 = d["d0"].year, d["d1"].year
    daily = d["mode"] == "Daily"
    step = "Daily" if daily else "Monthly"
    ts = df[df.kind != "normal"].copy()
    ts["date"] = pd.to_datetime(dict(year=ts.year, month=ts.month, day=ts.day))
    ts = ts.sort_values("date")
    temps = [e for e in elements if e != "ppt"]

    t1, t2, t3, t4 = st.tabs([f"{step} time series", "Summary" if daily else "By calendar month", "Annual summary", "Data"])

    with t1:
        if "ppt" in elements:
            s = ts[ts.element == "ppt"]
            f = go.Figure(go.Bar(x=s.date, y=s.value, marker_color=COLOR["ppt"], name="Precipitation"))
            f.update_layout(title=f"{step} precipitation", yaxis_title=p_unit, height=350)
            st.plotly_chart(f, use_container_width=True)
        if temps:
            f = go.Figure()
            for e in temps:
                s = ts[ts.element == e]
                f.add_scatter(x=s.date, y=s.value, mode="lines+markers", name=LABEL[e], line_color=COLOR[e])
            f.update_layout(title=f"{step} temperature", yaxis_title=t_unit, height=350)
            st.plotly_chart(f, use_container_width=True)

    with t2:
        if daily:
            daily_summary(ts, elements, p_unit, t_unit)
        else:
            avg = ts.groupby(["element", "month"]).value.mean().reset_index()
            if "ppt" in elements:
                a = avg[avg.element == "ppt"]
                f = go.Figure(go.Bar(x=[MONTHS[m - 1] for m in a.month], y=a.value, marker_color=COLOR["ppt"]))
                f.update_layout(title=f"Average monthly precipitation ({d['d0']:%b %Y} – {d['d1']:%b %Y})", yaxis_title=p_unit, height=350)
                st.plotly_chart(f, use_container_width=True)
            if temps:
                f = go.Figure()
                for e in temps:
                    a = avg[avg.element == e]
                    f.add_scatter(x=[MONTHS[m - 1] for m in a.month], y=a.value, mode="lines+markers", name=LABEL[e], line_color=COLOR[e])
                f.update_layout(title="Average temperature by calendar month", yaxis_title=t_unit, height=350)
                st.plotly_chart(f, use_container_width=True)

    with t3:
        full = ts.groupby(["element", "year"]).filter(lambda g: len(g) == 12)
        if daily:
            st.info("Annual summary is available in Monthly mode.")
        elif full.empty:
            st.info("No complete calendar years in the selected range.")
        else:
            if "ppt" in elements:
                s = full[full.element == "ppt"].groupby("year").value.sum().reset_index()
                f = go.Figure(go.Bar(x=s.year, y=s.value, marker_color=COLOR["ppt"]))
                f.update_layout(title="Annual precipitation total", yaxis_title=p_unit, height=330)
                st.plotly_chart(f, use_container_width=True)
            if temps:
                f = go.Figure()
                for e in temps:
                    s = full[full.element == e].groupby("year").value.mean().reset_index()
                    f.add_scatter(x=s.year, y=s.value, mode="lines+markers", name=LABEL[e], line_color=COLOR[e])
                f.update_layout(title="Annual mean temperature", yaxis_title=t_unit, height=330)
                st.plotly_chart(f, use_container_width=True)

    with t4:
        wide = df.pivot_table(index=["kind", "year", "month", "day"], columns="element", values="value").reset_index()
        st.dataframe(wide, use_container_width=True)
        st.download_button("Download CSV", wide.to_csv(index=False), "prism_selection.csv", "text/csv")
