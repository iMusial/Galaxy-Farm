"""
PRISM Climate Explorer
----------------------
Search for a place, draw a rectangle/polygon (or drop a pin) on a satellite map,
and pull PRISM monthly climate data for that land. Results are averaged over the
4 km PRISM grid cells inside your selection and shown as interactive graphs.

Install:   pip install streamlit streamlit-folium folium requests numpy pandas plotly
Run:       streamlit run prism_explorer.py

Data: PRISM Climate Group, Oregon State University (https://prism.oregonstate.edu)
Web service: https://services.nacse.org/prism/data/public/...   (continental US only)
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

BASE = "https://services.nacse.org/prism/data/public"
CACHE = Path("prism_cache")
CACHE.mkdir(exist_ok=True)
LABEL = {"ppt": "Precipitation", "tmin": "Min temp", "tmax": "Max temp", "tmean": "Mean temp"}
COLOR = {"ppt": "#2b7bba", "tmin": "#4a90d9", "tmax": "#d9534f", "tmean": "#e6a23c"}
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# ----------------------------------------------------------------- PRISM access
def fetch_zip(url):
    """Download a PRISM zip (with on-disk cache). Returns bytes or None if unavailable."""
    fp = CACHE / (re.sub(r"[^A-Za-z0-9]+", "_", url.replace(BASE, "")) + ".zip")
    if fp.exists():
        return fp.read_bytes()
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=90)
        except requests.RequestException:
            time.sleep(2 * (attempt + 1))
            continue
        if r.status_code == 200 and r.content[:2] == b"PK":
            fp.write_bytes(r.content)
            return r.content
        if r.status_code in (400, 404):
            return None
        time.sleep(2 * (attempt + 1))
    return None


def read_bil(zip_bytes):
    """Parse the .bil grid + .hdr inside a PRISM zip (no GDAL needed)."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        names = z.namelist()
        bil = next(n for n in names if n.lower().endswith(".bil"))
        hdr = next(n for n in names if n.lower().endswith(".hdr"))
        h = dict(l.split(None, 1) for l in z.read(hdr).decode().splitlines() if len(l.split()) > 1)
        h = {k.upper(): v.strip() for k, v in h.items()}
        dtype = "<f4" if h.get("BYTEORDER", "I") == "I" else ">f4"
        arr = np.frombuffer(z.read(bil), dtype=dtype).reshape(int(h["NROWS"]), int(h["NCOLS"])).astype(float)
    arr[arr <= float(h.get("NODATA", -9999)) + 1] = np.nan
    meta = dict(ulx=float(h["ULXMAP"]), uly=float(h["ULYMAP"]), xdim=float(h["XDIM"]), ydim=float(h["YDIM"]))
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


def run_fetch(geom, elements, y0, y1, use_normals):
    month0 = dt.date.today().replace(day=1)
    tasks = []
    for el in elements:
        for y in range(y0, y1 + 1):
            for m in range(1, 13):
                if dt.date(y, m, 1) < month0:
                    tasks.append(("monthly", el, y, m, f"{BASE}/4km/{el}/{y}{m:02d}"))
        if use_normals:
            for m in range(1, 13):
                tasks.append(("normal", el, 0, m, f"{BASE}/normals/4km/{el}/{m:02d}"))
    rows, missing, ncells = [], 0, 0
    bar = st.progress(0.0, "Downloading PRISM grids…")
    with ThreadPoolExecutor(4) as ex:
        futs = {ex.submit(fetch_zip, t[4]): t for t in tasks}
        for i, fut in enumerate(as_completed(futs), 1):
            kind, el, y, m, _ = futs[fut]
            data = fut.result()
            if data is None:
                missing += 1
            else:
                v, n = area_value(data, geom)
                ncells = max(ncells, n)
                rows.append(dict(kind=kind, element=el, year=y, month=m, value=v))
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
    this_year = dt.date.today().year
    y0, y1 = st.slider("Years", 1981, this_year, (this_year - 5, this_year - 1))
    use_normals = st.checkbox("Compare with 1991–2020 normals", value=True)
    units = st.radio("Units", ["Metric (mm, °C)", "US (in, °F)"])
    n_files = len(elements) * ((y1 - y0 + 1) * 12 + (12 if use_normals else 0))
    st.caption(f"≈ {n_files} small downloads (cached after the first run).")

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
    if st.button("📊 Fetch PRISM data", type="primary", disabled=not elements):
        df, missing, ncells = run_fetch(geom, elements, y0, y1, use_normals)
        if df.empty or df.value.isna().all():
            st.error("No data returned. PRISM covers the continental US only — is your selection inside it?")
            ss.data = None
        else:
            ss.data = dict(df=df, missing=missing, ncells=ncells)

# Graphs
if ss.data:
    d = ss.data
    st.markdown(f"**Grid cells averaged:** {d['ncells']}" + (f"  ·  ⚠️ {d['missing']} grids unavailable (likely not yet published)" if d["missing"] else ""))
    df = convert(d["df"], units)
    p_unit, t_unit = ("mm", "°C") if units.startswith("Metric") else ("in", "°F")
    ts = df[df.kind == "monthly"].copy()
    ts["date"] = pd.to_datetime(dict(year=ts.year, month=ts.month, day=1))
    ts = ts.sort_values("date")
    nrm = df[df.kind == "normal"]
    temps = [e for e in elements if e != "ppt"]

    t1, t2, t3, t4 = st.tabs(["Monthly time series", "vs. 30-year normal", "Annual summary", "Data"])

    with t1:
        if "ppt" in elements:
            s = ts[ts.element == "ppt"]
            f = go.Figure(go.Bar(x=s.date, y=s.value, marker_color=COLOR["ppt"], name="Precipitation"))
            f.update_layout(title="Monthly precipitation", yaxis_title=p_unit, height=350)
            st.plotly_chart(f, use_container_width=True)
        if temps:
            f = go.Figure()
            for e in temps:
                s = ts[ts.element == e]
                f.add_scatter(x=s.date, y=s.value, mode="lines+markers", name=LABEL[e], line_color=COLOR[e])
            f.update_layout(title="Monthly temperature", yaxis_title=t_unit, height=350)
            st.plotly_chart(f, use_container_width=True)

    with t2:
        if nrm.empty:
            st.info("Enable “Compare with 1991–2020 normals” and fetch again.")
        else:
            avg = ts.groupby(["element", "month"]).value.mean().reset_index()
            if "ppt" in elements:
                a, n = avg[avg.element == "ppt"], nrm[nrm.element == "ppt"].sort_values("month")
                f = go.Figure()
                f.add_bar(x=MONTHS, y=a.value, name=f"{y0}–{y1} average", marker_color=COLOR["ppt"])
                f.add_bar(x=MONTHS, y=n.value, name="1991–2020 normal", marker_color="#b0b7c3")
                f.update_layout(title="Precipitation by calendar month", barmode="group", yaxis_title=p_unit, height=350)
                st.plotly_chart(f, use_container_width=True)
            if temps:
                f = go.Figure()
                for e in temps:
                    a, n = avg[avg.element == e], nrm[nrm.element == e].sort_values("month")
                    f.add_scatter(x=MONTHS, y=a.value, name=f"{LABEL[e]} ({y0}–{y1})", line_color=COLOR[e])
                    f.add_scatter(x=MONTHS, y=n.value, name=f"{LABEL[e]} normal", line=dict(color=COLOR[e], dash="dash"))
                f.update_layout(title="Temperature by calendar month", yaxis_title=t_unit, height=380)
                st.plotly_chart(f, use_container_width=True)

    with t3:
        full = ts.groupby(["element", "year"]).filter(lambda g: len(g) == 12)
        if full.empty:
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
        wide = df.pivot_table(index=["kind", "year", "month"], columns="element", values="value").reset_index()
        st.dataframe(wide, use_container_width=True)
        st.download_button("Download CSV", wide.to_csv(index=False), "prism_selection.csv", "text/csv")