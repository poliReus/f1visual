"""
app.py – Full-featured Formula 1 live dashboard for the OpenF1 API
------------------------------------------------------------------

* Choose year ▸ Grand Prix ▸ session ▸ one or many drivers
* Four UI tabs:
    🏁 Live Timing      – gaps, intervals, positions
    📊 Lap Analysis     – lap times, stint bar chart, heat-map, lap table
    🔧 Telemetry        – fast-lap overlays: speed, throttle, brake, gear, RPM
    🗺 Track            – 3-D scatter of car locations + live weather

Author: you 😊   |   Data: https://api.openf1.org

------------------------------------------------------------------
"""

from __future__ import annotations
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
from functools import lru_cache
from datetime import datetime, timedelta, timezone
import time
from collections import deque
from requests.exceptions import HTTPError

# ── keep timestamps of our last requests ──────────────────────────────
_recent = deque(maxlen=30)          # at most 30 entries (10-s window)

################################################################################
# -------------------------------  CONSTANTS  ----------------------------------
################################################################################
BASE = "https://api.openf1.org/v1"
LIVE_TTL   = 20      # seconds – car_data / intervals / position / weather
HIST_TTL   = 3_600   # seconds – meetings / sessions / drivers / laps / stints
POLL_EVERY = 10_000   # ms – auto-refresh for “live” tabs

# ─────────────────── Command-line argument hook ──────────────────────
import argparse, sys

argp = argparse.ArgumentParser(add_help=False)
argp.add_argument("--session_key",  type=int, help="OpenF1 session_key to load")
args, _ = argp.parse_known_args(sys.argv[1:])

FORCED_SESSION = args.session_key            # None → normal interactive mode


# ───────────────────── Out-lier helper ─────────────────────
def remove_outliers(df: pd.DataFrame, time_col: str, group_col: str) -> pd.DataFrame:
    """Return df with rows outside 1.5×IQR per group removed."""
    clean = []
    for _, g in df.groupby(group_col):
        q1 = g[time_col].quantile(0.25)
        q3 = g[time_col].quantile(0.75)
        iqr = q3 - q1
        mask = (g[time_col] >= q1 - 1.5 * iqr) & (g[time_col] <= q3 + 1.5 * iqr)
        clean.append(g[mask])
    return pd.concat(clean, ignore_index=True) if clean else df


################################################################################
# -----------------------------  LOW-LEVEL I/O  --------------------------------
################################################################################
def _json_df(endpoint: str, params: dict | None = None) -> pd.DataFrame:
    """GET endpoint → DataFrame (UTC dates) while
    • enforcing 30-requests / 10-s client-side limit
    • backing off when server returns HTTP 429
    """
    # client-side throttle
    now = time.perf_counter()
    while _recent and now - _recent[0] < 10:
        if len(_recent) >= 30:
            time.sleep(0.25)
            now = time.perf_counter()
        else:
            break

    url = f"{BASE}/{endpoint}"
    try:
        r = requests.get(url, params=params or {}, timeout=10)

        if r.status_code == 429:                   # server-side limit hit
            retry = int(r.headers.get("Retry-After", "1"))
            time.sleep(retry)
            return _json_df(endpoint, params)      # one recursive retry

        r.raise_for_status()

    except HTTPError as e:
        st.error(f"OpenF1 error: {e}")
        return pd.DataFrame()                      # let caller handle empty

    _recent.append(now)                            # log successful call

    df = pd.json_normalize(r.json())
    for c in df.filter(like="date").columns:
        df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")
    return df

# ─────────────────────────────────  HELPERS  ────────────────────────────────
def _concat_per_driver(endpoint: str, session_key: int, drivers: list[int]) -> pd.DataFrame:
    """Call <endpoint> once per driver_number and concatenate the results."""
    if not drivers:                       # safety: empty selection
        return pd.DataFrame()
    frames = [
        _json_df(endpoint, {"session_key": session_key, "driver_number": drv})
        for drv in drivers
    ]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

# ─────────────────────────────────  LOADERS  ────────────────────────────────
@lru_cache(maxsize=None)
def meetings(year: int | None = None) -> pd.DataFrame:
    return _json_df("meetings", {"year": year} if year else None)

@lru_cache(maxsize=None)
def sessions(meeting_key: int) -> pd.DataFrame:
    return _json_df("sessions", {"meeting_key": meeting_key})

@st.cache_data(ttl=HIST_TTL, show_spinner=False)
def drivers(meeting_key: int) -> pd.DataFrame:
    return _json_df("drivers", {"meeting_key": meeting_key})

# ── driver-scoped endpoints: one request per driver, concat results ─────────
@st.cache_data(ttl=HIST_TTL, show_spinner=False)
def laps(session_key: int, drivers: list[int]) -> pd.DataFrame:
    return _concat_per_driver("laps", session_key, drivers)

@st.cache_data(ttl=LIVE_TTL, show_spinner=False)
def car_data(session_key: int, drivers: list[int]) -> pd.DataFrame:
    return _concat_per_driver("car_data", session_key, drivers)

@st.cache_data(ttl=HIST_TTL, show_spinner=False)
def stints(session_key: int, drivers: list[int]) -> pd.DataFrame:
    return _concat_per_driver("stints", session_key, drivers)

@st.cache_data(ttl=LIVE_TTL, show_spinner=False)
def location(session_key: int, drivers: list[int]) -> pd.DataFrame:
    return _concat_per_driver("location", session_key, drivers)

# ── session-wide endpoints: unchanged ───────────────────────────────────────
@st.cache_data(ttl=LIVE_TTL, show_spinner=False)
def intervals(session_key: int) -> pd.DataFrame:
    return _json_df("intervals", {"session_key": session_key})

@st.cache_data(ttl=LIVE_TTL, show_spinner=False)
def position(session_key: int) -> pd.DataFrame:
    return _json_df("position", {"session_key": session_key})

@st.cache_data(ttl=LIVE_TTL, show_spinner=False)
def weather(session_key: int) -> pd.Series | None:
    df = _json_df("weather", {"session_key": session_key})
    return df.sort_values("date").iloc[-1] if not df.empty else None

################################################################################
# ----------------------------  STREAMLIT CONFIG  ------------------------------
################################################################################
st.set_page_config("OpenF1 Live Dashboard", "🏎", layout="wide")
st.title("🏎  OpenF1 Live Dashboard")

################################################################################
# -------------------------------- SIDEBAR ------------------------------------
################################################################################
if FORCED_SESSION is None:
    with st.sidebar:
        st.header("🗓  Choose session")

        # Year
        seasons = sorted(meetings()["year"].dropna().unique().astype(int), reverse=True)
        year    = st.selectbox("Season", seasons, index=0)

        # Meeting
        year_meet = meetings(year)
        gp_names  = year_meet["meeting_name"].tolist()
        gp        = st.selectbox("Grand Prix", gp_names, index=len(gp_names)-1)
        meeting_key = int(year_meet.loc[year_meet["meeting_name"] == gp, "meeting_key"])

        # Session
        sess_df   = sessions(meeting_key).sort_values("date_start")
        sess_df["label"] = sess_df["session_name"] + " · " + sess_df["session_type"]
        sess_label = st.selectbox("Session", sess_df["label"], index=len(sess_df)-1)
        session_key = int(sess_df.loc[sess_df["label"] == sess_label, "session_key"])

        # Drivers
        drv_df  = drivers(meeting_key)
        drv_df["display"] = drv_df["driver_number"].astype(str)+" · "+drv_df["full_name"]
        default_drvs = drv_df["display"].tolist()[:2]
        chosen = st.multiselect("Drivers", drv_df["display"], default=default_drvs)
        driver_numbers = drv_df.loc[drv_df["display"].isin(chosen), "driver_number"].astype(int).tolist()

        st.divider()
        live = st.toggle("⟳ auto-refresh every 3 s", value=True)

    # ─────────── NEW – works with Streamlit ≥ 1.35 ───────────
    if live:
        if "live" in st.query_params:
            st.rerun()          # flag already set → just rerun
        else:
            st.query_params["live"] = "1"    # first pass → set flag

        # refresh the script every POLL_EVERY ms
        st_autorefresh(interval=POLL_EVERY, key="live")
else:
    # ---------- command-line override ----------
    sess_meta = _json_df("sessions", {"session_key": FORCED_SESSION})
    if sess_meta.empty:
        st.error(f"session_key {FORCED_SESSION} not found in OpenF1.")
        st.stop()

    session_key = int(FORCED_SESSION)
    meeting_key = int(sess_meta["meeting_key"].iloc[0])
    session_name = (
        f"{sess_meta['session_name'].iloc[0]} · "
        f"{sess_meta['session_type'].iloc[0]}"
    )

    st.sidebar.success(f"Fixed session: {session_name}\n(key {session_key})")

################################################################################
# -----------------------  UNIVERSAL LOOK-UPS / MERGES  ------------------------
################################################################################
drv_meta = drivers(meeting_key).set_index("driver_number")
team_colour = {int(k): f"#{v}" for k, v in drv_meta["team_colour"].dropna().items()}
number_to_name = drv_meta["full_name"].to_dict()

################################################################################
# ------------------------------  LOAD DATA  -----------------------------------
################################################################################
lap_df   = laps(session_key, driver_numbers)          # ✔ drivers arg added
tele_df  = car_data(session_key, driver_numbers)      # already correct
gap_df   = intervals(session_key)                     # session-wide
pos_df   = position(session_key)                      # session-wide
stint_df = stints(session_key, driver_numbers)        # ✔ drivers arg added
loc_df   = location(session_key, driver_numbers)      # already correct
weath    = weather(session_key)                       # series

################################################################################
# ---------------------------  DRIVER META MERGE  ------------------------------
################################################################################
# add a human-readable column once, *before* we filter on it
for _df in (lap_df, tele_df, gap_df, pos_df, stint_df, loc_df):
    if not _df.empty and "driver_number" in _df.columns:
        _df["Driver"] = _df["driver_number"].map(number_to_name)

################################################################################
# --------------------------  KEEP ONLY SELECTED DRIVERS  ----------------------
################################################################################
selected_names = [number_to_name[n] for n in driver_numbers]

def keep_only(df):
    """Return rows belonging to the drivers currently selected in the sidebar."""
    if df.empty:
        return df
    if "driver_number" in df.columns:
        return df[df["driver_number"].isin(driver_numbers)]
    if "Driver" in df.columns:
        return df[df["Driver"].isin(selected_names)]
    return df            # unexpected shape: leave unchanged

lap_df   = keep_only(lap_df)
gap_df   = keep_only(gap_df)
pos_df   = keep_only(pos_df)
stint_df = keep_only(stint_df)
loc_df   = keep_only(loc_df)
# tele_df is already limited by the API call


################################################################################
# ---------------------------------  TABS  ------------------------------------
################################################################################
tabs = st.tabs(["🏁 Live Timing", "📊 Lap Analysis", "🔧 Telemetry", "🗺 Track"])

################################################################################
# ------------------------------- LIVE TIMING ----------------------------------
################################################################################
with tabs[0]:
    st.header("🏁 Live Timing")

    colA, colB = st.columns(2)

    with colA:
        st.subheader("Gap to leader")
        if gap_df.empty:
            st.info("No interval data.")
        else:
            fig_gap = px.line(gap_df, x="date", y="gap_to_leader", color="Driver",
                              color_discrete_map=team_colour,
                              labels={"date":"Time (UTC)", "gap_to_leader":"Gap [s]"})
            st.plotly_chart(fig_gap, use_container_width=True)

    with colB:
        st.subheader("Position evolution")
        if pos_df.empty:
            st.info("No position data.")
        else:
            fig_pos = px.line(pos_df, x="date", y="position", color="Driver",
                              color_discrete_map=team_colour, height=350)
            fig_pos.update_yaxes(autorange="reversed", dtick=1)
            st.plotly_chart(fig_pos, use_container_width=True)

    st.divider()
    st.subheader("📋 Current Order")
    if pos_df.empty:
        st.info("Waiting for first sample …")
    else:
        latest = pos_df.sort_values(["date"]).groupby("driver_number").tail(1)
        latest = latest.sort_values("position")
        latest["Car"] = latest["driver_number"].astype(int)
        latest["Gap"] = latest["Driver"].map(
            latest.set_index("Driver")["position"].to_dict()
        )  # dummy, can compute delta to leader here
        st.dataframe(latest[["position","Driver","Car"]]
                     .rename(columns={"position":"Pos"}),
                     hide_index=True, use_container_width=True)

################################################################################
# ------------------------------- LAP ANALYSIS ---------------------------------
################################################################################
# ────────────────────────────  TAB 1  📊 Lap Analysis  ──────────────────────────
with tabs[1]:
    st.header("📊 Lap Analysis")

    # ── per-driver out-lier remover (1.5×IQR, after dropping out-laps) ─────────
    def clean_laps_fn(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        # drop explicit out-laps first
        df = df[~df["is_pit_out_lap"]].copy()
        keep = []
        for _, grp in df.groupby("Driver"):
            q1, q3 = grp["lap_duration"].quantile([0.25, 0.75])
            iqr    = q3 - q1
            mask   = grp["lap_duration"].between(
                q1 - 1.5 * iqr, q3 + 1.5 * iqr, inclusive="both"
            )
            keep.append(grp[mask])
        out = pd.concat(keep, ignore_index=True) if keep else df
        return out.sort_values(["Driver", "lap_number"])

    # ensure numeric lap_duration
    if lap_df["lap_duration"].dtype == object:
        lap_df["lap_duration"] = pd.to_numeric(lap_df["lap_duration"], errors="coerce")

    clean_laps = clean_laps_fn(lap_df)

    col1, col2 = st.columns([3, 2])

    # --- Lap-time lines ----------------------------------------------------------
    with col1:
        st.subheader("Lap times (best ± 20 s)")

        if lap_df.empty:
            st.info("No lap data for the selected drivers.")
        else:
            # numeric and drop explicit out-laps
            lap_df["lap_duration"] = pd.to_numeric(lap_df["lap_duration"], errors="coerce")
            flying = lap_df[~lap_df["is_pit_out_lap"]].copy()

            # keep laps within [best+5, best+20] for each driver
            best = flying.groupby("Driver")["lap_duration"].min()
            flying = flying[
                flying.apply(
                    lambda r: best[r["Driver"]] + 5 <= r["lap_duration"] <= best[r["Driver"]] + 20,
                    axis=1
                )
            ].sort_values(["Driver", "lap_number"])

            fig_lt = px.line(
                flying,
                x="lap_number",
                y="lap_duration",
                color="Driver",
                markers=True,
                color_discrete_map=team_colour,
                labels={"lap_number": "Lap", "lap_duration": "Lap time [s]"},
            )
            fig_lt.update_traces(connectgaps=False)
            st.plotly_chart(fig_lt, use_container_width=True)

    # ── Stint bar chart (unchanged) ------------------------------------------
    with col2:
        st.subheader("Stints (compound)")
        if stint_df.empty:
            st.info("No stints data.")
        else:
            stint_df["colour"] = stint_df["compound"].map(
                {"SOFT": "#ff4d4d", "MEDIUM": "#ffd93d", "HARD": "#d0d0d0",
                 "INTERMEDIATE": "#39c14f", "WET": "#2693ff"}
            )
            fig_st = go.Figure()
            for (drv, comp), g in stint_df.groupby(["Driver", "compound"]):
                fig_st.add_trace(
                    go.Bar(
                        x=g["lap_end"] - g["lap_start"] + 1,
                        y=[drv] * len(g),
                        base=g["lap_start"] - 1,
                        orientation="h",
                        marker_color=g["colour"],
                        name=comp,
                        legendgroup=comp,
                        showlegend=(drv == stint_df["Driver"].iloc[0]),
                    )
                )
            fig_st.update_layout(
                barmode="stack", xaxis_title="Lap", height=260,
                margin=dict(l=0, r=0, t=20, b=0)
            )
            st.plotly_chart(fig_st, use_container_width=True)
    # ╭─ Stint pace candlesticks (using out-lier-free laps) ─────────────────────────
    st.subheader("Stint pace (candlestick)")
    if stint_df.empty or clean_laps.empty:
        st.info("Need both stint and lap data.")
    else:
        ohlc_rows = []
        for _, stint in stint_df.iterrows():
            drv  = stint["Driver"]
            mask = (
                (clean_laps["Driver"] == drv)
                & (clean_laps["lap_number"] >= stint["lap_start"])
                & (clean_laps["lap_number"] <= stint["lap_end"])
            )
            laps_in = clean_laps[mask].sort_values("lap_number")
            if laps_in.empty:
                continue
            ohlc_rows.append(
                dict(
                    Driver=drv,
                    Stint=f"{drv} #{stint['stint_number']}",
                    Open = laps_in["lap_duration"].iloc[0],
                    High = laps_in["lap_duration"].max(),
                    Low  = laps_in["lap_duration"].min(),
                    Close= laps_in["lap_duration"].iloc[-1],
                )
            )

        if ohlc_rows:
            ohlc_df = pd.DataFrame(ohlc_rows)
            fig_cdl = go.Figure()
            for drv, g in ohlc_df.groupby("Driver"):
                fig_cdl.add_trace(
                    go.Candlestick(
                        x=g["Stint"],
                        open=g["Open"],
                        high=g["High"],
                        low=g["Low"],
                        close=g["Close"],
                        name=drv,
                        increasing_line_color=team_colour.get(drv, "#2ca02c"),
                        decreasing_line_color=team_colour.get(drv, "#d62728"),
                    )
                )
            fig_cdl.update_layout(
                yaxis_title="Lap time [s]",
                xaxis_title="Stint",
                height=320,
                margin=dict(l=0, r=0, t=10, b=0),
            )
            st.plotly_chart(fig_cdl, use_container_width=True)

    # ╭─ Mini-sector heat-map (unchanged, uses clean_laps) ──────────────────────────
    st.divider()
    st.subheader("Mini-sector delta heat-map")
    if (
        clean_laps.empty
        or clean_laps[["segments_sector_1", "segments_sector_2"]].isna().all(axis=None)
    ):
        st.info("Mini-sector data not available for this session.")
    else:
        recs = []
        for _, row in clean_laps.iterrows():
            for s, seg_col in enumerate(
                ("segments_sector_1", "segments_sector_2", "segments_sector_3"), 1
            ):
                segs = row[seg_col] or []
                for i, code in enumerate(segs, 1):
                    recs.append(
                        dict(
                            Driver=row["Driver"],
                            Lap=row["lap_number"],
                            Seg=f"S{s}-{i}",
                            Colour=code,
                        )
                    )

        if recs:
            mini = pd.DataFrame(recs)
            pivot = (
                mini.pivot_table(
                    index=["Driver", "Lap"],
                    columns="Seg",
                    values="Colour",
                    aggfunc="first",
                )
                .sort_index(level=["Driver", "Lap"])
            )
            FIA = {2048: "#00c800", 2049: "#f7e400", 2064: "#bf00ff"}
            z   = pivot.map(lambda v: FIA.get(v, "#ffffff"))

            header_vals = ["Driver", "Lap"] + z.columns.tolist()
            driver_col  = [idx[0] for idx in z.index]
            lap_col     = [idx[1] for idx in z.index]

            cell_vals   = [driver_col, lap_col] + [pivot[c].tolist() for c in z.columns]
            cell_colors = (
                [["#ffffff"] * len(z)] * 2  # driver & lap cols
                + [z[c].tolist() for c in z.columns]
            )

            fig_hm = go.Figure(
                data=go.Table(
                    columnwidth=[90, 50] + [22] * len(z.columns),
                    header=dict(values=header_vals, align="left"),
                    cells=dict(values=cell_vals, fill_color=cell_colors, height=18),
                )
            )
            fig_hm.update_layout(
                height=min(420, 18 * len(z) + 40),
                margin=dict(l=0, r=0, t=10, b=0),
            )
            st.plotly_chart(fig_hm, use_container_width=True)

    # ╭─ Detailed lap table (out-lier-free) ─────────────────────────────────────────
    st.divider()
    st.subheader("Detailed lap table")

    if clean_laps.empty:
        st.info("No lap data.")
    else:
        lap_table = (
            clean_laps[
                [
                    "lap_number",
                    "Driver",
                    "lap_duration",
                    "duration_sector_1",
                    "duration_sector_2",
                    "duration_sector_3",
                    "is_pit_out_lap",
                ]
            ]
            .rename(
                columns={
                    "lap_number": "Lap",
                    "lap_duration": "Lap [s]",
                    "duration_sector_1": "S1",
                    "duration_sector_2": "S2",
                    "duration_sector_3": "S3",
                    "is_pit_out_lap": "Out-lap",
                }
            )
            .sort_values(["Lap", "Driver"])
        )
        st.dataframe(lap_table, hide_index=True, use_container_width=True)

# ──────────────────────────────  TAB 2  🔧 Telemetry  ─────────────────────────────
with tabs[2]:
    st.header("🔧 Telemetry – fastest-lap overlay")

    if lap_df.empty or tele_df.empty:
        st.info("Need both lap and telemetry data.")
        st.stop()

    # fastest lap per selected driver
    fastest_idx = lap_df.groupby("driver_number")["lap_duration"].idxmin()
    best_laps   = lap_df.loc[fastest_idx]

    snippets = []
    for _, row in best_laps.iterrows():
        drv   = int(row["driver_number"])
        tele  = tele_df[tele_df["driver_number"] == drv].copy()
        start = row["date_start"]

        # lap end = next start for same driver, else telemetry max
        next_start = (
            lap_df[(lap_df["driver_number"] == drv) & (lap_df["date_start"] > start)]
            ["date_start"]
            .min()
        )
        end = next_start if pd.notnull(next_start) else tele["date"].max()

        tele = tele[(tele["date"] >= start) & (tele["date"] <= end)].copy()
        tele["t"] = (tele["date"] - start).dt.total_seconds()
        tele["Driver"] = row["Driver"]
        snippets.append(tele)

    fastlap = pd.concat(snippets, ignore_index=True)

    # ── FIX: guarantee time monotonic *within* each driver ────────────────
    fastlap.sort_values(["Driver", "t"], inplace=True)
    fastlap.reset_index(drop=True, inplace=True)

    metric_names = {
        "Speed": ("speed", "Speed [km/h]"),
        "Throttle": ("throttle", "Throttle %"),
        "Brake": ("brake", "Brake %"),
        "Gear": ("n_gear", "Gear"),
        "RPM": ("rpm", "Engine RPM"),
    }

    show_together = st.multiselect(
        "🗂  Stack these metrics below the tabs (optional)",
        list(metric_names.keys()),
    )

    # inner tabs
    sub = st.tabs(["Speed", "Throttle", "Brake", "Gear", "RPM", "Δ-time"])

    def trace(y, lbl):
        fig = px.line(
            fastlap,
            x="t",
            y=y,
            color="Driver",
            color_discrete_map=team_colour,
            labels={"t": "Lap time [s]", y: lbl},
        )
        fig.update_traces(connectgaps=False)  # don’t draw across discontinuity
        st.plotly_chart(fig, use_container_width=True)

    with sub[0]:
        trace("speed", "Speed [km/h]")
    with sub[1]:
        trace("throttle", "Throttle %")
    with sub[2]:
        trace("brake", "Brake %")
    with sub[3]:
        trace("n_gear", "Gear")
    with sub[4]:
        trace("rpm", "Engine RPM")

    # Δ-time vs first selected driver
    with sub[5]:
        if len(driver_numbers) < 2:
            st.info("Select at least two drivers for the delta trace.")
        else:
            delta_df = (
                fastlap.pivot_table(
                    index="t", columns="Driver", values="speed", aggfunc="mean"
                )
                .sort_index()
            )

            # drop duplicated column labels
            delta_df = delta_df.loc[:, ~delta_df.columns.duplicated(keep="first")]

            if delta_df.shape[1] < 2:
                st.warning("Not enough overlapping telemetry to compute delta.")
                st.stop()

            ref_name   = delta_df.columns[0]          # pick by position
            ref_series = delta_df.iloc[:, 0]

            delta = delta_df.subtract(ref_series, axis=0)
            delta.columns = [f"{c} – {ref_name}" for c in delta.columns]

            fig_delta = px.line(
                delta,
                x=delta.index,
                y=delta.columns,
                labels={"value": "Speed Δ [km/h]", "t": "Lap time [s]"},
            )
            st.plotly_chart(fig_delta, use_container_width=True)
    # ---------- stacked multi-metric view ----------------------------------
    if show_together:  # multiselect not empty
        st.divider()
        st.subheader("Stacked view")

        from plotly.subplots import make_subplots

        rows = len(show_together)

        fig_stack = make_subplots(
            rows=rows,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=[metric_names[m][1] for m in show_together],
        )

        for r, metr in enumerate(show_together, start=1):
            ycol, label = metric_names[metr]

            # guarantee at least one trace even with a single driver
            for drv, g in fastlap.groupby("Driver", sort=False):
                line_col = team_colour.get(drv, px.colors.qualitative.Set1[0])

                fig_stack.add_scatter(
                    x=g["t"],
                    y=g[ycol],
                    mode="lines",
                    name=drv if r == 1 else None,  # legend only once
                    legendgroup=drv,  # colour consistency
                    line=dict(color=line_col),
                    showlegend=(r == 1),
                    row=r,
                    col=1,
                )

            fig_stack.update_yaxes(title_text=label, row=r, col=1)

        fig_stack.update_xaxes(title_text="Lap time [s]", row=rows, col=1)
        fig_stack.update_layout(
            height=280 * rows if rows else 300,
            margin=dict(l=0, r=0, t=30, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        )
        st.plotly_chart(fig_stack, use_container_width=True)

################################################################################
# -------------------------------- TRACK MAP ----------------------------------
################################################################################
with tabs[3]:
    st.header("🗺 Track & Weather")

    colL, colR = st.columns([3, 1])

    # ---------- Track map ----------------------------------------------------
    with colL:
        if loc_df.empty:
            st.info("Car location not available for this session.")
        else:
            # keep latest position only – avoid huge scatter
            latest_t = loc_df.groupby("driver_number")["date"].idxmax()
            show     = loc_df.loc[latest_t]

            fig_map = px.scatter_3d(
                show,
                x="x", y="y", z="z",
                color="Driver",
                color_discrete_map=team_colour,
                size=[6]*len(show),
            )
            fig_map.update_layout(
                scene=dict(
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False),
                    zaxis=dict(visible=False),
                ),
                margin=dict(l=0, r=0, t=0, b=0),
                height=520,
            )
            st.plotly_chart(fig_map, use_container_width=True)

    # ---------- Weather ------------------------------------------------------
    with colR:
        st.subheader("Weather")
        if weath is None or weath.empty:
            st.info("No weather data yet.")
        else:
            txt = lambda v, u="°C": f"**{v:.1f}{u}**"
            st.markdown(
                f"""
*{weath['date'].strftime('%H:%M:%S UTC')}*

* Air&nbsp;&nbsp;{txt(weath['air_temperature'])}
* Track {txt(weath['track_temperature'])}
* Humidity **{weath['humidity']} %**
* Pressure **{weath['pressure']} hPa**
* Wind **{weath['wind_speed']} km/h** @ {weath['wind_direction']}°
* Rainfall **{weath['rainfall']}**
"""
            )

################################################################################
# --------------------------------  FOOTER  -----------------------------------
################################################################################
st.markdown("---")
st.caption(
    "Dashboard powered by **OpenF1** – historical data is free; live access requires an API key. "
    "Built with ❤️ using Streamlit & Plotly."
)
