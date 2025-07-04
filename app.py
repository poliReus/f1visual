import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import requests
import altair as alt
from typing import List, Dict

BASE_URL = "https://api.openf1.org/v1"


# ---------- helpers ----------------------------------------------------------
def fetch_json(endpoint: str, params: Dict | None = None) -> List[Dict]:
    """GET a list of JSON objects from OpenF1, handling errors gracefully."""
    try:
        resp = requests.get(f"{BASE_URL}/{endpoint}", params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        st.error(f"Failed to fetch {endpoint}: {exc}")
        return []


@st.cache_data(ttl=30)
def get_latest_session() -> Dict:
    data = fetch_json("sessions", {"session_key": "latest"})
    return data[0] if data else {}


@st.cache_data(ttl=30)
def get_drivers(session_key: int) -> List[Dict]:
    return fetch_json("drivers", {"session_key": session_key})


@st.cache_data(ttl=15)
def get_lap_data(session_key: int) -> pd.DataFrame:
    df = pd.DataFrame(fetch_json("laps", {"session_key": session_key}))
    if not df.empty:
        df["date_start"] = pd.to_datetime(df["date_start"],format='ISO8601')
        # ⬇️  normalizza
        df["lap_number"]   = pd.to_numeric(df["lap_number"],   errors="coerce")
        df["lap_duration"] = pd.to_numeric(df["lap_duration"], errors="coerce")
    return df


@st.cache_data(ttl=15)
def get_position_data(session_key: int) -> pd.DataFrame:
    df = pd.DataFrame(fetch_json("position", {"session_key": session_key}))
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"],format='ISO8601')
        if "lap_number" in df.columns:
            df["lap_number"] = pd.to_numeric(df["lap_number"], errors="coerce")
        if "position" in df.columns:
            df["position"] = pd.to_numeric(df["position"], errors="coerce")
    return df




@st.cache_data(ttl=30)
def get_pit_data(session_key: int) -> pd.DataFrame:
    df = pd.DataFrame(fetch_json("pit", {"session_key": session_key}))
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], format='ISO8601')
    return df


# ---------- page setup -------------------------------------------------------
st.set_page_config(page_title="F1 Live Dashboard", layout="wide")

session = get_latest_session()
if not session:
    st.error("No active session found.")
    st.stop()

session_key = session["session_key"]
drivers = get_drivers(session_key)
driver_map = {d["driver_number"]: d for d in drivers}

# ---------- sidebar ----------------------------------------------------------
st.sidebar.title("Driver selection")
options = list(driver_map.keys())
selected = st.sidebar.multiselect(
    "Compare drivers",
    options,
    default=options[:5],
    format_func=lambda x: driver_map[x]["name_acronym"],
)

refresh = st.sidebar.number_input("Refresh interval (sec)", 5, 60, 15)
st_autorefresh(interval=int(refresh) * 1000, key="refresh")

st.title(f"{session.get('session_name', 'F1 session')} – Live Dashboard")

# ---------- data pulls (reuse across sections) -------------------------------
positions = get_position_data(session_key)
laps = get_lap_data(session_key)
pits = get_pit_data(session_key)

# ---------- live timing ------------------------------------------------------
st.header("Live timing")

if positions.empty:
    st.write("No position data available.")
else:
    latest = (
        positions.sort_values("date")
        .drop_duplicates("driver_number", keep="last")
        .assign(
            Driver=lambda df: df["driver_number"].map(
                lambda x: driver_map.get(x, {}).get("name_acronym", x)
            )
        )
        .sort_values("position")
    )

    # rebuild “laps” if the /position payload doesn’t include it
    if "laps" not in latest.columns and not laps.empty:
        lap_counts = laps.groupby("driver_number")["lap_number"].max().rename("laps")
        latest = latest.merge(lap_counts, on="driver_number", how="left")

    # columns that actually exist
    wanted = ["position", "Driver"]
    for col in ("laps", "gap_to_leader", "interval"):
        if col in latest.columns:
            wanted.append(col)

    st.dataframe(latest[wanted], hide_index=True)

# ---------- race trace (lap-duration line) -----------------------------------
st.header("Race trace")

if laps.empty:
    st.write("No lap data available.")
else:
    lap_subset = laps[laps["driver_number"].isin(selected)].copy()
    lap_subset["Driver"] = lap_subset["driver_number"].map(
        lambda x: driver_map.get(x, {}).get("name_acronym", x)
    )

    chart = (
        alt.Chart(lap_subset)
        .mark_line()
        .encode(
            x=alt.X("lap_number:Q", title="Lap"),
            y=alt.Y("lap_duration:Q", title="Lap time (s)"),
            color="Driver:N",
            tooltip=["Driver", "lap_number", "lap_duration"],
        )
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)

# ---------- position vs lap chart --------------------------------------------
st.header("Lap chart (position vs lap)")

if positions.empty:
    st.write("No position data for lap chart.")
else:
    pos_laps = positions[positions["driver_number"].isin(selected)].copy()
    pos_laps["Driver"] = pos_laps["driver_number"].map(
        lambda x: driver_map.get(x, {}).get("name_acronym", x)
    )

    chart = (
        alt.Chart(pos_laps)
        .mark_line()
        .encode(
            x=alt.X("lap_number:Q", title="Lap"),
            y=alt.Y("position:Q", sort="ascending", title="Position"),
            color="Driver:N",
            tooltip=["Driver", "lap_number", "position"],
        )
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)

# ---------- lap-time heatmap --------------------------------------------------
st.header("Lap time heatmap")

if laps.empty:
    st.write("No lap times for heatmap.")
else:
    lap_subset = laps[laps["driver_number"].isin(selected)].copy()
    lap_subset["Driver"] = lap_subset["driver_number"].map(
        lambda x: driver_map.get(x, {}).get("name_acronym", x)
    )

    heatmap = (
        alt.Chart(lap_subset)
        .mark_rect()
        .encode(
            x=alt.X("lap_number:O", title="Lap"),  # ordinale (=discreto) va bene
            y="Driver:N",
            color=alt.Color("lap_duration:Q", scale=alt.Scale(scheme="viridis")),
            tooltip=["Driver", "lap_number", "lap_duration"],
        )
    )
    st.altair_chart(heatmap, use_container_width=True)

# ---------- pit-stop timeline -------------------------------------------------
st.header("Pit-stop timeline")

if pits.empty:
    st.write("No pit data available.")
else:
    pit_subset = pits[pits["driver_number"].isin(selected)].copy()
    pit_subset["Driver"] = pit_subset["driver_number"].map(
        lambda x: driver_map.get(x, {}).get("name_acronym", x)
    )

    chart = (
        alt.Chart(pit_subset)
        .mark_circle(size=100)
        .encode(
            x="date",
            y="Driver",
            color="compound",
            tooltip=["Driver", "lap", "pit_duration"],
        )
    )
    st.altair_chart(chart, use_container_width=True)
