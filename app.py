import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import requests
import altair as alt
from typing import List, Dict
import numpy as np

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
    data = fetch_json("laps", {"session_key": session_key})
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    if not df.empty:
        df["date_start"] = pd.to_datetime(df.get("date_start"), utc=True, errors="coerce")

        numeric_cols = [
            "lap_number",
            "duration_sector_1",
            "duration_sector_2",
            "duration_sector_3",
            "i1_speed",
            "i2_speed",
            "lap_duration",
            "st_speed",
        ]

        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["lap_duration"])
    return df


@st.cache_data(ttl=15)
def get_position_data(session_key: int) -> pd.DataFrame:
    data = fetch_json("position", {"session_key": session_key})
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")

        numeric_cols = ["lap_number", "position"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


@st.cache_data(ttl=30)
def get_pit_data(session_key: int) -> pd.DataFrame:
    data = fetch_json("pit", {"session_key": session_key})
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")

        numeric_cols = ["pit_duration", "lap_number"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


@st.cache_data(ttl=15)
def get_car_data(
    session_key: int,
    driver_number: int,
    date_start: str,
    date_end: str,
) -> pd.DataFrame:
    """Return car telemetry for a specific time range."""
    params = {
        "session_key": session_key,
        "driver_number": driver_number,
        "date>": date_start,
        "date<": date_end,
    }
    data = fetch_json("car_data", params)
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")

        numeric_cols = ["throttle", "speed", "rpm", "brake", "n_gear"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["date"])
    return df


# ---------- page setup -------------------------------------------------------
st.set_page_config(page_title="F1 Live Dashboard", layout="wide")

session = get_latest_session()
if not session:
    st.error("No active session found.")
    st.stop()

session_key = session["session_key"]
drivers = get_drivers(session_key)

if not drivers:
    st.error("No drivers found for this session.")
    st.stop()

driver_map = {d["driver_number"]: d for d in drivers}

# ---------- sidebar ----------------------------------------------------------
st.sidebar.title("Driver selection")
options = list(driver_map.keys())
selected = st.sidebar.multiselect(
    "Compare drivers",
    options,
    default=options[: min(5, len(options))],
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
        .copy()
    )

    # Add driver names
    latest["Driver"] = latest["driver_number"].map(
        lambda x: driver_map.get(x, {}).get("name_acronym", str(x))
    )

    # Add lap counts if available
    if not laps.empty and "lap_number" in laps.columns:
        lap_counts = laps.groupby("driver_number")["lap_number"].max().reset_index()
        lap_counts.columns = ["driver_number", "laps"]
        latest = latest.merge(lap_counts, on="driver_number", how="left")

    # Sort by position
    latest = latest.sort_values("position")

    # Select columns to display
    display_cols = ["position", "Driver"]
    for col in ["laps", "gap_to_leader", "interval"]:
        if col in latest.columns:
            display_cols.append(col)

    st.dataframe(latest[display_cols], hide_index=True)

# ---------- race trace (lap-duration line) -----------------------------------
st.header("Race trace")

if laps.empty:
    st.write("No lap data available.")
else:
    lap_subset = laps[laps["driver_number"].isin(selected)].copy()

    if lap_subset.empty:
        st.write("No lap data for selected drivers.")
    else:
        lap_subset["Driver"] = lap_subset["driver_number"].map(
            lambda x: driver_map.get(x, {}).get("name_acronym", str(x))
        )

        # Remove invalid lap times
        if "lap_number" in lap_subset.columns:
            lap_subset = lap_subset.dropna(subset=["lap_duration", "lap_number"])
        else:
            lap_subset = lap_subset.dropna(subset=["lap_duration"])

        if "lap_number" not in lap_subset.columns:
            st.write("Lap numbers not available for race trace.")
        elif not lap_subset.empty:
            chart = (
                alt.Chart(lap_subset)
                .mark_line(point=True)
                .encode(
                    x=alt.X("lap_number:Q", title="Lap"),
                    y=alt.Y("lap_duration:Q", title="Lap time (s)"),
                    color=alt.Color("Driver:N", legend=alt.Legend(title="Driver")),
                    tooltip=["Driver", "lap_number", "lap_duration"],
                )
                .interactive()
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.write("No valid lap data for selected drivers.")

# ---------- position vs lap chart --------------------------------------------
st.header("Lap chart (position vs lap)")

if positions.empty:
    st.write("No position data for lap chart.")
else:
    pos_laps = positions[positions["driver_number"].isin(selected)].copy()

    if pos_laps.empty:
        st.write("No position data for selected drivers.")
    else:
        pos_laps["Driver"] = pos_laps["driver_number"].map(
            lambda x: driver_map.get(x, {}).get("name_acronym", str(x))
        )

        # Filter out invalid positions and lap numbers
        if "lap_number" in pos_laps.columns:
            pos_laps = pos_laps.dropna(subset=["position", "lap_number"])
        else:
            pos_laps = pos_laps.dropna(subset=["position"])

        if "lap_number" not in pos_laps.columns:
            st.write("Lap numbers not available for lap chart.")
        elif not pos_laps.empty:
            chart = (
                alt.Chart(pos_laps)
                .mark_line(point=True)
                .encode(
                    x=alt.X("lap_number:Q", title="Lap"),
                    y=alt.Y(
                        "position:Q", scale=alt.Scale(reverse=True), title="Position"
                    ),
                    color=alt.Color("Driver:N", legend=alt.Legend(title="Driver")),
                    tooltip=["Driver", "lap_number", "position"],
                )
                .interactive()
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.write("No valid position data for selected drivers.")

# ---------- lap-time heatmap --------------------------------------------------
st.header("Lap time heatmap")

if laps.empty:
    st.write("No lap times for heatmap.")
else:
    lap_subset = laps[laps["driver_number"].isin(selected)].copy()

    if lap_subset.empty:
        st.write("No lap data for selected drivers.")
    else:
        lap_subset["Driver"] = lap_subset["driver_number"].map(
            lambda x: driver_map.get(x, {}).get("name_acronym", str(x))
        )

        # Remove invalid data
        if "lap_number" in lap_subset.columns:
            lap_subset = lap_subset.dropna(subset=["lap_duration", "lap_number"])
        else:
            lap_subset = lap_subset.dropna(subset=["lap_duration"])

        if "lap_number" not in lap_subset.columns:
            st.write("Lap numbers not available for heatmap.")
        elif not lap_subset.empty:
            heatmap = (
                alt.Chart(lap_subset)
                .mark_rect()
                .encode(
                    x=alt.X("lap_number:O", title="Lap"),
                    y=alt.Y("Driver:N", title="Driver"),
                    color=alt.Color(
                        "lap_duration:Q",
                        scale=alt.Scale(scheme="viridis", reverse=True),
                        title="Lap Time (s)",
                    ),
                    tooltip=["Driver", "lap_number", "lap_duration"],
                )
            )
            st.altair_chart(heatmap, use_container_width=True)
        else:
            st.write("No valid lap data for selected drivers.")

# ---------- throttle trace over best lap ------------------------------------
st.header("Best lap throttle trace")

if laps.empty:
    st.write("No lap data available for throttle traces.")
else:
    # Get best lap for each selected driver
    best_laps = (
        laps[laps["driver_number"].isin(selected)]
        .dropna(subset=["lap_duration", "date_start"])
        .loc[laps.groupby("driver_number")["lap_duration"].idxmin()]
        .reset_index(drop=True)
    )

    if best_laps.empty:
        st.write("No best lap data available.")
    else:
        traces = []
        for _, row in best_laps.iterrows():
            try:
                start = row["date_start"].isoformat()
                end = (
                    row["date_start"] + pd.to_timedelta(row["lap_duration"], unit="s")
                ).isoformat()

                df = get_car_data(session_key, row["driver_number"], start, end)

                if df.empty:
                    continue

                # Filter out invalid throttle data
                df = df.dropna(subset=["throttle", "date"])

                if df.empty:
                    continue

                # Calculate time from start
                df["t"] = (df["date"] - df["date"].iloc[0]).dt.total_seconds()
                df["Driver"] = driver_map.get(row["driver_number"], {}).get(
                    "name_acronym", str(row["driver_number"])
                )
                df["Best_Lap_Time"] = row["lap_duration"]

                traces.append(df[["t", "throttle", "Driver", "Best_Lap_Time"]])

            except Exception as e:
                st.warning(
                    f"Could not fetch throttle data for driver {row['driver_number']}: {e}"
                )
                continue

        if not traces:
            st.write("No throttle telemetry available for selected drivers.")
        else:
            data = pd.concat(traces, ignore_index=True)

            # Create throttle chart
            chart = (
                alt.Chart(data)
                .mark_line(strokeWidth=2)
                .encode(
                    x=alt.X("t:Q", title="Time since lap start (s)"),
                    y=alt.Y(
                        "throttle:Q",
                        title="Throttle (%)",
                        scale=alt.Scale(domain=[0, 100]),
                    ),
                    color=alt.Color("Driver:N", legend=alt.Legend(title="Driver")),
                    tooltip=["Driver", "t:Q", "throttle:Q", "Best_Lap_Time:Q"],
                )
                .interactive()
            )
            st.altair_chart(chart, use_container_width=True)

            # Show best lap times
            best_times = data.groupby("Driver")["Best_Lap_Time"].first().sort_values()
            st.subheader("Best lap times")
            for driver, time in best_times.items():
                st.write(f"**{driver}**: {time:.3f}s")

# ---------- speed trace over best lap ---------------------------------------
st.header("Best lap speed trace")

if laps.empty:
    st.write("No lap data available for speed traces.")
else:
    # Reuse the best_laps from throttle section
    best_laps = (
        laps[laps["driver_number"].isin(selected)]
        .dropna(subset=["lap_duration", "date_start"])
        .loc[laps.groupby("driver_number")["lap_duration"].idxmin()]
        .reset_index(drop=True)
    )

    if best_laps.empty:
        st.write("No best lap data available.")
    else:
        speed_traces = []
        for _, row in best_laps.iterrows():
            try:
                start = row["date_start"].isoformat()
                end = (
                    row["date_start"] + pd.to_timedelta(row["lap_duration"], unit="s")
                ).isoformat()

                df = get_car_data(session_key, row["driver_number"], start, end)

                if df.empty:
                    continue

                # Filter out invalid speed data
                df = df.dropna(subset=["speed", "date"])

                if df.empty:
                    continue

                # Calculate time from start
                df["t"] = (df["date"] - df["date"].iloc[0]).dt.total_seconds()
                df["Driver"] = driver_map.get(row["driver_number"], {}).get(
                    "name_acronym", str(row["driver_number"])
                )

                speed_traces.append(df[["t", "speed", "Driver"]])

            except Exception as e:
                continue

        if not speed_traces:
            st.write("No speed telemetry available for selected drivers.")
        else:
            speed_data = pd.concat(speed_traces, ignore_index=True)

            # Create speed chart
            speed_chart = (
                alt.Chart(speed_data)
                .mark_line(strokeWidth=2)
                .encode(
                    x=alt.X("t:Q", title="Time since lap start (s)"),
                    y=alt.Y("speed:Q", title="Speed (km/h)"),
                    color=alt.Color("Driver:N", legend=alt.Legend(title="Driver")),
                    tooltip=["Driver", "t:Q", "speed:Q"],
                )
                .interactive()
            )
            st.altair_chart(speed_chart, use_container_width=True)

# ---------- pit-stop timeline -------------------------------------------------
st.header("Pit-stop timeline")

if pits.empty:
    st.write("No pit data available.")
else:
    pit_subset = pits[pits["driver_number"].isin(selected)].copy()

    if pit_subset.empty:
        st.write("No pit data for selected drivers.")
    else:
        pit_subset["Driver"] = pit_subset["driver_number"].map(
            lambda x: driver_map.get(x, {}).get("name_acronym", str(x))
        )

        # Remove invalid pit data
        pit_subset = pit_subset.dropna(subset=["date"])

        if not pit_subset.empty:
            tooltips = ["Driver", "date", "pit_duration"]
            if "lap_number" in pit_subset.columns:
                tooltips.insert(2, "lap_number")

            chart = (
                alt.Chart(pit_subset)
                .mark_circle(size=100)
                .encode(
                    x=alt.X("date:T", title="Time"),
                    y=alt.Y("Driver:N", title="Driver"),
                    color=(
                        alt.Color("compound:N", title="Compound")
                        if "compound" in pit_subset.columns
                        else alt.value("orange")
                    ),
                    tooltip=tooltips,
                )
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.write("No valid pit data for selected drivers.")

# ---------- session info -----------------------------------------------------
st.sidebar.header("Session Info")
st.sidebar.write(f"**Session**: {session.get('session_name', 'Unknown')}")
st.sidebar.write(f"**Session Key**: {session_key}")
if "date_start" in session:
    st.sidebar.write(f"**Start**: {session['date_start']}")
if "date_end" in session:
    st.sidebar.write(f"**End**: {session['date_end']}")

# ---------- data status ------------------------------------------------------
st.sidebar.header("Data Status")
st.sidebar.write(f"**Drivers**: {len(drivers)}")
st.sidebar.write(f"**Position records**: {len(positions)}")
st.sidebar.write(f"**Lap records**: {len(laps)}")
st.sidebar.write(f"**Pit records**: {len(pits)}")
