import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import requests
import altair as alt
from typing import List, Dict

BASE_URL = 'https://api.openf1.org/v1'


def fetch_json(endpoint: str, params: Dict = None) -> List[Dict]:
    try:
        resp = requests.get(f'{BASE_URL}/{endpoint}', params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        st.error(f'Failed to fetch {endpoint}: {exc}')
        return []


@st.cache_data(ttl=30)
def get_latest_session() -> Dict:
    data = fetch_json('sessions', {'session_key': 'latest'})
    return data[0] if data else {}


@st.cache_data(ttl=30)
def get_drivers(session_key: int) -> List[Dict]:
    return fetch_json('drivers', {'session_key': session_key})


def get_position_data(session_key: int) -> pd.DataFrame:
    data = fetch_json('position', {'session_key': session_key})
    df = pd.DataFrame(data)
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
    return df


def get_lap_data(session_key: int) -> pd.DataFrame:
    data = fetch_json('laps', {'session_key': session_key})
    df = pd.DataFrame(data)
    if not df.empty:
        df['date_start'] = pd.to_datetime(df['date_start'])
    return df


def get_pit_data(session_key: int) -> pd.DataFrame:
    data = fetch_json('pit', {'session_key': session_key})
    df = pd.DataFrame(data)
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
    return df

st.set_page_config(page_title='F1 Live Dashboard', layout='wide')

session = get_latest_session()
if not session:
    st.error('No active session found.')
    st.stop()

session_key = session['session_key']
drivers = get_drivers(session_key)

driver_map = {d['driver_number']: d for d in drivers}
options = list(driver_map.keys())

st.sidebar.title('Driver Selection')
selected = st.sidebar.multiselect(
    'Compare Drivers',
    options,
    default=options[:5],
    format_func=lambda x: driver_map[x]['name_acronym'],
)

refresh = st.sidebar.number_input('Refresh interval (sec)', 5, 60, 15)
st_autorefresh(interval=refresh * 1000, key='refresh')

st.title(f"{session.get('session_name', 'F1 Session')} - Live Dashboard")

st.header('Live Timing')
positions = get_position_data(session_key)
if not positions.empty:
    latest = positions.sort_values('date').drop_duplicates('driver_number', keep='last')
    latest['Driver'] = latest['driver_number'].map(lambda x: driver_map.get(x, {}).get('name_acronym', x))
    latest = latest.sort_values('position')
    st.dataframe(latest[['position', 'Driver', 'laps', 'gap_to_leader', 'interval']], hide_index=True)
else:
    st.write('No position data available.')

st.header('Race Trace')
laps = get_lap_data(session_key)
if not laps.empty:
    laps = laps[laps['driver_number'].isin(selected)]
    laps['Driver'] = laps['driver_number'].map(lambda x: driver_map.get(x, {}).get('name_acronym', x))
    chart = (
        alt.Chart(laps)
        .mark_line()
        .encode(
            x='lap_number',
            y='lap_duration',
            color='Driver',
            tooltip=['Driver', 'lap_number', 'lap_duration']
        )
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)
else:
    st.write('No lap data available.')

st.header('Lap Chart')
if not positions.empty:
    pos_laps = positions[positions['driver_number'].isin(selected)]
    pos_laps['Driver'] = pos_laps['driver_number'].map(lambda x: driver_map.get(x, {}).get('name_acronym', x))
    chart = (
        alt.Chart(pos_laps)
        .mark_line()
        .encode(
            x='lap_number',
            y='position',
            color='Driver',
            tooltip=['Driver', 'lap_number', 'position']
        )
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)
else:
    st.write('No position data for lap chart.')

st.header('Lap Time Series')
if not laps.empty:
    heatmap = (
        alt.Chart(laps)
        .mark_rect()
        .encode(
            x='lap_number:O',
            y='Driver:N',
            color=alt.Color('lap_duration:Q', scale=alt.Scale(scheme='viridis')),
            tooltip=['Driver', 'lap_number', 'lap_duration']
        )
    )
    st.altair_chart(heatmap, use_container_width=True)
else:
    st.write('No lap times for heatmap.')

st.header('Pit Stop Timeline')
pits = get_pit_data(session_key)
if not pits.empty:
    pits = pits[pits['driver_number'].isin(selected)]
    pits['Driver'] = pits['driver_number'].map(lambda x: driver_map.get(x, {}).get('name_acronym', x))
    chart = (
        alt.Chart(pits)
        .mark_circle(size=100)
        .encode(
            x='date',
            y='Driver',
            color='compound',
            tooltip=['Driver', 'lap', 'pit_duration']
        )
    )
    st.altair_chart(chart, use_container_width=True)
else:
    st.write('No pit data available.')
