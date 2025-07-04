import argparse
import time
from typing import List

import fastf1
from fastf1 import plotting
import pandas as pd
import matplotlib.pyplot as plt
import os


# Enable cache directory
CACHE_DIR = 'fastf1_cache'
os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)


def detect_latest_session() -> tuple[int, int, str]:
    """Return (year, round, session_code) for the most recent session."""
    now = pd.Timestamp.utcnow()
    year = now.year
    schedule = fastf1.get_event_schedule(year)
    # collect all sessions with their UTC start times
    sess_list: list[tuple[pd.Timestamp, int, str]] = []
    for _, row in schedule.iterrows():
        rnd = int(row['RoundNumber'])
        for i in range(1, 6):
            name = row.get(f'Session{i}')
            if not isinstance(name, str) or not name:
                continue
            date_col = f'Session{i}DateUtc'
            start = row.get(date_col)
            if pd.isna(start):
                continue
            start = pd.to_datetime(start, utc=True)
            sess_list.append((start, rnd, name))
    past_sessions = [s for s in sess_list if s[0] <= now]
    if not past_sessions:
        # fall back to first race session
        return year, int(schedule.iloc[0]['RoundNumber']), 'R'
    latest = max(past_sessions, key=lambda x: x[0])
    return year, latest[1], latest[2]


def load_session(year: int, rnd: int, sess: str) -> fastf1.core.Session:
    ses = fastf1.get_session(year, rnd, sess)
    ses.load(laps=True, telemetry=True)
    return ses


def build_race_trace(session: fastf1.core.Session) -> pd.DataFrame:
    laps = session.laps[['Driver', 'LapNumber', 'LapTime']].copy()
    laps['LapTime'] = laps['LapTime'].fillna(pd.Timedelta(0))
    laps['CumTime'] = laps.groupby('Driver')['LapTime'].cumsum()
    trace = laps.pivot(index='LapNumber', columns='Driver', values='CumTime')
    trace = trace.apply(lambda x: x.dt.total_seconds())
    trace = trace.ffill()
    leader = trace.min(axis=1)
    gaps = trace.subtract(leader, axis=0)
    return gaps


def plot_dashboard(session: fastf1.core.Session, drivers: List[str]):
    plotting.setup_mpl()
    trace = build_race_trace(session)
    laps = session.laps.pick_quicklaps().reset_index(drop=True)
    best_laps = {drv: laps.pick_driver(drv).pick_fastest() for drv in drivers}

    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(2, 2)

    ax_trace = fig.add_subplot(gs[0, 0])
    for drv in drivers:
        if drv not in trace.columns:
            continue
        ax_trace.plot(trace.index, trace[drv], label=drv,
                       color=plotting.get_driver_color(drv, session))
    ax_trace.set_xlabel('Lap')
    ax_trace.set_ylabel('Gap to Leader (s)')
    ax_trace.set_title('Race Trace')
    ax_trace.legend(ncol=2)
    ax_trace.grid(True, alpha=0.3)

    ax_laps = fig.add_subplot(gs[0, 1])
    for drv in drivers:
        dlaps = laps.pick_driver(drv)
        if dlaps.empty:
            continue
        ax_laps.plot(dlaps['LapNumber'], dlaps['LapTime'].dt.total_seconds(),
                     label=drv,
                     color=plotting.get_driver_color(drv, session))
    ax_laps.set_xlabel('Lap')
    ax_laps.set_ylabel('Lap Time (s)')
    ax_laps.set_title('Lap Times')
    ax_laps.legend()
    ax_laps.grid(True, alpha=0.3)

    ax_tel = fig.add_subplot(gs[1, :])
    for drv, lap in best_laps.items():
        if lap is None:
            continue
        tel = lap.get_car_data().add_distance()
        ax_tel.plot(tel['Distance'], tel['Speed'],
                    label=f'{drv} Lap {int(lap["LapNumber"])}',
                    color=plotting.get_driver_color(drv, session))
    ax_tel.set_xlabel('Distance (m)')
    ax_tel.set_ylabel('Speed (km/h)')
    ax_tel.set_title('Telemetry Speed - Best Laps')
    ax_tel.legend(ncol=2)
    ax_tel.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show(block=False)
    return fig


def main():
    parser = argparse.ArgumentParser(description='FastF1 Live Dashboard')
    parser.add_argument('--year', type=int)
    parser.add_argument('--round', type=int)
    parser.add_argument('--session', type=str)
    parser.add_argument('--drivers', nargs='+', default=['VER', 'NOR'])
    parser.add_argument('--refresh', type=int, default=0,
                        help='Refresh interval in seconds (0 to disable)')
    args = parser.parse_args()

    if args.year and args.round and args.session:
        year, rnd, sess = args.year, args.round, args.session
    else:
        year, rnd, sess = detect_latest_session()

    session = load_session(year, rnd, sess)
    fig = plot_dashboard(session, args.drivers)

    if args.refresh > 0:
        print(f'Refreshing every {args.refresh} seconds. Close window to stop.')
        try:
            while plt.fignum_exists(fig.number):
                time.sleep(args.refresh)
                session = load_session(year, rnd, sess)
                fig.clf()
                plot_dashboard(session, args.drivers)
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
