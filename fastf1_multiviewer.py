import argparse
from typing import List

import fastf1
from fastf1 import plotting
import pandas as pd
import matplotlib.pyplot as plt

# Enable cache directory for faster loads
fastf1.Cache.enable_cache('fastf1_cache')


def load_session(year: int, event: int, session: str) -> fastf1.core.Session:
    """Load a session using FastF1."""
    ses = fastf1.get_session(year, event, session)
    ses.load(laps=True, telemetry=True, weather=False, messages=False)
    return ses


def build_race_trace(session: fastf1.core.Session) -> pd.DataFrame:
    """Return gap to leader per lap for all drivers."""
    laps = session.laps[['Driver', 'LapNumber', 'LapTime']].copy()
    laps['LapTime'] = laps['LapTime'].fillna(pd.Timedelta(0))
    laps['CumTime'] = laps.groupby('Driver')['LapTime'].cumsum()
    trace = laps.pivot(index='LapNumber', columns='Driver', values='CumTime')
    trace = trace.apply(lambda x: x.dt.total_seconds())
    trace = trace.ffill()
    leader = trace.min(axis=1)
    gaps = trace.subtract(leader, axis=0)
    return gaps


def plot_race_trace(trace: pd.DataFrame, session: fastf1.core.Session, drivers: List[str]) -> None:
    """Plot the race trace chart."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for drv in drivers:
        if drv not in trace.columns:
            continue
        color = plotting.get_driver_color(drv, session)
        ax.plot(trace.index, trace[drv], label=drv, color=color)
    ax.set_xlabel('Lap')
    ax.set_ylabel('Gap to Leader (s)')
    ax.set_title('Race Trace')
    ax.legend(ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()


def plot_telemetry_overlay(lap: fastf1.core.Lap, driver: str) -> None:
    """Display a minimal telemetry overlay for a lap."""
    tel = lap.get_car_data().add_distance()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(tel['Distance'], tel['Speed'], label='Speed')
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Speed (km/h)')
    ax.set_title(f'Telemetry - {driver} best lap')
    ax.grid(True, alpha=0.3)

    box = fig.add_axes([0.7, 0.6, 0.25, 0.3])
    box.axis('off')
    metrics = {
        'Lap': int(lap['LapNumber']),
        'LapTime': lap['LapTime'],
        'Compound': lap['Compound'],
    }
    for i, (k, v) in enumerate(metrics.items()):
        box.text(0.05, 0.9 - i * 0.25, f"{k}: {v}")

    fig.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description='FastF1 multi viewer')
    parser.add_argument('--year', type=int, default=pd.Timestamp.now().year)
    parser.add_argument('--round', type=int, required=True)
    parser.add_argument('--session', default='R')
    parser.add_argument('--drivers', nargs='+', default=[])
    args = parser.parse_args()

    plotting.setup_mpl()
    ses = load_session(args.year, args.round, args.session)

    trace = build_race_trace(ses)
    drivers = args.drivers or list(trace.columns)
    plot_race_trace(trace, ses, drivers)

    for drv in drivers:
        laps = ses.laps.pick_driver(drv)
        if not laps.empty:
            best = laps.pick_fastest()
            if best is not None:
                plot_telemetry_overlay(best, drv)


if __name__ == '__main__':
    main()
