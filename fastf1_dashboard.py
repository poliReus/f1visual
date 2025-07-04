import argparse
import fastf1
from fastf1 import plotting
import pandas as pd
import matplotlib.pyplot as plt

# Enable cache directory for faster subsequent loads
fastf1.Cache.enable_cache('fastf1_cache')


def load_session(year: int, event: int, session: str):
    """Load a session using FastF1 with caching."""
    return fastf1.get_session(year, event, session)


def plot_laptimes(laps: pd.DataFrame, drivers: list[str]):
    """Plot lap times for given drivers."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for drv in drivers:
        dlaps = laps.pick_driver(drv)
        if dlaps.empty:
            continue
        ax.plot(dlaps['LapNumber'], dlaps['LapTime'].dt.total_seconds(), label=drv)
    ax.set_xlabel('Lap')
    ax.set_ylabel('Lap Time (s)')
    ax.set_title('Lap Time Comparison')
    ax.legend()
    fig.tight_layout()
    plt.show()


def plot_telemetry(lap: fastf1.core.Lap, driver: str):
    """Plot telemetry data for a single lap."""
    telemetry = lap.get_car_data().add_distance()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(telemetry['Distance'], telemetry['Speed'])
    lap_num = lap.get('LapNumber', None)
    ax.set_title(f'Telemetry Speed - {driver} Lap {lap_num}')
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Speed (km/h)')
    fig.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='FastF1 simple dashboard')
    parser.add_argument('--year', type=int, default=pd.Timestamp.now().year)
    parser.add_argument('--round', type=int, required=True, help='Race round number')
    parser.add_argument('--session', type=str, default='R', help='Session code (FP1, FP2, Q, R)')
    parser.add_argument('--drivers', nargs='+', default=['VER', 'NOR'], help='Driver abbreviations')
    args = parser.parse_args()

    plotting.setup_mpl()
    ses = load_session(args.year, args.round, args.session)
    ses.load(laps=True, telemetry=True)

    laps = ses.laps.pick_quicklaps().reset_index(drop=True)
    plot_laptimes(laps, args.drivers)

    best_laps = {drv: laps.pick_driver(drv).pick_fastest() for drv in args.drivers}
    for drv, lap in best_laps.items():
        if lap is not None:
            plot_telemetry(lap, drv)


if __name__ == '__main__':
    main()
