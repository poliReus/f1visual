"""F1 Multi-Viewer GUI

This script provides a desktop-style multi viewer for Formula 1 live data using
OpenF1 API. It embeds several Matplotlib charts inside a Qt application and
updates them in real time via a background thread.

Run with:
    python f1_multiviewer.py
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from typing import Dict, Optional

import matplotlib
import numpy as np
import pandas as pd
import requests
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6 import QtCore, QtWidgets, QtGui

try:
    import qdarkstyle
    _HAS_DARK = True
except Exception:
    _HAS_DARK = False

matplotlib.use("Agg")  # use non-interactive backend for embedding


class F1API:
    """Wrapper around the OpenF1 API."""

    BASE_URL = "https://api.openf1.org/v1"

    def __init__(self) -> None:
        self.session_key: Optional[int] = None
        self.meeting_key: Optional[int] = None
        self.drivers: Dict[int, Dict] = {}
        self.session = requests.Session()

    def _get(self, endpoint: str) -> list:
        resp = self.session.get(f"{self.BASE_URL}/{endpoint}")
        if resp.status_code == 200:
            try:
                return resp.json()
            except Exception:
                return []
        return []

    def get_session(self, key: int) -> Optional[Dict]:
        data = self._get(f"sessions?session_key={key}")
        return data[0] if data else None

    def get_latest_session(self) -> Optional[Dict]:
        data = self._get("sessions?session_key=latest")
        return data[0] if data else None

    def detect_current_session(self) -> Optional[Dict]:
        """Return live session or last completed session."""
        latest = self.get_latest_session()
        if not latest:
            return None
        now = datetime.now(timezone.utc)
        start = pd.to_datetime(latest["date_start"])
        end = pd.to_datetime(latest["date_end"])
        if start <= now <= end:
            return latest
        key = latest["session_key"] - 1
        for _ in range(10):
            prev = self.get_session(key)
            if not prev:
                key -= 1
                continue
            if pd.to_datetime(prev["date_end"]) <= now:
                return prev
            key -= 1
        return latest

    def get_session_drivers(self) -> Dict[int, Dict]:
        if not self.session_key:
            return {}
        data = self._get(f"drivers?session_key={self.session_key}")
        self.drivers = {d["driver_number"]: d for d in data}
        return self.drivers

    def get_car_data(self, driver_number: int, limit: int = 50) -> pd.DataFrame:
        if not self.session_key:
            return pd.DataFrame()
        data = self._get(
            f"car_data?session_key={self.session_key}&driver_number={driver_number}&limit={limit}"
        )
        df = pd.DataFrame(data)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
        return df

    def get_position_data(self) -> pd.DataFrame:
        if not self.session_key:
            return pd.DataFrame()
        data = self._get(f"position?session_key={self.session_key}")
        df = pd.DataFrame(data)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
        return df

    def get_lap_data(self) -> pd.DataFrame:
        if not self.session_key:
            return pd.DataFrame()
        data = self._get(f"laps?session_key={self.session_key}")
        df = pd.DataFrame(data)
        if not df.empty:
            df["date_start"] = pd.to_datetime(df["date_start"])
        return df

    def get_weather_data(self) -> pd.DataFrame:
        if not self.session_key:
            return pd.DataFrame()
        data = self._get(f"weather?session_key={self.session_key}")
        df = pd.DataFrame(data)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
        return df

    def get_intervals(self) -> pd.DataFrame:
        if not self.session_key:
            return pd.DataFrame()
        data = self._get(f"intervals?session_key={self.session_key}")
        df = pd.DataFrame(data)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
        return df

    def get_pit_data(self) -> pd.DataFrame:
        if not self.session_key:
            return pd.DataFrame()
        data = self._get(f"pit?session_key={self.session_key}")
        df = pd.DataFrame(data)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
        return df


class DataFetcher(QtCore.QThread):
    """Background thread polling the API."""

    dataFetched = QtCore.Signal(dict)

    def __init__(self, api: F1API, interval: int = 5) -> None:
        super().__init__()
        self.api = api
        self.interval = interval
        self._running = True

    def run(self) -> None:
        while self._running:
            session = self.api.detect_current_session()
            if not session:
                self.msleep(self.interval * 1000)
                continue
            self.api.session_key = session["session_key"]
            self.api.meeting_key = session["meeting_key"]
            drivers = self.api.get_session_drivers()
            car_data = {num: self.api.get_car_data(num, 50) for num in list(drivers.keys())[:5]}
            payload = {
                "session": session,
                "drivers": drivers,
                "car_data": car_data,
                "position": self.api.get_position_data(),
                "lap": self.api.get_lap_data(),
                "weather": self.api.get_weather_data(),
                "intervals": self.api.get_intervals(),
                "pit": self.api.get_pit_data(),
            }
            self.dataFetched.emit(payload)
            self.msleep(self.interval * 1000)

    def stop(self) -> None:
        self._running = False


class MplWidget(QtWidgets.QWidget):
    """Base widget embedding a Matplotlib figure."""

    def __init__(self, width: float = 5, height: float = 4, dpi: int = 100) -> None:
        super().__init__()
        self.figure = Figure(figsize=(width, height), dpi=dpi)
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.canvas)
        layout.setContentsMargins(0, 0, 0, 0)


class SpeedAnalysisWidget(MplWidget):
    def __init__(self) -> None:
        super().__init__(10, 8)
        self.axes = self.figure.subplots(2, 2)

    def update_chart(self, car_data: Dict[int, pd.DataFrame], drivers: Dict[int, Dict]) -> None:
        ax1, ax2, ax3, ax4 = self.axes.flat
        for ax in self.axes.flat:
            ax.clear()
        for num, df in car_data.items():
            if df.empty:
                continue
            name = drivers.get(num, {}).get("name_acronym", f"#{num}")
            ax1.plot(df["date"], df["speed"], label=name)
        ax1.set_title("Speed Over Time")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Speed (km/h)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        speeds = [df["speed"].values for df in car_data.values() if not df.empty]
        labels = [drivers.get(n, {}).get("name_acronym", f"#{n}") for n in car_data]
        if speeds:
            ax2.hist(speeds, bins=20, label=labels, alpha=0.7)
            ax2.set_title("Speed Distribution")
            ax2.set_xlabel("Speed (km/h)")
            ax2.set_ylabel("Frequency")
            ax2.legend()

        for num, df in car_data.items():
            if df.empty:
                continue
            name = drivers.get(num, {}).get("name_acronym", f"#{num}")
            ax3.scatter(df["throttle"], df["speed"], label=name, alpha=0.6)
        ax3.set_title("Throttle vs Speed")
        ax3.set_xlabel("Throttle (%)")
        ax3.set_ylabel("Speed (km/h)")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        for num, df in car_data.items():
            if df.empty or "rpm" not in df:
                continue
            name = drivers.get(num, {}).get("name_acronym", f"#{num}")
            ax4.plot(df["date"], df["rpm"], label=name)
        ax4.set_title("RPM Over Time")
        ax4.set_xlabel("Time")
        ax4.set_ylabel("RPM")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        self.figure.tight_layout()
        self.canvas.draw_idle()


class PositionTrackerWidget(MplWidget):
    def __init__(self) -> None:
        super().__init__(8, 6)
        self.ax = self.figure.add_subplot(111)

    def update_chart(self, df: pd.DataFrame, drivers: Dict[int, Dict]) -> None:
        self.ax.clear()
        if not df.empty:
            for num in df["driver_number"].unique():
                d = df[df["driver_number"] == num]
                name = drivers.get(num, {}).get("name_acronym", f"#{num}")
                self.ax.plot(d["date"], d["position"], label=name, marker="o", markersize=3)
        self.ax.set_title("Driver Position Tracker")
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Position")
        self.ax.invert_yaxis()
        self.ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        self.ax.grid(True, alpha=0.3)
        self.figure.tight_layout()
        self.canvas.draw_idle()


class LapTimeWidget(MplWidget):
    def __init__(self) -> None:
        super().__init__(10, 8)
        self.axes = self.figure.subplots(2, 2)

    def update_chart(self, df: pd.DataFrame, drivers: Dict[int, Dict]) -> None:
        ax1, ax2, ax3, ax4 = self.axes.flat
        for ax in self.axes.flat:
            ax.clear()
        if df.empty:
            self.canvas.draw_idle()
            return

        for num in df["driver_number"].unique():
            d = df[df["driver_number"] == num]
            name = drivers.get(num, {}).get("name_acronym", f"#{num}")
            ax1.plot(d["lap_number"], d["lap_duration"], marker="o", label=name)
        ax1.set_title("Lap Times by Driver")
        ax1.set_xlabel("Lap Number")
        ax1.set_ylabel("Lap Duration (s)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        sectors = ["duration_sector_1", "duration_sector_2", "duration_sector_3"]
        means = [df[s].mean() for s in sectors]
        ax2.bar(["Sector 1", "Sector 2", "Sector 3"], means, color=["#FF6B6B", "#4ECDC4", "#45B7D1"])
        ax2.set_title("Average Sector Times")
        ax2.set_ylabel("Duration (s)")

        cols = [c for c in ["i1_speed", "i2_speed", "st_speed"] if c in df.columns]
        for col in cols:
            ax3.hist(df[col].dropna(), bins=15, alpha=0.7, label=col)
        ax3.set_title("Speed Trap Distribution")
        ax3.set_xlabel("Speed (km/h)")
        ax3.set_ylabel("Frequency")
        if cols:
            ax3.legend()

        best = df.groupby("driver_number")["lap_duration"].min().sort_values()
        names = [drivers.get(n, {}).get("name_acronym", f"#{n}") for n in best.index]
        bars = ax4.bar(names, best.values)
        ax4.set_title("Best Lap Times")
        ax4.set_ylabel("Lap Duration (s)")
        ax4.tick_params(axis="x", rotation=45)
        for bar, val in zip(bars, best.values):
            ax4.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.2f}s", ha="center", va="bottom", fontsize=8)

        self.figure.tight_layout()
        self.canvas.draw_idle()


class WeatherWidget(MplWidget):
    def __init__(self) -> None:
        super().__init__(10, 8)
        self.axes = np.array([
            [self.figure.add_subplot(2, 2, 1), self.figure.add_subplot(2, 2, 2)],
            [self.figure.add_subplot(2, 2, 3), self.figure.add_subplot(2, 2, 4, projection="polar")],
        ])

    def update_chart(self, df: pd.DataFrame) -> None:
        ax1, ax2, ax3, ax4 = self.axes.flat
        for ax in self.axes.flat:
            ax.clear()
        if df.empty:
            self.canvas.draw_idle()
            return

        ax1.plot(df["date"], df["air_temperature"], label="Air", color="#FF6B6B")
        ax1.plot(df["date"], df["track_temperature"], label="Track", color="#FF9F43")
        ax1.set_title("Temperature")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("°C")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2_t = ax2.twinx()
        line1 = ax2.plot(df["date"], df["humidity"], color="#4ECDC4", label="Humidity")
        line2 = ax2_t.plot(df["date"], df["pressure"], color="#45B7D1", label="Pressure")
        ax2.set_title("Humidity & Pressure")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Humidity (%)", color="#4ECDC4")
        ax2_t.set_ylabel("Pressure (mbar)", color="#45B7D1")
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax2.legend(lines, labels, loc="upper left")

        ax3.plot(df["date"], df["wind_speed"], color="#26D0CE")
        ax3.set_title("Wind Speed")
        ax3.set_xlabel("Time")
        ax3.set_ylabel("m/s")
        ax3.grid(True, alpha=0.3)

        dirs = np.radians(df["wind_direction"].values)
        ax4.scatter(dirs, df["wind_speed"], c=df["wind_speed"], cmap="viridis", alpha=0.7)
        ax4.set_title("Wind Direction & Speed")
        ax4.set_ylim(0, df["wind_speed"].max() * 1.1)

        self.figure.tight_layout()
        self.canvas.draw_idle()


class IntervalsWidget(MplWidget):
    def __init__(self) -> None:
        super().__init__(10, 4)
        self.axes = self.figure.subplots(1, 2)

    def update_chart(self, df: pd.DataFrame, drivers: Dict[int, Dict]) -> None:
        ax1, ax2 = self.axes
        for ax in self.axes:
            ax.clear()
        if df.empty:
            self.canvas.draw_idle()
            return

        for num in df["driver_number"].unique():
            d = df[df["driver_number"] == num]
            name = drivers.get(num, {}).get("name_acronym", f"#{num}")
            ax1.plot(d["date"], d["gap_to_leader"], marker="o", markersize=3, label=name)
        ax1.set_title("Gap to Leader")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Gap (s)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        latest = df.groupby("driver_number").last()
        names = [drivers.get(n, {}).get("name_acronym", f"#{n}") for n in latest.index]
        bars = ax2.bar(names, latest["gap_to_leader"])
        ax2.set_title("Current Gap to Leader")
        ax2.set_ylabel("Gap (s)")
        ax2.tick_params(axis="x", rotation=45)
        for bar, val in zip(bars, latest["gap_to_leader"]):
            ax2.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.1f}s", ha="center", va="bottom", fontsize=8)

        self.figure.tight_layout()
        self.canvas.draw_idle()


class PitStopWidget(MplWidget):
    def __init__(self) -> None:
        super().__init__(10, 4)
        self.axes = self.figure.subplots(1, 2)

    def update_chart(self, df: pd.DataFrame, drivers: Dict[int, Dict]) -> None:
        ax1, ax2 = self.axes
        for ax in self.axes:
            ax.clear()
        if df.empty:
            self.canvas.draw_idle()
            return

        names = [drivers.get(n, {}).get("name_acronym", f"#{n}") for n in df["driver_number"]]
        bars = ax1.bar(names, df["pit_duration"])
        ax1.set_title("Pit Stop Durations")
        ax1.set_ylabel("Duration (s)")
        ax1.tick_params(axis="x", rotation=45)
        for bar, val in zip(bars, df["pit_duration"]):
            ax1.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.1f}s", ha="center", va="bottom", fontsize=8)

        for _, row in df.iterrows():
            name = drivers.get(row["driver_number"], {}).get("name_acronym", f"#{row['driver_number']}")
            ax2.scatter(row["date"], name, s=100)
        ax2.set_title("Pit Stop Timeline")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Driver")
        ax2.grid(True, alpha=0.3)

        self.figure.tight_layout()
        self.canvas.draw_idle()


class F1MultiViewer(QtWidgets.QMainWindow):
    """Main window for the application."""

    def __init__(self, api: F1API) -> None:
        super().__init__()
        self.api = api
        self.fetcher = DataFetcher(api)
        self.fetcher.dataFetched.connect(self.update_data)
        self.setWindowTitle("F1 Multi Viewer")
        self._init_ui()
        self.fetcher.start()

    def _init_ui(self) -> None:
        if _HAS_DARK:
            self.setStyleSheet(qdarkstyle.load_stylesheet(qt_api="pyside6"))
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(2, 2, 2, 2)

        self.header_label = QtWidgets.QLabel()
        font = self.header_label.font()
        font.setPointSize(12)
        self.header_label.setFont(font)
        layout.addWidget(self.header_label)

        self.tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.tabs)

        self.track_tab = QtWidgets.QWidget()
        t_layout = QtWidgets.QVBoxLayout(self.track_tab)
        self.position_widget = PositionTrackerWidget()
        self.speed_widget = SpeedAnalysisWidget()
        t_layout.addWidget(self.position_widget)
        t_layout.addWidget(self.speed_widget)
        scroll1 = QtWidgets.QScrollArea()
        scroll1.setWidget(self.track_tab)
        scroll1.setWidgetResizable(True)
        self.tabs.addTab(scroll1, "Track")

        self.lap_tab = QtWidgets.QWidget()
        l_layout = QtWidgets.QVBoxLayout(self.lap_tab)
        self.laptime_widget = LapTimeWidget()
        self.intervals_widget = IntervalsWidget()
        l_layout.addWidget(self.laptime_widget)
        l_layout.addWidget(self.intervals_widget)
        scroll2 = QtWidgets.QScrollArea()
        scroll2.setWidget(self.lap_tab)
        scroll2.setWidgetResizable(True)
        self.tabs.addTab(scroll2, "Laps")

        self.strategy_tab = QtWidgets.QWidget()
        s_layout = QtWidgets.QVBoxLayout(self.strategy_tab)
        self.pit_widget = PitStopWidget()
        self.weather_widget = WeatherWidget()
        s_layout.addWidget(self.pit_widget)
        s_layout.addWidget(self.weather_widget)
        scroll3 = QtWidgets.QScrollArea()
        scroll3.setWidget(self.strategy_tab)
        scroll3.setWidgetResizable(True)
        self.tabs.addTab(scroll3, "Strategy")

        self.resize(1200, 800)

    def update_data(self, data: dict) -> None:
        session = data.get("session", {})
        drivers = data.get("drivers", {})
        car_data = data.get("car_data", {})
        pos = data.get("position", pd.DataFrame())
        lap = data.get("lap", pd.DataFrame())
        weather = data.get("weather", pd.DataFrame())
        intervals = data.get("intervals", pd.DataFrame())
        pit = data.get("pit", pd.DataFrame())

        start = pd.to_datetime(session.get("date_start"))
        end = pd.to_datetime(session.get("date_end"))
        now = datetime.now(timezone.utc)
        live = start <= now <= end
        color = "green" if live else "red"
        text = (
            f"<b>{session.get('location', 'Unknown')} - {session.get('session_name', '')}</b>"
            f" | Start: {start.strftime('%Y-%m-%d %H:%M UTC')} "
            f"<font color='{color}'>{'LIVE' if live else 'DEMO'}</font>"
        )
        self.header_label.setText(text)

        self.speed_widget.update_chart(car_data, drivers)
        self.position_widget.update_chart(pos, drivers)
        self.laptime_widget.update_chart(lap, drivers)
        self.weather_widget.update_chart(weather)
        self.pit_widget.update_chart(pit, drivers)

        if session.get("session_type", "").lower() == "race":
            self.intervals_widget.show()
            self.intervals_widget.update_chart(intervals, drivers)
        else:
            self.intervals_widget.hide()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # type: ignore
        self.fetcher.stop()
        self.fetcher.wait(1000)
        event.accept()


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    viewer = F1MultiViewer(F1API())
    viewer.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
