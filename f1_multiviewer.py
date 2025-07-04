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
from typing import Dict, List, Optional

import matplotlib
import numpy as np
import pandas as pd
import requests
import time
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6 import QtCore, QtWidgets, QtGui

try:
    import qdarkstyle
    _HAS_DARK = True
except Exception:
    _HAS_DARK = False


DEFAULT_COLOR = "#333333"


def get_color(drivers: Dict[int, Dict], num: int, default: str = DEFAULT_COLOR) -> str:
    """Return team color for driver as hex string.

    If the driver's team colour is missing or invalid, ``default`` is returned
    to ensure matplotlib always receives a valid colour value.
    """
    color = drivers.get(num, {}).get("team_colour")
    if color and isinstance(color, str) and color.startswith("#"):
        return color
    return default

matplotlib.use("Agg")  # use non-interactive backend for embedding


class F1API:
    """Wrapper around the OpenF1 API."""

    BASE_URL = "https://api.openf1.org/v1"

    def __init__(self) -> None:
        self.session_key: Optional[int] = None
        self.meeting_key: Optional[int] = None
        self.drivers: Dict[int, Dict] = {}
        self.session = requests.Session()
        # simple cache to avoid spamming the API and to survive
        # temporary network hiccups
        self._cache: Dict[str, tuple[float, list]] = {}
        self.cache_ttl = 5.0

    def _get(self, endpoint: str) -> list:
        now = time.time()
        if endpoint in self._cache:
            ts, data = self._cache[endpoint]
            if now - ts < self.cache_ttl:
                return data
        try:
            resp = self.session.get(f"{self.BASE_URL}/{endpoint}")
            if resp.status_code == 200:
                data = resp.json()
                self._cache[endpoint] = (now, data)
                return data
        except Exception:
            pass
        # fall back to cached data if available
        return self._cache.get(endpoint, (now, []))[1]

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
        start = pd.to_datetime(latest["date_start"], utc=True, format="ISO8601")
        end = pd.to_datetime(latest["date_end"], utc=True, format="ISO8601")
        if start <= now <= end:
            return latest
        key = latest["session_key"] - 1
        for _ in range(10):
            prev = self.get_session(key)
            if not prev:
                key -= 1
                continue
            if pd.to_datetime(prev["date_end"], utc=True, format="ISO8601") <= now:
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
            df["date"] = pd.to_datetime(df["date"], utc=True, format="ISO8601", errors="coerce")
        return df

    def get_car_data_range(self, driver_number: int, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """Return car telemetry for a specific time range."""
        if not self.session_key:
            return pd.DataFrame()
        start_str = start.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        end_str = end.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        query = (
            f"car_data?session_key={self.session_key}&driver_number={driver_number}"
            f"&date>={start_str}&date<={end_str}"
        )
        data = self._get(query)
        df = pd.DataFrame(data)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"], utc=True, format="ISO8601", errors="coerce")
        return df

    def get_position_data(self) -> pd.DataFrame:
        if not self.session_key:
            return pd.DataFrame()
        data = self._get(f"position?session_key={self.session_key}")
        df = pd.DataFrame(data)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"], utc=True, format="ISO8601", errors="coerce")
        return df

    def get_lap_data(self) -> pd.DataFrame:
        if not self.session_key:
            return pd.DataFrame()
        data = self._get(f"laps?session_key={self.session_key}")
        df = pd.DataFrame(data)
        if not df.empty:
            df["date_start"] = pd.to_datetime(df["date_start"], utc=True, format="ISO8601", errors="coerce")
        return df

    def get_weather_data(self) -> pd.DataFrame:
        if not self.session_key:
            return pd.DataFrame()
        data = self._get(f"weather?session_key={self.session_key}")
        df = pd.DataFrame(data)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"], utc=True, format="ISO8601", errors="coerce")
        return df

    def get_intervals(self) -> pd.DataFrame:
        if not self.session_key:
            return pd.DataFrame()
        data = self._get(f"intervals?session_key={self.session_key}")
        df = pd.DataFrame(data)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"], utc=True, format="ISO8601", errors="coerce")
        return df

    def get_pit_data(self) -> pd.DataFrame:
        if not self.session_key:
            return pd.DataFrame()
        data = self._get(f"pit?session_key={self.session_key}")
        df = pd.DataFrame(data)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"], utc=True, format="ISO8601", errors="coerce")
        return df

    def get_stint_data(self) -> pd.DataFrame:
        if not self.session_key:
            return pd.DataFrame()
        data = self._get(f"stints?session_key={self.session_key}")
        return pd.DataFrame(data)


class DataFetcher(QtCore.QThread):
    """Background thread polling the API."""

    dataFetched = QtCore.Signal(dict)

    def __init__(self, api: F1API, interval: int = 5) -> None:
        super().__init__()
        self.api = api
        self.interval = interval
        self._running = True
        self.selected_drivers: List[int] = []

    def set_selected_drivers(self, drivers: List[int]) -> None:
        self.selected_drivers = drivers

    def run(self) -> None:
        while self._running:
            session = self.api.detect_current_session()
            if not session:
                self.msleep(self.interval * 1000)
                continue
            self.api.session_key = session["session_key"]
            self.api.meeting_key = session["meeting_key"]
            drivers = self.api.get_session_drivers()
            selected = self.selected_drivers or list(drivers.keys())[:5]
            car_data = {num: self.api.get_car_data(num, 200) for num in selected}
            payload = {
                "session": session,
                "drivers": drivers,
                "car_data": car_data,
                "position": self.api.get_position_data(),
                "lap": self.api.get_lap_data(),
                "weather": self.api.get_weather_data(),
                "intervals": self.api.get_intervals(),
                "pit": self.api.get_pit_data(),
                "stints": self.api.get_stint_data(),
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
            color = get_color(drivers, num)
            ax1.plot(df["date"], df["speed"], label=name, color=color)
        ax1.set_title("Speed Over Time")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Speed (km/h)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        if car_data:
            for num, df in car_data.items():
                if df.empty:
                    continue
                name = drivers.get(num, {}).get("name_acronym", f"#{num}")
                color = get_color(drivers, num)
                ax2.hist(df["speed"].values, bins=20, label=name, alpha=0.7, color=color)
            ax2.set_title("Speed Distribution")
            ax2.set_xlabel("Speed (km/h)")
            ax2.set_ylabel("Frequency")
            ax2.legend()

        for num, df in car_data.items():
            if df.empty:
                continue
            name = drivers.get(num, {}).get("name_acronym", f"#{num}")
            color = get_color(drivers, num)
            ax3.scatter(df["throttle"], df["speed"], label=name, alpha=0.6, color=color)
        ax3.set_title("Throttle vs Speed")
        ax3.set_xlabel("Throttle (%)")
        ax3.set_ylabel("Speed (km/h)")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        for num, df in car_data.items():
            if df.empty or "rpm" not in df:
                continue
            name = drivers.get(num, {}).get("name_acronym", f"#{num}")
            color = get_color(drivers, num)
            ax4.plot(df["date"], df["rpm"], label=name, color=color)
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
                color = get_color(drivers, num)
                self.ax.plot(
                    d["date"],
                    d["position"],
                    label=name,
                    marker="o",
                    markersize=3,
                    color=color,
                )
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
            color = get_color(drivers, num)
            ax1.plot(d["lap_number"], d["lap_duration"], marker="o", label=name, color=color)
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
        colors = [get_color(drivers, n) for n in best.index]
        bars = ax4.bar(names, best.values, color=colors)
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
            color = get_color(drivers, num)
            ax1.plot(
                d["date"],
                d["gap_to_leader"],
                marker="o",
                markersize=3,
                label=name,
                color=color,
            )
        ax1.set_title("Gap to Leader")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Gap (s)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        latest = df.groupby("driver_number").last()
        names = [drivers.get(n, {}).get("name_acronym", f"#{n}") for n in latest.index]
        colors = [get_color(drivers, n) for n in latest.index]
        bars = ax2.bar(names, latest["gap_to_leader"], color=colors)
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
        colors = [get_color(drivers, n) for n in df["driver_number"]]
        bars = ax1.bar(names, df["pit_duration"], color=colors)
        ax1.set_title("Pit Stop Durations")
        ax1.set_ylabel("Duration (s)")
        ax1.tick_params(axis="x", rotation=45)
        for bar, val in zip(bars, df["pit_duration"]):
            ax1.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.1f}s", ha="center", va="bottom", fontsize=8)

        for _, row in df.iterrows():
            num = row["driver_number"]
            name = drivers.get(num, {}).get("name_acronym", f"#{num}")
            color = get_color(drivers, num)
            ax2.scatter(row["date"], name, s=100, color=color)
        ax2.set_title("Pit Stop Timeline")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Driver")
        ax2.grid(True, alpha=0.3)

        self.figure.tight_layout()
        self.canvas.draw_idle()


class TyreUsageWidget(MplWidget):
    def __init__(self) -> None:
        super().__init__(10, 4)
        self.ax = self.figure.add_subplot(111)

    def update_chart(self, df: pd.DataFrame, drivers: Dict[int, Dict]) -> None:
        self.ax.clear()
        if df.empty:
            self.canvas.draw_idle()
            return
        compound_colors = {
            "SOFT": "#ff4d4d",
            "MEDIUM": "#ffd700",
            "HARD": "#f2f2f2",
            "INTERMEDIATE": "#00ff7f",
            "WET": "#1e90ff",
        }
        driver_map = {n: i for i, n in enumerate(sorted(df["driver_number"].unique()))}
        for _, row in df.iterrows():
            y = driver_map[row["driver_number"]]
            start = row.get("lap_start") or 0
            end = row.get("lap_end") or start + 1
            color = compound_colors.get(str(row.get("compound", "")).upper(), "#cccccc")
            self.ax.broken_barh([(start, end - start)], (y - 0.4, 0.8), facecolors=color)
        self.ax.set_xlabel("Lap")
        self.ax.set_yticks(list(driver_map.values()))
        labels = [drivers.get(n, {}).get("name_acronym", f"#{n}") for n in sorted(driver_map.keys())]
        self.ax.set_yticklabels(labels)
        for tick, num in zip(self.ax.get_yticklabels(), sorted(driver_map.keys())):
            color = get_color(drivers, num) or "black"
            tick.set_color(color)
        self.ax.set_title("Tyre Stints")
        self.figure.tight_layout()
        self.canvas.draw_idle()


class DriverSelectionWidget(QtWidgets.QGroupBox):
    selectionChanged = QtCore.Signal()

    def __init__(self) -> None:
        super().__init__("Driver Selection")
        self.list = QtWidgets.QListWidget()
        self.list.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.list)
        # slightly larger height so the list is easily clickable
        self.list.setMinimumHeight(140)
        self.list.itemChanged.connect(lambda _=None: self.selectionChanged.emit())

    def update_drivers(self, drivers: Dict[int, Dict]) -> None:
        """Update the list of drivers keeping current selections."""
        current_checks = {
            self.list.item(i).data(QtCore.Qt.UserRole): self.list.item(i).checkState() == QtCore.Qt.Checked
            for i in range(self.list.count())
        }
        self.list.blockSignals(True)
        self.list.clear()
        for idx, (num, info) in enumerate(sorted(drivers.items())):
            item = QtWidgets.QListWidgetItem(info.get("name_acronym", f"#{num}"))
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setData(QtCore.Qt.UserRole, num)
            color = info.get("team_colour")
            if color:
                item.setForeground(QtGui.QColor(color))
            # check previously selected or first five by default
            checked = current_checks.get(num, idx < 5)
            item.setCheckState(QtCore.Qt.Checked if checked else QtCore.Qt.Unchecked)
            self.list.addItem(item)
        self.list.blockSignals(False)

    def selected_drivers(self) -> List[int]:
        result = []
        for i in range(self.list.count()):
            item = self.list.item(i)
            if item.checkState() == QtCore.Qt.Checked:
                result.append(item.data(QtCore.Qt.UserRole))
        return result


class BestLapTableWidget(QtWidgets.QTableWidget):
    def __init__(self) -> None:
        super().__init__(0, 2)
        self.setHorizontalHeaderLabels(["Driver", "Best Lap (s)"])

    def update_table(self, df: pd.DataFrame, drivers: Dict[int, Dict]) -> None:
        if df.empty:
            self.setRowCount(0)
            return
        best = df.groupby("driver_number")["lap_duration"].min().sort_values()
        self.setRowCount(len(best))
        for row, (num, val) in enumerate(best.items()):
            name = drivers.get(num, {}).get("name_acronym", f"#{num}")
            color = get_color(drivers, num)
            item_name = QtWidgets.QTableWidgetItem(name)
            if color:
                item_name.setForeground(QtGui.QColor(color))
            self.setItem(row, 0, item_name)
            self.setItem(row, 1, QtWidgets.QTableWidgetItem(f"{val:.3f}"))
        self.resizeColumnsToContents()


class LapTimesTableWidget(QtWidgets.QTableWidget):
    def update_table(self, df: pd.DataFrame, drivers: Dict[int, Dict]) -> None:
        if df.empty:
            self.setRowCount(0)
            self.setColumnCount(0)
            return
        df = df.sort_values("lap_number")
        pivot = df.pivot_table(index="lap_number", columns="driver_number", values="lap_duration")
        headers = ["Lap"] + [drivers.get(n, {}).get("name_acronym", f"#{n}") for n in pivot.columns]
        self.setColumnCount(len(headers))
        self.setHorizontalHeaderLabels(headers)
        for idx, num in enumerate(pivot.columns, start=1):
            color = get_color(drivers, num)
            if color:
                self.horizontalHeaderItem(idx).setForeground(QtGui.QColor(color))
        self.setRowCount(len(pivot))
        for r, (lap_num, row) in enumerate(pivot.iterrows()):
            self.setItem(r, 0, QtWidgets.QTableWidgetItem(str(int(lap_num))))
            for c, num in enumerate(pivot.columns, start=1):
                val = row[num]
                text = "" if pd.isna(val) else f"{val:.3f}"
                self.setItem(r, c, QtWidgets.QTableWidgetItem(text))
        self.resizeColumnsToContents()


class BestLapThrottleWidget(MplWidget):
    def __init__(self) -> None:
        super().__init__(8, 4)
        self.ax = self.figure.add_subplot(111)

    def update_chart(
        self,
        lap_df: pd.DataFrame,
        car_data: Dict[int, pd.DataFrame],
        drivers: Dict[int, Dict],
        api: F1API,
    ) -> None:
        self.ax.clear()
        if lap_df.empty:
            self.canvas.draw_idle()
            return
        for num in drivers:
            dlap = lap_df[lap_df["driver_number"] == num]
            if dlap.empty:
                continue
            best = dlap.loc[dlap["lap_duration"].idxmin()]
            start = best["date_start"]
            end = start + pd.to_timedelta(best["lap_duration"], unit="s")
            data = car_data.get(num)
            if data is None or data.empty or data["date"].min() > start or data["date"].max() < end:
                data = api.get_car_data_range(num, start, end)
            seg = data[(data["date"] >= start) & (data["date"] <= end)]
            if seg.empty:
                continue
            t = (seg["date"] - start).dt.total_seconds()
            label = drivers.get(num, {}).get("name_acronym", f"#{num}")
            color = get_color(drivers, num)
            self.ax.plot(t, seg["throttle"], label=label, color=color)
        self.ax.set_title("Throttle % over Best Lap")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Throttle %")
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)
        self.figure.tight_layout()
        self.canvas.draw_idle()


class RaceTraceWidget(MplWidget):
    """Simple live gap chart similar to MultiViewer's race trace."""

    def __init__(self) -> None:
        super().__init__(10, 4)
        self.ax = self.figure.add_subplot(111)

    def update_chart(self, df: pd.DataFrame, drivers: Dict[int, Dict]) -> None:
        self.ax.clear()
        if df.empty:
            self.canvas.draw_idle()
            return
        for num in df["driver_number"].unique():
            d = df[df["driver_number"] == num]
            name = drivers.get(num, {}).get("name_acronym", f"#{num}")
            color = get_color(drivers, num)
            gaps = pd.to_numeric(d["gap_to_leader"], errors="coerce")
            self.ax.plot(d["date"], gaps, label=name, color=color)
        self.ax.set_title("Race Trace - Gap to Leader")
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Gap (s)")
        self.ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        self.ax.grid(True, alpha=0.3)
        self.figure.tight_layout()
        self.canvas.draw_idle()


class TelemetryOverlayWidget(QtWidgets.QGroupBox):
    """Compact telemetry readout for a single driver."""

    def __init__(self) -> None:
        super().__init__("Telemetry")
        layout = QtWidgets.QVBoxLayout(self)
        self.header = QtWidgets.QLabel("-")
        font = self.header.font()
        font.setBold(True)
        self.header.setFont(font)
        self.header.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self.header)

        grid = QtWidgets.QGridLayout()
        layout.addLayout(grid)
        self.fields = {}
        labels = ["TLA", "Position", "Speed", "Gear", "Throttle", "Brake", "DRS"]
        for row, name in enumerate(labels):
            lab = QtWidgets.QLabel(f"{name}:")
            val = QtWidgets.QLabel("-")
            grid.addWidget(lab, row, 0)
            grid.addWidget(val, row, 1)
            self.fields[name] = val

    def update_data(self, car_df: pd.DataFrame, driver: Dict) -> None:
        if car_df is None or car_df.empty or not driver:
            self.header.setText("-")
            for val in self.fields.values():
                val.setText("-")
            return
        latest = car_df.iloc[-1]
        self.header.setText(f"{driver.get('broadcast_name', '')} - {driver.get('team_name', '')}")
        color = driver.get("team_colour", "#ffffff")
        self.setStyleSheet(f"QGroupBox{{border:2px solid {color}; margin-top:1ex}}")
        self.fields["TLA"].setText(driver.get("name_acronym", ""))
        self.fields["Position"].setText(str(driver.get("position", "?")))
        self.fields["Speed"].setText(f"{int(latest.get('speed', 0))} km/h")
        self.fields["Gear"].setText(str(latest.get("n_gear", "")))
        self.fields["Throttle"].setText(f"{latest.get('throttle', 0)}%")
        self.fields["Brake"].setText(f"{latest.get('brake', 0)}%")
        self.fields["DRS"].setText(str(latest.get("drs", 0)))


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

        self.driver_selector = DriverSelectionWidget()
        layout.addWidget(self.driver_selector)
        self.driver_selector.selectionChanged.connect(
            lambda: self.fetcher.set_selected_drivers(self.driver_selector.selected_drivers())
        )

        # create view menu for toggling docks
        self.view_menu = self.menuBar().addMenu("View")
        self.docks: Dict[str, QtWidgets.QDockWidget] = {}

        def _add_dock(title: str, widget: QtWidgets.QWidget, area: QtCore.Qt.DockWidgetArea) -> None:
            dock = QtWidgets.QDockWidget(title)
            dock.setObjectName(title)
            dock.setWidget(widget)
            dock.setAllowedAreas(QtCore.Qt.AllDockWidgetAreas)
            self.addDockWidget(area, dock)
            self.view_menu.addAction(dock.toggleViewAction())
            self.docks[title] = dock

        self.telemetry_widget = TelemetryOverlayWidget()
        _add_dock("Telemetry", self.telemetry_widget, QtCore.Qt.LeftDockWidgetArea)

        self.position_widget = PositionTrackerWidget()
        _add_dock("Position", self.position_widget, QtCore.Qt.RightDockWidgetArea)

        self.speed_widget = SpeedAnalysisWidget()
        _add_dock("Speed", self.speed_widget, QtCore.Qt.RightDockWidgetArea)

        self.trace_widget = RaceTraceWidget()
        _add_dock("Race Trace", self.trace_widget, QtCore.Qt.BottomDockWidgetArea)

        self.laptime_widget = LapTimeWidget()
        _add_dock("Lap Times", self.laptime_widget, QtCore.Qt.RightDockWidgetArea)

        self.throttle_widget = BestLapThrottleWidget()
        _add_dock("Throttle", self.throttle_widget, QtCore.Qt.BottomDockWidgetArea)

        self.intervals_widget = IntervalsWidget()
        _add_dock("Intervals", self.intervals_widget, QtCore.Qt.BottomDockWidgetArea)

        self.pit_widget = PitStopWidget()
        _add_dock("Pit Stops", self.pit_widget, QtCore.Qt.BottomDockWidgetArea)

        self.weather_widget = WeatherWidget()
        _add_dock("Weather", self.weather_widget, QtCore.Qt.BottomDockWidgetArea)

        self.tyre_widget = TyreUsageWidget()
        _add_dock("Tyres", self.tyre_widget, QtCore.Qt.BottomDockWidgetArea)

        self.bestlap_table = BestLapTableWidget()
        _add_dock("Best Laps", self.bestlap_table, QtCore.Qt.RightDockWidgetArea)

        self.lap_table = LapTimesTableWidget()
        _add_dock("Lap Table", self.lap_table, QtCore.Qt.RightDockWidgetArea)

        self.resize(1400, 900)

    def update_data(self, data: dict) -> None:
        session = data.get("session", {})
        drivers = data.get("drivers", {})
        car_data = data.get("car_data", {})
        pos = data.get("position", pd.DataFrame())
        lap = data.get("lap", pd.DataFrame())
        weather = data.get("weather", pd.DataFrame())
        intervals = data.get("intervals", pd.DataFrame())
        pit = data.get("pit", pd.DataFrame())
        stints = data.get("stints", pd.DataFrame())

        start = pd.to_datetime(session.get("date_start"), utc=True, format="ISO8601")
        end = pd.to_datetime(session.get("date_end"), utc=True, format="ISO8601")
        now = datetime.now(timezone.utc)
        live = start <= now <= end
        color = "green" if live else "red"
        text = (
            f"<b>{session.get('location', 'Unknown')} - {session.get('session_name', '')}</b>"
            f" | Start: {start.strftime('%Y-%m-%d %H:%M UTC')} "
            f"<font color='{color}'>{'LIVE' if live else 'DEMO'}</font>"
        )
        self.header_label.setText(text)

        self.driver_selector.update_drivers(drivers)
        selected = self.driver_selector.selected_drivers() or list(drivers.keys())[:5]
        self.fetcher.set_selected_drivers(selected)
        filtered_car = {n: car_data.get(n, pd.DataFrame()) for n in selected}
        first_driver = selected[0] if selected else None
        driver_info = drivers.get(first_driver, {}) if first_driver else {}
        if first_driver is not None and not pos.empty:
            p = pos[pos["driver_number"] == first_driver]
            if not p.empty:
                driver_info = dict(driver_info)
                driver_info["position"] = int(p.iloc[-1].get("position", 0))
        self.telemetry_widget.update_data(car_data.get(first_driver, pd.DataFrame()), driver_info)
        self.speed_widget.update_chart(filtered_car, drivers)
        self.position_widget.update_chart(pos, drivers)
        self.laptime_widget.update_chart(lap, drivers)
        self.throttle_widget.update_chart(lap, filtered_car, {n: drivers[n] for n in selected if n in drivers}, self.api)
        self.trace_widget.update_chart(intervals[intervals["driver_number"].isin(selected)], drivers)
        self.weather_widget.update_chart(weather)
        self.pit_widget.update_chart(pit, drivers)
        self.tyre_widget.update_chart(stints, drivers)
        self.bestlap_table.update_table(lap, drivers)
        self.lap_table.update_table(lap, drivers)

        if session.get("session_type", "").lower() == "race":
            self.docks.get("Intervals", self.intervals_widget).setVisible(True)
            self.intervals_widget.update_chart(intervals, drivers)
        else:
            self.docks.get("Intervals", self.intervals_widget).setVisible(False)

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
