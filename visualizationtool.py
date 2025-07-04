import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime, timedelta
import time
import json
from typing import Dict, List, Optional
import warnings

warnings.filterwarnings('ignore')


class F1LiveDashboard:
    """
    F1 Live Session Visualization Tool using OpenF1 API

    This tool provides real-time visualization of F1 session data including:
    - Speed analysis and telemetry
    - Position tracking
    - Lap time comparisons
    - Weather conditions
    - Driver intervals
    - Pit stop analysis
    """

    def __init__(self):
        self.base_url = "https://api.openf1.org/v1"
        self.session_key = None
        self.meeting_key = None
        self.drivers = {}
        self.colors = plt.cm.tab20(np.linspace(0, 1, 20))

        # Setup plot style
        plt.style.use('dark_background')
        sns.set_palette("husl")

    def get_latest_session(self) -> Dict:
        """Get the latest/current F1 session"""
        try:
            response = requests.get(f"{self.base_url}/sessions?session_key=latest")
            if response.status_code == 200:
                session_data = response.json()
                if session_data:
                    self.session_key = session_data[0]['session_key']
                    self.meeting_key = session_data[0]['meeting_key']
                    return session_data[0]
            return {}
        except Exception as e:
            print(f"Error fetching session: {e}")
            return {}

    def get_session_drivers(self) -> Dict:
        """Get all drivers in the current session"""
        if not self.session_key:
            return {}

        try:
            response = requests.get(f"{self.base_url}/drivers?session_key={self.session_key}")
            if response.status_code == 200:
                drivers_data = response.json()
                self.drivers = {d['driver_number']: d for d in drivers_data}
                return self.drivers
        except Exception as e:
            print(f"Error fetching drivers: {e}")
            return {}

    def get_car_data(self, driver_number: int = None, limit: int = 100) -> pd.DataFrame:
        """
        Get car telemetry data (speed, throttle, brake, etc.)
        Updates at ~3.7 Hz during sessions
        """
        if not self.session_key:
            return pd.DataFrame()

        url = f"{self.base_url}/car_data?session_key={self.session_key}"
        if driver_number:
            url += f"&driver_number={driver_number}"

        try:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                if data:
                    df = pd.DataFrame(data)
                    df['date'] = pd.to_datetime(df['date'])
                    return df.tail(limit)
            return pd.DataFrame()
        except Exception as e:
            print(f"Error fetching car data: {e}")
            return pd.DataFrame()

    def get_position_data(self) -> pd.DataFrame:
        """Get current driver positions"""
        if not self.session_key:
            return pd.DataFrame()

        try:
            response = requests.get(f"{self.base_url}/position?session_key={self.session_key}")
            if response.status_code == 200:
                data = response.json()
                if data:
                    df = pd.DataFrame(data)
                    df['date'] = pd.to_datetime(df['date'])
                    return df
            return pd.DataFrame()
        except Exception as e:
            print(f"Error fetching position data: {e}")
            return pd.DataFrame()

    def get_lap_data(self) -> pd.DataFrame:
        """Get lap times and sector information"""
        if not self.session_key:
            return pd.DataFrame()

        try:
            response = requests.get(f"{self.base_url}/laps?session_key={self.session_key}")
            if response.status_code == 200:
                data = response.json()
                if data:
                    df = pd.DataFrame(data)
                    df['date_start'] = pd.to_datetime(df['date_start'])
                    return df
            return pd.DataFrame()
        except Exception as e:
            print(f"Error fetching lap data: {e}")
            return pd.DataFrame()

    def get_weather_data(self) -> pd.DataFrame:
        """Get weather conditions (updated every minute)"""
        if not self.session_key:
            return pd.DataFrame()

        try:
            response = requests.get(f"{self.base_url}/weather?session_key={self.session_key}")
            if response.status_code == 200:
                data = response.json()
                if data:
                    df = pd.DataFrame(data)
                    df['date'] = pd.to_datetime(df['date'])
                    return df
            return pd.DataFrame()
        except Exception as e:
            print(f"Error fetching weather data: {e}")
            return pd.DataFrame()

    def get_intervals(self) -> pd.DataFrame:
        """Get real-time intervals between drivers (race only)"""
        if not self.session_key:
            return pd.DataFrame()

        try:
            response = requests.get(f"{self.base_url}/intervals?session_key={self.session_key}")
            if response.status_code == 200:
                data = response.json()
                if data:
                    df = pd.DataFrame(data)
                    df['date'] = pd.to_datetime(df['date'])
                    return df
            return pd.DataFrame()
        except Exception as e:
            print(f"Error fetching intervals: {e}")
            return pd.DataFrame()

    def get_pit_data(self) -> pd.DataFrame:
        """Get pit stop information"""
        if not self.session_key:
            return pd.DataFrame()

        try:
            response = requests.get(f"{self.base_url}/pit?session_key={self.session_key}")
            if response.status_code == 200:
                data = response.json()
                if data:
                    df = pd.DataFrame(data)
                    df['date'] = pd.to_datetime(df['date'])
                    return df
            return pd.DataFrame()
        except Exception as e:
            print(f"Error fetching pit data: {e}")
            return pd.DataFrame()

    def plot_speed_analysis(self, driver_numbers: List[int] = None):
        """Plot speed analysis for selected drivers"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Speed Analysis Dashboard', fontsize=16, fontweight='bold')

        if driver_numbers is None:
            driver_numbers = list(self.drivers.keys())[:5]  # Top 5 drivers

        # Plot 1: Speed over time
        ax1 = axes[0, 0]
        for i, driver_num in enumerate(driver_numbers):
            car_data = self.get_car_data(driver_num, limit=50)
            if not car_data.empty:
                driver_name = self.drivers.get(driver_num, {}).get('name_acronym', f'#{driver_num}')
                ax1.plot(car_data['date'], car_data['speed'],
                         label=f'{driver_name}', color=self.colors[i], alpha=0.8)

        ax1.set_title('Speed Over Time')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Speed (km/h)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Speed distribution
        ax2 = axes[0, 1]
        all_speeds = []
        labels = []
        for driver_num in driver_numbers:
            car_data = self.get_car_data(driver_num, limit=100)
            if not car_data.empty:
                all_speeds.append(car_data['speed'].values)
                labels.append(self.drivers.get(driver_num, {}).get('name_acronym', f'#{driver_num}'))

        if all_speeds:
            ax2.hist(all_speeds, bins=20, label=labels, alpha=0.7)
            ax2.set_title('Speed Distribution')
            ax2.set_xlabel('Speed (km/h)')
            ax2.set_ylabel('Frequency')
            ax2.legend()

        # Plot 3: Throttle vs Speed
        ax3 = axes[1, 0]
        for i, driver_num in enumerate(driver_numbers):
            car_data = self.get_car_data(driver_num, limit=100)
            if not car_data.empty:
                driver_name = self.drivers.get(driver_num, {}).get('name_acronym', f'#{driver_num}')
                ax3.scatter(car_data['throttle'], car_data['speed'],
                            alpha=0.6, label=f'{driver_name}', color=self.colors[i])

        ax3.set_title('Throttle vs Speed')
        ax3.set_xlabel('Throttle (%)')
        ax3.set_ylabel('Speed (km/h)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: RPM Analysis
        ax4 = axes[1, 1]
        for i, driver_num in enumerate(driver_numbers):
            car_data = self.get_car_data(driver_num, limit=50)
            if not car_data.empty:
                driver_name = self.drivers.get(driver_num, {}).get('name_acronym', f'#{driver_num}')
                ax4.plot(car_data['date'], car_data['rpm'],
                         label=f'{driver_name}', color=self.colors[i], alpha=0.8)

        ax4.set_title('RPM Over Time')
        ax4.set_xlabel('Time')
        ax4.set_ylabel('RPM')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_position_tracker(self):
        """Plot driver positions over time"""
        position_data = self.get_position_data()
        if position_data.empty:
            print("No position data available")
            return

        fig, ax = plt.subplots(figsize=(12, 8))

        # Get unique drivers and create position timeline
        for driver_num in position_data['driver_number'].unique():
            driver_pos = position_data[position_data['driver_number'] == driver_num]
            driver_name = self.drivers.get(driver_num, {}).get('name_acronym', f'#{driver_num}')

            ax.plot(driver_pos['date'], driver_pos['position'],
                    marker='o', label=f'{driver_name}', linewidth=2, markersize=4)

        ax.set_title('Driver Position Tracker', fontsize=16, fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel('Position')
        ax.invert_yaxis()  # Position 1 at top
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_lap_times(self):
        """Plot lap time analysis"""
        lap_data = self.get_lap_data()
        if lap_data.empty:
            print("No lap data available")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Lap Time Analysis', fontsize=16, fontweight='bold')

        # Plot 1: Lap times by driver
        ax1 = axes[0, 0]
        for i, driver_num in enumerate(lap_data['driver_number'].unique()):
            driver_laps = lap_data[lap_data['driver_number'] == driver_num]
            driver_name = self.drivers.get(driver_num, {}).get('name_acronym', f'#{driver_num}')

            ax1.plot(driver_laps['lap_number'], driver_laps['lap_duration'],
                     marker='o', label=f'{driver_name}', color=self.colors[i % len(self.colors)])

        ax1.set_title('Lap Times by Driver')
        ax1.set_xlabel('Lap Number')
        ax1.set_ylabel('Lap Duration (seconds)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Sector times comparison
        ax2 = axes[0, 1]
        sectors = ['duration_sector_1', 'duration_sector_2', 'duration_sector_3']
        sector_means = [lap_data[sector].mean() for sector in sectors]

        ax2.bar(['Sector 1', 'Sector 2', 'Sector 3'], sector_means,
                color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax2.set_title('Average Sector Times')
        ax2.set_ylabel('Duration (seconds)')

        # Plot 3: Speed traps
        ax3 = axes[1, 0]
        speed_cols = ['i1_speed', 'i2_speed', 'st_speed']
        for i, speed_col in enumerate(speed_cols):
            if speed_col in lap_data.columns:
                ax3.hist(lap_data[speed_col].dropna(), bins=15, alpha=0.7,
                         label=f'{speed_col.replace("_", " ").title()}')

        ax3.set_title('Speed Trap Distribution')
        ax3.set_xlabel('Speed (km/h)')
        ax3.set_ylabel('Frequency')
        ax3.legend()

        # Plot 4: Best lap times
        ax4 = axes[1, 1]
        best_laps = lap_data.groupby('driver_number')['lap_duration'].min().sort_values()
        driver_names = [self.drivers.get(num, {}).get('name_acronym', f'#{num}')
                        for num in best_laps.index]

        bars = ax4.bar(driver_names, best_laps.values, color=self.colors[:len(best_laps)])
        ax4.set_title('Best Lap Times')
        ax4.set_ylabel('Lap Duration (seconds)')
        ax4.tick_params(axis='x', rotation=45)

        # Add value labels on bars
        for bar, value in zip(bars, best_laps.values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{value:.3f}s', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        plt.show()

    def plot_weather_conditions(self):
        """Plot weather conditions over time"""
        weather_data = self.get_weather_data()
        if weather_data.empty:
            print("No weather data available")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Weather Conditions', fontsize=16, fontweight='bold')

        # Plot 1: Temperature
        ax1 = axes[0, 0]
        ax1.plot(weather_data['date'], weather_data['air_temperature'],
                 label='Air Temp', color='#FF6B6B', linewidth=2)
        ax1.plot(weather_data['date'], weather_data['track_temperature'],
                 label='Track Temp', color='#FF9F43', linewidth=2)
        ax1.set_title('Temperature')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Temperature (°C)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Humidity and Pressure
        ax2 = axes[0, 1]
        ax2_twin = ax2.twinx()

        line1 = ax2.plot(weather_data['date'], weather_data['humidity'],
                         color='#4ECDC4', linewidth=2, label='Humidity')
        line2 = ax2_twin.plot(weather_data['date'], weather_data['pressure'],
                              color='#45B7D1', linewidth=2, label='Pressure')

        ax2.set_title('Humidity & Pressure')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Humidity (%)', color='#4ECDC4')
        ax2_twin.set_ylabel('Pressure (mbar)', color='#45B7D1')

        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax2.legend(lines, labels, loc='upper left')

        # Plot 3: Wind
        ax3 = axes[1, 0]
        ax3.plot(weather_data['date'], weather_data['wind_speed'],
                 color='#26D0CE', linewidth=2)
        ax3.set_title('Wind Speed')
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Wind Speed (m/s)')
        ax3.grid(True, alpha=0.3)

        # Plot 4: Wind direction (polar plot)
        ax4 = plt.subplot(2, 2, 4, projection='polar')
        wind_dirs = np.radians(weather_data['wind_direction'])
        ax4.scatter(wind_dirs, weather_data['wind_speed'],
                    c=weather_data['wind_speed'], cmap='viridis', alpha=0.7)
        ax4.set_title('Wind Direction & Speed')
        ax4.set_ylim(0, weather_data['wind_speed'].max() * 1.1)

        plt.tight_layout()
        plt.show()

    def plot_intervals_race(self):
        """Plot real-time race intervals (race sessions only)"""
        intervals = self.get_intervals()
        if intervals.empty:
            print("No interval data available (only available during races)")
            return

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Race Intervals', fontsize=16, fontweight='bold')

        # Plot 1: Gap to leader
        ax1 = axes[0]
        for driver_num in intervals['driver_number'].unique():
            driver_intervals = intervals[intervals['driver_number'] == driver_num]
            driver_name = self.drivers.get(driver_num, {}).get('name_acronym', f'#{driver_num}')

            ax1.plot(driver_intervals['date'], driver_intervals['gap_to_leader'],
                     label=f'{driver_name}', marker='o', markersize=3)

        ax1.set_title('Gap to Leader')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Gap (seconds)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Current intervals
        ax2 = axes[1]
        latest_intervals = intervals.groupby('driver_number').last()
        driver_names = [self.drivers.get(num, {}).get('name_acronym', f'#{num}')
                        for num in latest_intervals.index]

        bars = ax2.bar(driver_names, latest_intervals['gap_to_leader'],
                       color=self.colors[:len(latest_intervals)])
        ax2.set_title('Current Gap to Leader')
        ax2.set_ylabel('Gap (seconds)')
        ax2.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.show()

    def plot_pit_analysis(self):
        """Plot pit stop analysis"""
        pit_data = self.get_pit_data()
        if pit_data.empty:
            print("No pit stop data available")
            return

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Pit Stop Analysis', fontsize=16, fontweight='bold')

        # Plot 1: Pit stop durations
        ax1 = axes[0]
        driver_names = [self.drivers.get(num, {}).get('name_acronym', f'#{num}')
                        for num in pit_data['driver_number']]

        bars = ax1.bar(driver_names, pit_data['pit_duration'],
                       color=self.colors[:len(pit_data)])
        ax1.set_title('Pit Stop Durations')
        ax1.set_ylabel('Duration (seconds)')
        ax1.tick_params(axis='x', rotation=45)

        # Add value labels on bars
        for bar, value in zip(bars, pit_data['pit_duration']):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{value:.1f}s', ha='center', va='bottom', fontsize=9)

        # Plot 2: Pit stops timeline
        ax2 = axes[1]
        for i, (_, pit_stop) in enumerate(pit_data.iterrows()):
            driver_name = self.drivers.get(pit_stop['driver_number'], {}).get('name_acronym',
                                                                              f"#{pit_stop['driver_number']}")
            ax2.scatter(pit_stop['date'], driver_name, s=100,
                        c=self.colors[i % len(self.colors)], alpha=0.8)

        ax2.set_title('Pit Stop Timeline')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Driver')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def live_monitor(self, refresh_interval: int = 30):
        """
        Live monitoring dashboard that refreshes every N seconds
        """
        print(f"Starting live monitor (refresh every {refresh_interval} seconds)")
        print("Press Ctrl+C to stop")

        try:
            while True:
                # Clear screen (works in most terminals)
                print("\033[2J\033[H")

                # Get session info
                session = self.get_latest_session()
                if session:
                    print(f"Session: {session.get('session_name', 'Unknown')}")
                    print(f"Location: {session.get('location', 'Unknown')}")
                    print(f"Date: {session.get('date_start', 'Unknown')}")
                    print("-" * 50)

                # Get and display current data
                self.get_session_drivers()

                # Display multiple visualizations
                self.plot_speed_analysis()
                self.plot_position_tracker()
                self.plot_lap_times()
                self.plot_weather_conditions()
                self.plot_intervals_race()
                self.plot_pit_analysis()

                print(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")
                print(f"Next refresh in {refresh_interval} seconds...")

                time.sleep(refresh_interval)

        except KeyboardInterrupt:
            print("\nLive monitor stopped.")

    def run_dashboard(self):
        """Run the complete dashboard"""
        print("F1 Live Session Dashboard")
        print("=" * 40)

        # Initialize session
        session = self.get_latest_session()
        if not session:
            print("No active session found. Using demo data...")
            # For demo, use a recent session
            self.session_key = 9165  # Example session key
            self.meeting_key = 1219  # Example meeting key
        else:
            print(f"Active session: {session.get('session_name', 'Unknown')}")
            print(f"Location: {session.get('location', 'Unknown')}")

        # Get drivers
        self.get_session_drivers()
        if not self.drivers:
            print("No drivers found in session")
            return

        print(f"Found {len(self.drivers)} drivers")
        print("-" * 40)

        # Generate all visualizations
        print("1. Speed Analysis...")
        self.plot_speed_analysis()

        print("2. Position Tracker...")
        self.plot_position_tracker()

        print("3. Lap Times...")
        self.plot_lap_times()

        print("4. Weather Conditions...")
        self.plot_weather_conditions()

        print("5. Race Intervals...")
        self.plot_intervals_race()

        print("6. Pit Analysis...")
        self.plot_pit_analysis()

        print("\nDashboard complete!")

        # Ask if user wants live monitoring
        response = input("\nWould you like to start live monitoring? (y/n): ")
        if response.lower() == 'y':
            self.live_monitor()


# Usage example
if __name__ == "__main__":
    # Create dashboard instance
    dashboard = F1LiveDashboard()

    # Run the complete dashboard
    dashboard.run_dashboard()

    # Or run specific analyses:
    # dashboard.plot_speed_analysis([1, 16, 55])  # Specific drivers
    # dashboard.plot_weather_conditions()
    # dashboard.live_monitor(refresh_interval=60)  # Live monitoring every 60 seconds