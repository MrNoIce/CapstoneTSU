#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

##############################################################################
# CONFIGURATION
##############################################################################
# Set the file paths to your CSV files.
# Use raw string literals (r"") or double backslashes for Windows paths.
CSV_FILE1 = r"C:\Users\Jake\Documents\Workspace\WheelChairDataVideo\Caster_test_jig_video\casterapparatus_test_data.csv"
CSV_FILE2 = r"C:\Users\Jake\Documents\Workspace\WheelChairDataVideo\Uniball_Test_jig_video\uniballapparatus_test_data.csv"

# Column names in your CSV files:
TIME_COL = "Timestamp"  # Name of the timestamp column
VALUE_COL = "Y"         # Name of the measured value column

# Timestamp format in your CSV files (adjust as needed)
TIMESTAMP_FORMAT = "%d-%b-%Y %H:%M:%S.%f"

# Frequency for creating a common time axis (here 10 milliseconds = 10ms)
COMMON_FREQ = "10ms"

##############################################################################
# FUNCTIONS TO LOAD AND PROCESS DATA
##############################################################################
def load_and_rebase_csv(csv_file):
    """
    Loads a CSV file, converts the timestamp column to datetime,
    sorts the data, takes the absolute value of the measured value,
    and creates a new time column (in seconds) relative to the first timestamp.
    Returns the processed DataFrame.
    """
    df = pd.read_csv(csv_file)
    # Convert the timestamp column to datetime
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], format=TIMESTAMP_FORMAT, errors='raise')
    df.sort_values(by=TIME_COL, inplace=True)
    df.reset_index(drop=True, inplace=True)
    # Ensure values are positive by taking absolute values
    df[VALUE_COL] = df[VALUE_COL].abs()
    # Create a new column: time in seconds relative to the first timestamp
    first_time = df[TIME_COL].iloc[0]
    df["time_sec"] = (df[TIME_COL] - first_time).dt.total_seconds()
    return df

# Load and rebase both datasets so that each starts at 0 seconds.
df1 = load_and_rebase_csv(CSV_FILE1)
df2 = load_and_rebase_csv(CSV_FILE2)

# Determine the common time window based on the durations of the two datasets.
common_start = 0.0
common_end = min(df1["time_sec"].max(), df2["time_sec"].max())
print(f"Common time range: {common_start} s to {common_end:.2f} s")

# Trim each DataFrame to the common time window.
df1 = df1[df1["time_sec"] <= common_end].copy()
df2 = df2[df2["time_sec"] <= common_end].copy()
df1.reset_index(drop=True, inplace=True)
df2.reset_index(drop=True, inplace=True)

##############################################################################
# INTERPOLATION ON A COMMON TIME AXIS
##############################################################################
# Create a common time axis from 0 to common_end with a high-resolution frequency.
common_time_axis = pd.date_range(start=pd.to_datetime(common_start, unit='s', origin=pd.Timestamp("1970-01-01")),
                                 end=pd.to_datetime(common_end, unit='s', origin=pd.Timestamp("1970-01-01")),
                                 freq=COMMON_FREQ)

# For interpolation, convert the common time axis to float seconds.
# Since our datasets have time_sec already, we’ll create an array spanning 0 to common_end.
common_time_sec = np.linspace(common_start, common_end, len(common_time_axis))

def interpolate_dataset(df):
    """
    Interpolates the VALUE_COL data onto the common time axis.
    Uses numpy.interp which requires numeric arrays.
    """
    t = df["time_sec"].values
    y = df[VALUE_COL].values
    # Perform linear interpolation
    y_interp = np.interp(common_time_sec, t, y)
    return y_interp

y1_interp = interpolate_dataset(df1)
y2_interp = interpolate_dataset(df2)

# Create a combined DataFrame for convenience.
df_common = pd.DataFrame({
    "time_sec": common_time_sec,
    VALUE_COL + "_1": y1_interp,
    VALUE_COL + "_2": y2_interp
})

##############################################################################
# STATIC OVERLAY GRAPH
##############################################################################
def create_static_overlay():
    plt.figure(figsize=(10, 6))
    plt.plot(df_common["time_sec"], df_common[VALUE_COL + "_1"], label="Caster Apparatus", color='blue')
    plt.plot(df_common["time_sec"], df_common[VALUE_COL + "_2"], label="Uniball Apparatus", color='orange')
    plt.xlabel("Time (s)")
    plt.ylabel(VALUE_COL)
    plt.title("Static Overlay of Two Wheel Test Datasets")
    plt.legend()
    plt.tight_layout()
    plt.savefig("static_overlay.png")
    plt.show()

##############################################################################
# DYNAMIC (ANIMATED) OVERLAY VIDEO
##############################################################################
def create_dynamic_overlay():
    fig, ax = plt.subplots(figsize=(10, 6))
    line1, = ax.plot([], [], label="Caster Apparatus", color='blue')
    line2, = ax.plot([], [], label="Uniball Apparatus", color='orange')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(VALUE_COL)
    ax.set_title("Dynamic Overlay of Two Wheel Test Datasets")
    ax.legend()

    time_data = df_common["time_sec"].values
    y1_data = df_common[VALUE_COL + "_1"].values
    y2_data = df_common[VALUE_COL + "_2"].values
    num_frames = len(time_data)

    ax.set_xlim(time_data[0], time_data[-1])
    y_min = min(y1_data.min(), y2_data.min())
    y_max = max(y1_data.max(), y2_data.max())
    ax.set_ylim(y_min, y_max)

    def init():
        line1.set_data([], [])
        line2.set_data([], [])
        return line1, line2

    def animate(i):
        line1.set_data(time_data[:i], y1_data[:i])
        line2.set_data(time_data[:i], y2_data[:i])
        return line1, line2

    ani = animation.FuncAnimation(fig, animate, frames=num_frames, init_func=init, blit=True)

    try:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=30, metadata=dict(artist='Your Name'), bitrate=1800)
        ani.save("dynamic_overlay.mp4", writer=writer)
        print("Dynamic overlay video saved as dynamic_overlay.mp4")
    except Exception as e:
        print("Error saving dynamic overlay video:", e)
    plt.show()

##############################################################################
# MAIN
##############################################################################
if __name__ == "__main__":
    print("Creating static overlay graph...")
    create_static_overlay()
    print("Creating dynamic overlay video...")
    create_dynamic_overlay()
