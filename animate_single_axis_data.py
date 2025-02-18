import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

##############################################################################
# 1. CONFIGURATION
##############################################################################
CSV_FILE       = "angularVelocity_data.csv"   # Path to your CSV
TIME_COL       = "Timestamp"                  # Column name for timestamps
Y_COL          = "Y"                          # Column name for your measured value
OUTPUT_VIDEO   = "single_axis_animation.mp4"
FPS            = 30                           # Frames per second for output video

# Date/time cutoff to start from (only keep data at or after this time):
START_TIME_STR = "2025-02-17 15:07:40"        # Adjust if needed

# Optionally limit how many seconds of data to animate (None = all):
DURATION_LIMIT = None  

# Timestamp format if your file has entries like "17-Feb-2025 15:04:17.070"
TIMESTAMP_FORMAT = "%d-%b-%Y %H:%M:%S.%f"

##############################################################################
# 2. LOAD THE DATA
##############################################################################
df = pd.read_csv(CSV_FILE)

# Convert the Timestamp column to datetime
try:
    df[TIME_COL] = pd.to_datetime(
        df[TIME_COL],
        format=TIMESTAMP_FORMAT,
        errors='raise'
    )
    print("Timestamps converted to datetime successfully.")
except Exception as e:
    print(f"Could not parse datetime: {e}")
    print("Using numeric/time values as-is.")

# 2.1 Trim data to start at START_TIME_STR
start_time = pd.to_datetime(START_TIME_STR)
df = df[df[TIME_COL] >= start_time].copy()

# 2.2 Sort by time and reset index
df.sort_values(by=TIME_COL, inplace=True)
df.reset_index(drop=True, inplace=True)

# 2.3 Shift Y so everything is positive
df[Y_COL] = df[Y_COL].abs()

# 2.4 Create a relative seconds column (time since first sample in the trimmed dataset)
if pd.api.types.is_datetime64_any_dtype(df[TIME_COL]):
    first_time = df[TIME_COL].iloc[0]
    df["time_sec"] = (df[TIME_COL] - first_time).dt.total_seconds()
else:
    # If it's numeric, just rename it to "time_sec"
    df["time_sec"] = df[TIME_COL]

# 2.5 Optionally limit data by a certain duration in seconds
if DURATION_LIMIT is not None:
    df = df[df["time_sec"] <= DURATION_LIMIT].copy()
    df.reset_index(drop=True, inplace=True)

# Extract arrays for plotting
time_data = df["time_sec"].values
y_data = df[Y_COL].values

# The total duration (in seconds) from start to finish
if len(time_data) > 1:
    total_time = time_data[-1] - time_data[0]
else:
    total_time = 0

if total_time <= 0:
    total_time = len(time_data)  # fallback if times are identical or single row

# Decide how many frames to create in the animation
num_frames = int(FPS * total_time) if total_time > 0 else len(time_data)
if num_frames < 1:
    num_frames = len(time_data)

print(f"Total time in data: {total_time:.2f} s, creating {num_frames} frames at {FPS} FPS.")

##############################################################################
# 3. SET UP THE MATPLOTLIB FIGURE
##############################################################################
fig, ax = plt.subplots()
fig.set_size_inches(8, 4.5)

# Create an empty line that we'll update frame-by-frame
(line_y,) = ax.plot([], [], color="blue", lw=2, label=Y_COL)

# Optionally a vertical line for "current time"
(current_time_line,) = ax.plot([], [], color='red', lw=2, alpha=0.5)

ax.set_title("Single-Axis Data Over Time")
ax.set_xlabel("Time (seconds)")
ax.set_ylabel(Y_COL)
ax.legend()

# Define the axis limits
y_min, y_max = np.min(y_data), np.max(y_data)
padding = 0.1 * (y_max - y_min) if (y_max - y_min) > 0 else 1
ax.set_xlim(time_data[0], time_data[-1])
ax.set_ylim(y_min - padding, y_max + padding)

##############################################################################
# 4. ANIMATION FUNCTIONS
##############################################################################
def init():
    """Initialize empty data for the line objects."""
    line_y.set_data([], [])
    current_time_line.set_data([], [])
    return (line_y, current_time_line)

def update(frame_idx):
    """
    Called at each frame to update the plot.
    We map frame_idx -> a time 't' within [time_data[0], time_data[-1]].
    """
    # Calculate current time 't'
    if num_frames > 1:
        t = time_data[0] + (time_data[-1] - time_data[0]) * (frame_idx / (num_frames - 1))
    else:
        t = time_data[0]

    # Find all data up to time t
    idx = np.searchsorted(time_data, t, side='right')

    # Update the line with data up to idx
    line_y.set_data(time_data[:idx], y_data[:idx])

    # Vertical line at t
    current_time_line.set_data([t, t], [y_min - padding, y_max + padding])

    return (line_y, current_time_line)

anim = animation.FuncAnimation(
    fig, 
    update, 
    frames=num_frames, 
    init_func=init, 
    blit=True
)

##############################################################################
# 5. SAVE THE ANIMATION TO MP4
##############################################################################
# Requires ffmpeg installed and on PATH.
try:
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=FPS, metadata=dict(artist='My Data'), bitrate=1800)
    anim.save(OUTPUT_VIDEO, writer=writer)
    print(f"Animation saved to {OUTPUT_VIDEO}")
except Exception as e:
    print("Could not write MP4. Make sure ffmpeg is installed and on your PATH.")
    print(e)
