import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

##############################################################################
# 1. CONFIGURATION
##############################################################################
CSV_FILE       = "acceleration_data.csv"  # Path to your CSV
TIME_COL       = "Timestamp"              # Column name for timestamps
ACCEL_X_COL    = "X"                      # Column name for X acceleration
ACCEL_Y_COL    = "Y"                      # Column name for Y acceleration
ACCEL_Z_COL    = "Z"                      # Column name for Z acceleration
OUTPUT_VIDEO   = "phone_data_animation.mp4"
FPS            = 30                       # Frames per second in the output
DURATION_LIMIT = None                     # e.g., 10 for 10 seconds, or None for all data

##############################################################################
# 2. READ AND PREP THE DATA
##############################################################################
df = pd.read_csv(CSV_FILE)

# If your timestamps are strings like "17-Feb-2025 15:04:17.070",
# parse them with the correct datetime format:
try:
    df[TIME_COL] = pd.to_datetime(
        df[TIME_COL],
        format='%d-%b-%Y %H:%M:%S.%f',  # adjust if needed
        errors='raise'
    )
except Exception as e:
    print(f"Could not parse datetime: {e}")
    print("Using raw values or numeric index as time instead.")

# Sort by timestamp just in case
df.sort_values(by=TIME_COL, inplace=True)
df.reset_index(drop=True, inplace=True)

# Convert your timestamps to relative seconds (if you want a 0-based time)
if pd.api.types.is_datetime64_any_dtype(df[TIME_COL]):
    start_time = df[TIME_COL].iloc[0]
    df['time_sec'] = (df[TIME_COL] - start_time).dt.total_seconds()
else:
    # If 'timestamp' is already numeric, rename it to 'time_sec'
    df['time_sec'] = df[TIME_COL]

if DURATION_LIMIT:
    df = df[df['time_sec'] <= DURATION_LIMIT].copy()
    df.reset_index(drop=True, inplace=True)

# Extract arrays for plotting
time_data = df['time_sec'].values
x_data = df[ACCEL_X_COL].values
y_data = df[ACCEL_Y_COL].values
z_data = df[ACCEL_Z_COL].values

# Let's define how many frames we want in our animation.
# If we have an entire dataset with length N, we can map that onto the desired FPS.
# For example, we might have 10 seconds of data at 100 Hz = 1000 samples.
# We'll create an animation that "plays" in real time at 30 FPS.

total_time = time_data[-1] - time_data[0]  # in seconds
if total_time <= 0:
    total_time = len(time_data)  # fallback if times are weird or the same

# We'll define a total number of frames = FPS * total_time
# If your data is too large or sampling is too high, you can reduce it.
num_frames = int(FPS * total_time) if total_time > 0 else len(time_data)
if num_frames < 1:
    num_frames = len(time_data)

print(f"Total time in data: {total_time:.2f} s, creating {num_frames} frames at {FPS} FPS.")


##############################################################################
# 3. CREATE THE FIGURE AND ANIMATION
##############################################################################
fig, ax = plt.subplots()
fig.set_size_inches(8, 4.5)

# We'll plot all data as lines for X, Y, Z, but reveal them gradually.
# Create empty line objects for each axis:
(line_x,) = ax.plot([], [], label='Accel X', color='red')
(line_y,) = ax.plot([], [], label='Accel Y', color='green')
(line_z,) = ax.plot([], [], label='Accel Z', color='blue')

# We also might want a vertical line indicating the "current" time
(current_time_line,) = ax.plot([], [], color='black', lw=2, alpha=0.4)

ax.set_title("Acceleration Over Time")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Acceleration (m/s^2) or G?")
ax.legend()

# Decide on y-limits. You might want to auto-calc or pick something standard:
y_min = min(x_data.min(), y_data.min(), z_data.min()) * 1.1
y_max = max(x_data.max(), y_data.max(), z_data.max()) * 1.1
ax.set_xlim(time_data[0], time_data[-1])
ax.set_ylim(y_min, y_max)


def init():
    """Initialize the lines so they start empty."""
    line_x.set_data([], [])
    line_y.set_data([], [])
    line_z.set_data([], [])
    current_time_line.set_data([], [])
    return (line_x, line_y, line_z, current_time_line)


def update(frame_idx):
    """
    This function is called for each 'frame' in the animation.
    We pick the portion of data up to 'frame_idx' in time, or
    find the actual time that corresponds to this frame.
    """
    # Map frame_idx -> a real time "t" in seconds:
    # For a total of num_frames frames from time_data[0] to time_data[-1]:
    t = time_data[0] + (time_data[-1] - time_data[0]) * (frame_idx / (num_frames - 1))

    # We want to plot data from the start up to time t.
    # We'll find all indices in time_data that are <= t.
    idx = np.searchsorted(time_data, t, side='right')
    # idx is how far into the data we go.

    # For lines, we show everything up to idx:
    line_x.set_data(time_data[:idx], x_data[:idx])
    line_y.set_data(time_data[:idx], y_data[:idx])
    line_z.set_data(time_data[:idx], z_data[:idx])

    # The "current time" line is just a vertical line at t
    current_time_line.set_data([t, t], [y_min, y_max])

    return (line_x, line_y, line_z, current_time_line)


# Use Matplotlib's FuncAnimation to produce a dynamic plot
anim = animation.FuncAnimation(
    fig,
    update,
    frames=num_frames,
    init_func=init,
    blit=True  # or False if you have issues
)

##############################################################################
# 4. SAVE ANIMATION AS A VIDEO
##############################################################################
# We'll need FFmpeg installed on your system for 'ffmpeg' writer, or you can use 'pillow' for GIF.
# But MP4 is generally best for video.
try:
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=FPS, metadata=dict(artist='Me'), bitrate=1800)
    anim.save(OUTPUT_VIDEO, writer=writer)
    print(f"Animation saved to {OUTPUT_VIDEO}")
except Exception as e:
    print("FFmpeg might not be installed or not found. Install FFmpeg or set the path properly.")
    print(e)
