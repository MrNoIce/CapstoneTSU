import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Configuration: update these file names and column names as needed
CSV_FILE1 = "wheel_test1.csv"   # First dataset
CSV_FILE2 = "wheel_test2.csv"   # Second dataset

TIME_COL = "Timestamp"          # Column with timestamps (assumed in a parseable format)
VALUE_COL = "Y"                 # Column with measurement values

# Timestamp format (adjust as needed)
TIMESTAMP_FORMAT = "%d-%b-%Y %H:%M:%S.%f"

# Load datasets
df1 = pd.read_csv(CSV_FILE1)
df2 = pd.read_csv(CSV_FILE2)

# Convert timestamp columns to datetime
df1[TIME_COL] = pd.to_datetime(df1[TIME_COL], format=TIMESTAMP_FORMAT)
df2[TIME_COL] = pd.to_datetime(df2[TIME_COL], format=TIMESTAMP_FORMAT)

# Optionally, you may trim the data to a common time window (if needed):
# For example, define a common start and stop time based on known events.
common_start = max(df1[TIME_COL].min(), df2[TIME_COL].min())
common_end = min(df1[TIME_COL].max(), df2[TIME_COL].max())

df1 = df1[(df1[TIME_COL] >= common_start) & (df1[TIME_COL] <= common_end)].copy()
df2 = df2[(df2[TIME_COL] >= common_start) & (df2[TIME_COL] <= common_end)].copy()

# Create a common time axis for overlaying if they are sampled at different rates.
# For example, create a time series at 100 Hz over the common time window.
time_axis = pd.date_range(start=common_start, end=common_end, freq='10L')  # 10L = 10 milliseconds

# Interpolate both datasets onto the common time axis:
df1_interp = pd.DataFrame({
    TIME_COL: time_axis,
    VALUE_COL: np.interp(time_axis.astype(np.int64), 
                         df1[TIME_COL].astype(np.int64), 
                         df1[VALUE_COL])
})
df2_interp = pd.DataFrame({
    TIME_COL: time_axis,
    VALUE_COL: np.interp(time_axis.astype(np.int64), 
                         df2[TIME_COL].astype(np.int64), 
                         df2[VALUE_COL])
})

# Plot the two datasets on the same static graph
plt.figure(figsize=(10, 5))
plt.plot(df1_interp[TIME_COL], df1_interp[VALUE_COL], label="Wheel Test 1", color='blue')
plt.plot(df2_interp[TIME_COL], df2_interp[VALUE_COL], label="Wheel Test 2", color='orange')

# Highlight the blip if needed:
# You can add markers or vertical lines if you know the approximate time of the event.
# For example, if the blip is around common_start + 5 seconds:
blip_time = common_start + pd.Timedelta(seconds=5)
plt.axvline(blip_time, color='red', linestyle='--', label='Blip Time')

plt.xlabel("Time")
plt.ylabel("Measurement Value")
plt.title("Overlay of Two Wheel Test Datasets")
plt.legend()
plt.tight_layout()
plt.show()
