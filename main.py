#!/usr/bin/env python3

import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import timedelta

###############################################################################
# 1. CONFIGURATION
###############################################################################
ACCEL_CSV_PATH = "acceleration_data.csv"   # Path to your acceleration CSV
VIDEO_PATH     = "IMG_1227.mp4"            # Path to your video

# Column names in your CSV that hold the acceleration data
ACCEL_X_COL  = "X"
ACCEL_Y_COL  = "Y"
ACCEL_Z_COL  = "Z"
TIME_COL     = "Timestamp"

# Output paths
ANNOTATED_VIDEO_PATH = "annotated_video.mp4"   # Final video with overlay

# If your CSV timestamps start at time zero or are in a different time zone,
# you can manually set an offset (in seconds) to align with the video.
# Positive offset means the CSV data "starts" LATER than the video’s start.
MANUAL_OFFSET_SECONDS = 0.0

# For side-by-side comparison (optional), you can specify a second video or
# approach. This script only includes an example snippet at the end.
DO_SIDE_BY_SIDE = False
SECOND_VIDEO_PATH = "IMG_1227_alternate.mp4"
SIDE_BY_SIDE_OUTPUT_PATH = "comparison_side_by_side.mp4"

###############################################################################
# 2. LOAD AND INSPECT THE ACCELERATION DATA
###############################################################################
def load_acceleration_data(csv_path):
    df = pd.read_csv(csv_path)
    
    # Inspect first few rows
    print("=== Acceleration Data Preview ===")
    print(df.head())
    print("=== Data Info ===")
    print(df.info())
    print("=== Summary Statistics ===")
    print(df.describe())
    
    # Check for missing data
    missing_values = df.isnull().sum()
    print("\n=== Missing Values ===")
    print(missing_values)
    
    # Attempt to convert timestamps to a usable format
    try:
        # If your timestamps are numeric (e.g. Unix epoch), you might need:
        # df[TIME_COL] = pd.to_datetime(df[TIME_COL], unit='s')
        
        # If they are string-based (e.g. ISO 8601: "2023-07-01 12:00:00"):
        df[TIME_COL] = pd.to_datetime(df[TIME_COL])
        print("\nTimestamps converted to datetime successfully.")
    except Exception as e:
        print(f"\nCould not auto-convert timestamp column to datetime: {e}")
        print("Leaving timestamp as-is. Make sure it's numeric or a valid datetime.")
    
    # Forward-fill or interpolate if needed (simple approach):
    df[[ACCEL_X_COL, ACCEL_Y_COL, ACCEL_Z_COL]] = df[[ACCEL_X_COL, ACCEL_Y_COL, ACCEL_Z_COL]].interpolate(method='linear')
    
    return df

###############################################################################
# 3. PROCESS THE VIDEO (READ METADATA & PREP FOR OVERLAY)
###############################################################################
def get_video_info(video_path):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found at path: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps != 0 else 0
    
    cap.release()
    
    info = {
        'fps': fps,
        'frame_count': frame_count,
        'width': width,
        'height': height,
        'duration': duration
    }
    
    print("=== Video Info ===")
    print(f"FPS: {fps}")
    print(f"Frame Count: {frame_count}")
    print(f"Resolution: {width}x{height}")
    print(f"Duration (s): {duration:.2f}")
    
    return info

###############################################################################
# 4. SYNC TIMESTAMPS:  FIND/SET OFFSET BETWEEN DATA AND VIDEO
###############################################################################
def synchronize_timestamps(df, video_info, manual_offset=0.0):
    """
    This function demonstrates how you might apply an offset to your
    acceleration data to align it with the video’s start time.
    
    For example, if your video started at 2023-07-01 12:00:10,
    and your CSV started at 2023-07-01 12:00:05, you have a 5-second offset.
    You could manually set that offset to +5.0 (meaning the CSV started earlier).
    """
    
    # Suppose the video’s "start" time is the earliest time in the CSV, or vice versa.
    # If you know the exact start time of the video, you could parse it here.
    # For simplicity, we'll just shift by the user-provided manual offset.
    
    # If your timestamps are datetime, you can shift by a timedelta:
    df['adjusted_timestamp'] = df[TIME_COL] + pd.to_timedelta(manual_offset, unit='s')
    
    print(f"\nApplied an offset of {manual_offset} seconds to acceleration data.")
    print("Now storing these times in `adjusted_timestamp` column.\n")
    
    return df

###############################################################################
# 5. RESAMPLE OR MATCH ACCELERATION TO EACH VIDEO FRAME
###############################################################################
def get_accel_at_time(df, target_time):
    """
    Find the row in 'df' whose adjusted_timestamp is closest to 'target_time' (in seconds).
    We'll assume the earliest adjusted_timestamp is t=0 for convenience.
    """
    # Convert the df's adjusted_timestamp to a relative float (seconds) from the min
    min_time = df['adjusted_timestamp'].min()
    df['relative_sec'] = (df['adjusted_timestamp'] - min_time).dt.total_seconds()
    
    # target_time is the # of seconds since the video start, so we want
    # the row in df with 'relative_sec' closest to target_time
    idx = (df['relative_sec'] - target_time).abs().idxmin()
    row = df.loc[idx]
    return row

###############################################################################
# 6. CREATE ANNOTATED VIDEO WITH ACCELERATION OVERLAY
###############################################################################
def create_annotated_video(df, video_info, video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID', 'avc1', etc.
    out = cv2.VideoWriter(
        output_path, 
        fourcc, 
        video_info['fps'], 
        (video_info['width'], video_info['height'])
    )
    
    frame_index = 0
    min_time = df['adjusted_timestamp'].min()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Calculate the timestamp (relative) for this frame
        current_time_s = frame_index / video_info['fps']  # seconds from video start
        # Retrieve nearest acceleration data
        accel_row = get_accel_at_time(df, current_time_s)
        
        ax = accel_row[ACCEL_X_COL]
        ay = accel_row[ACCEL_Y_COL]
        az = accel_row[ACCEL_Z_COL]
        
        text = f"t={current_time_s:4.2f}s, Ax={ax:.2f}, Ay={ay:.2f}, Az={az:.2f}"
        
        # Overlay text in top-left corner
        cv2.putText(
            frame, 
            text, 
            (30, 50), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1.0, 
            (0, 255, 0), 2, cv2.LINE_AA
        )
        
        # You can draw small bar graphs or lines if you want a visual plot:
        # (Simple example: draw lines corresponding to accelerations)
        # Scale factor to visualize acceleration
        scale = 20
        center_x = 100
        center_y = 150
        
        # Draw lines for each axis acceleration
        # X-axis: from (center_x, center_y) to (center_x+scaled_value, center_y)
        x_endpoint = center_x + int(ax * scale)
        cv2.line(frame, (center_x, center_y), (x_endpoint, center_y), (255, 0, 0), 2)
        cv2.putText(frame, "Ax", (x_endpoint+5, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Y-axis: from (center_x, center_y) to (center_x, center_y - scaled_value)
        y_endpoint = center_y - int(ay * scale)
        cv2.line(frame, (center_x, center_y), (center_x, y_endpoint), (0, 255, 0), 2)
        cv2.putText(frame, "Ay", (center_x, y_endpoint-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Z-axis: diagonal or separate offset
        z_endpoint_x = center_x + int(az * scale * 0.7)
        z_endpoint_y = center_y + int(az * scale * 0.7)
        cv2.line(frame, (center_x, center_y), (z_endpoint_x, z_endpoint_y), (0, 0, 255), 2)
        cv2.putText(frame, "Az", (z_endpoint_x+5, z_endpoint_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        out.write(frame)
        frame_index += 1
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Annotated video saved to: {output_path}")

###############################################################################
# 7. (OPTIONAL) CREATE A SIDE-BY-SIDE COMPARISON
###############################################################################
def create_side_by_side_video(video_path1, video_path2, output_path):
    """
    Demonstration of how to place two videos side-by-side using OpenCV.
    NOTE: If you want an easier approach for more complex layouts,
    consider using MoviePy or FFmpeg command-line filters.
    """
    cap1 = cv2.VideoCapture(video_path1)
    cap2 = cv2.VideoCapture(video_path2)
    
    fps1 = cap1.get(cv2.CAP_PROP_FPS)
    fps2 = cap2.get(cv2.CAP_PROP_FPS)
    # For simplicity, let's assume both videos have the same FPS & resolution
    fps = min(fps1, fps2)  # or pick one
    
    width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Output dimensions: side by side horizontally
    out_width = width1 + width2
    out_height = max(height1, height2)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))
    
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if not ret1 or not ret2:
            break
        
        # If resolutions differ, we might need to resize:
        if height1 != height2:
            # Example: match everything to the smaller dimension or to one specifically
            frame2 = cv2.resize(frame2, (width1, height1))
        
        # Concatenate horizontally
        combined = cv2.hconcat([frame1, frame2])
        out.write(combined)
    
    cap1.release()
    cap2.release()
    out.release()
    print(f"Side-by-side video saved to: {output_path}")

###############################################################################
# 8. MAIN SCRIPT
###############################################################################
def main():
    # Step 1: Load and clean acceleration data
    df = load_acceleration_data(ACCEL_CSV_PATH)
    
    # Step 2: Get video metadata
    video_info = get_video_info(VIDEO_PATH)
    
    # Step 3: Synchronize timestamps (apply manual offset, if needed)
    df = synchronize_timestamps(df, video_info, MANUAL_OFFSET_SECONDS)
    
    # Step 4: Create annotated video with acceleration overlay
    create_annotated_video(df, video_info, VIDEO_PATH, ANNOTATED_VIDEO_PATH)
    
    # Step 5: (Optional) Create side-by-side comparison
    if DO_SIDE_BY_SIDE:
        # You might want to annotate the second video too (or not).
        # For now, let's assume you already have a second annotated video.
        create_side_by_side_video(ANNOTATED_VIDEO_PATH, SECOND_VIDEO_PATH, SIDE_BY_SIDE_OUTPUT_PATH)

if __name__ == "__main__":
    main()
