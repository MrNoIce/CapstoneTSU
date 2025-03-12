#!/usr/bin/env python3

import cv2

def create_side_by_side_video(
    video_path1, 
    video_path2, 
    output_path, 
    start_offset_video1=0.0,
    start_offset_video2=0.0
):
    """
    Places two videos side by side horizontally.
    Optionally skip 'start_offset_video1' seconds in the first video,
    and/or skip 'start_offset_video2' seconds in the second video.
    Preserves the second video's aspect ratio if their heights differ.
    """

    cap1 = cv2.VideoCapture(video_path1)
    cap2 = cv2.VideoCapture(video_path2)

    # Get FPS for each video
    fps1 = cap1.get(cv2.CAP_PROP_FPS)
    fps2 = cap2.get(cv2.CAP_PROP_FPS)
    if fps1 <= 0:
        fps1 = 30
    if fps2 <= 0:
        fps2 = 30

    # We'll use the lower FPS so both videos read frames in "sync"
    fps = min(fps1, fps2)

    # Skip frames on video 1 if requested
    if start_offset_video1 > 0:
        skip_frames_1 = int(start_offset_video1 * fps1)
        cap1.set(cv2.CAP_PROP_POS_FRAMES, skip_frames_1)
        print(f"[INFO] Skipping first {skip_frames_1} frames on video 1 ({start_offset_video1} s)")

    # Skip frames on video 2 if requested
    if start_offset_video2 > 0:
        skip_frames_2 = int(start_offset_video2 * fps2)
        cap2.set(cv2.CAP_PROP_POS_FRAMES, skip_frames_2)
        print(f"[INFO] Skipping first {skip_frames_2} frames on video 2 ({start_offset_video2} s)")

    # Read the first frames to figure out final output dimensions
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    # If either can't read even one frame, just abort
    if not ret1 or not ret2:
        print("[ERROR] One of the videos has no frames after skipping offsets.")
        cap1.release()
        cap2.release()
        return

    # Match heights without distorting the second frame
    h1, w1, _ = frame1.shape
    h2, w2, _ = frame2.shape

    if h1 != h2:
        ratio = h1 / float(h2)
        new_w2 = int(w2 * ratio)
        frame2 = cv2.resize(frame2, (new_w2, h1))

    # Now that we have two frames with matched height, compute final dims
    h2, w2, _ = frame2.shape  # update after resize
    out_width = w1 + w2
    out_height = max(h1, h2)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))

    # Write the first combined frame
    combined = cv2.hconcat([frame1, frame2])
    out.write(combined)

    # Main loop: read subsequent frames until one video ends
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            # No more frames in one of the videos
            break

        # Check if we need to resize again for each new frame (usually yes)
        h1, w1, _ = frame1.shape
        h2, w2, _ = frame2.shape
        if h1 != h2:
            ratio = h1 / float(h2)
            new_w2 = int(w2 * ratio)
            frame2 = cv2.resize(frame2, (new_w2, h1))

        # Concatenate horizontally
        combined = cv2.hconcat([frame1, frame2])
        out.write(combined)

    cap1.release()
    cap2.release()
    out.release()
    print(f"[INFO] Side-by-side video saved to: {output_path}")

################################################################
# Example usage
################################################################
if __name__ == "__main__":
    # For example, we skip first 15 seconds of 'IMG_1227.mp4'
    # and start 'single_axis_animation.mp4' at t=0
    create_side_by_side_video(
        "IMG_1227.mp4",
        "single_axis_animation_2.mp4",
        "comparison_side_by_side.mp4",
        start_offset_video1=14.9,   # skip 15s on the first video
        start_offset_video2=0.0     # skip 0s on the second
    )
