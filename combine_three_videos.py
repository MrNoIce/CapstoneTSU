#!/usr/bin/env python3
import cv2

def resize_to_height(frame, target_height):
    h, w, _ = frame.shape
    if h == target_height:
        return frame
    ratio = target_height / float(h)
    new_width = int(w * ratio)
    return cv2.resize(frame, (new_width, target_height))

def create_three_video_composite(video_path1, video_path2, video_path3, output_path,
                                 start_offset1=0.0, start_offset2=0.0, start_offset3=0.0):
    """
    Combines three videos side by side horizontally.
    Optionally skip a number of seconds in each video.
    """
    cap1 = cv2.VideoCapture(video_path1)
    cap2 = cv2.VideoCapture(video_path2)
    cap3 = cv2.VideoCapture(video_path3)

    # Get FPS for each video; use the lowest FPS
    fps1 = cap1.get(cv2.CAP_PROP_FPS) or 30
    fps2 = cap2.get(cv2.CAP_PROP_FPS) or 30
    fps3 = cap3.get(cv2.CAP_PROP_FPS) or 30
    fps = min(fps1, fps2, fps3)

    # Skip frames based on offsets (if any)
    if start_offset1 > 0:
        cap1.set(cv2.CAP_PROP_POS_FRAMES, int(start_offset1 * fps1))
    if start_offset2 > 0:
        cap2.set(cv2.CAP_PROP_POS_FRAMES, int(start_offset2 * fps2))
    if start_offset3 > 0:
        cap3.set(cv2.CAP_PROP_POS_FRAMES, int(start_offset3 * fps3))

    # Read first frames from each video to determine dimensions
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    ret3, frame3 = cap3.read()

    if not (ret1 and ret2 and ret3):
        print("Error reading initial frames from videos.")
        cap1.release(); cap2.release(); cap3.release()
        return

    # Use frame1's height as the reference.
    target_height = frame1.shape[0]
    frame2 = resize_to_height(frame2, target_height)
    frame3 = resize_to_height(frame3, target_height)

    # Compute composite dimensions
    composite_width = frame1.shape[1] + frame2.shape[1] + frame3.shape[1]
    composite_height = target_height

    # Set up video writer with the composite dimensions
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (composite_width, composite_height))

    # Write the first composite frame
    composite_frame = cv2.hconcat([frame1, frame2, frame3])
    out.write(composite_frame)

    # Main loop: read next frames until one video ends
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        ret3, frame3 = cap3.read()
        if not (ret1 and ret2 and ret3):
            break

        # Resize frame2 and frame3 to match target height
        frame2 = resize_to_height(frame2, target_height)
        frame3 = resize_to_height(frame3, target_height)

        composite_frame = cv2.hconcat([frame1, frame2, frame3])
        out.write(composite_frame)

    cap1.release()
    cap2.release()
    cap3.release()
    out.release()
    print(f"Composite video saved to: {output_path}")

if __name__ == "__main__":
    # Adjust file names and offsets as needed:
    # For instance, if you want video1 to start 15 seconds in:
    create_three_video_composite(
        video_path1="casterCloseup.mp4",                # e.g., the graph video (or any one)
        video_path2="single_axis_animation_2.mp4",      # new video 1
        video_path3="wholeChair.mp4",                   # new video 2
        output_path="final_composite.mp4",
        start_offset1=2.0,                              # Video Delay Time
        start_offset2=0.0,
        start_offset3=2.0
    )
