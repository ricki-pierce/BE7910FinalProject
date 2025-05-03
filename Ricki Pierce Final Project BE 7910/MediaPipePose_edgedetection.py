# Import necessary libraries
import os
import cv2
from pymediainfo import MediaInfo
from pathlib import Path
from datetime import timedelta
import mediapipe as mp
import random
import csv
import subprocess
import platform

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.9, min_tracking_confidence=0.9)
mp_drawing = mp.solutions.drawing_utils
padding = 60

# Function to extract and display video metadata
def get_video_metadata(video_path):
    print(f"\n--- Metadata for {video_path.name} ---")

    size_bytes = video_path.stat().st_size
    size_kb = size_bytes / 1024
    size_mb = size_kb / 1024
    print(f"File size: {size_mb:.2f} MB ({size_kb:.2f} KB)")

    print(f"File format: {video_path.suffix.upper().lstrip('.')}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("Failed to open video.")
        return None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = frame_count / fps if fps else 0
    cap.release()

    print(f"Resolution: {width} x {height}")
    print(f"Frames per second (FPS): {fps:.2f}")
    print(f"Frame count: {frame_count}")
    print(f"Duration: {str(timedelta(seconds=int(duration_sec)))}")

    media_info = MediaInfo.parse(video_path)
    video_track = next((track for track in media_info.tracks if track.track_type == "Video"), None)
    general_track = next((track for track in media_info.tracks if track.track_type == "General"), None)

    if video_track:
        print(f"Codec: {video_track.codec_id or video_track.format}")
        print(f"Bit rate: {video_track.bit_rate or 'Unknown'} bps")
        print(f"Bit depth: {video_track.bit_depth or 'Unknown'} bit")
        print(f"Color space: {video_track.color_space or 'Unknown'}")
        print(f"Color range: {video_track.color_range or 'Unknown'}")
        print(f"Pixel aspect ratio: {video_track.pixel_aspect_ratio or 'Unknown'}")
        print(f"Display aspect ratio: {video_track.display_aspect_ratio or 'Unknown'}")
    else:
        print("No detailed video track metadata found.")

    if general_track:
        print(f"Date: {general_track.encoded_date or 'Unknown'}")
    else:
        print("No general metadata found.")

    return analyze_subject_in_frame(video_path, width, height, fps)

# Function to analyze subject in frame and return a random frame
def analyze_subject_in_frame(video_path, frame_width, frame_height, fps):
    cap = cv2.VideoCapture(str(video_path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_of_frame_times = []
    in_frame_count = 0
    current_out_start = None

    random_frame_index = random.randint(0, frame_count - 1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            x_min = min([landmark.x for landmark in results.pose_landmarks.landmark])
            x_max = max([landmark.x for landmark in results.pose_landmarks.landmark])
            y_min = min([landmark.y for landmark in results.pose_landmarks.landmark])
            y_max = max([landmark.y for landmark in results.pose_landmarks.landmark])

            x_min = int(x_min * frame_width) - padding
            x_max = int(x_max * frame_width) + padding
            y_min = int(y_min * frame_height) - padding
            y_max = int(y_max * frame_height) + padding

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            margin = 20
            if x_min > margin and y_min > margin and x_max < (frame_width - margin) and y_max < (frame_height - margin):
                in_frame_count += 1
                if current_out_start is not None:
                    out_end = cap.get(cv2.CAP_PROP_POS_FRAMES) / fps
                    out_of_frame_times.append((current_out_start, out_end))
                    current_out_start = None
            else:
                if current_out_start is None:
                    current_out_start = cap.get(cv2.CAP_PROP_POS_FRAMES) / fps
        else:
            if current_out_start is None:
                current_out_start = cap.get(cv2.CAP_PROP_POS_FRAMES) / fps

        cv2.imshow('Video with Bounding Box', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if current_out_start is not None:
        out_of_frame_times.append((current_out_start, frame_count / fps))

    in_frame_percentage = (in_frame_count / frame_count) * 100
    print(f"\nSubject fully in frame: {in_frame_percentage:.2f}% of frames.")

    if out_of_frame_times:
        print("\nTime intervals where subject was out of frame:")
        for start, end in out_of_frame_times:
            print(f" - From {str(timedelta(seconds=int(start)))} to {str(timedelta(seconds=int(end)))}")
    else:
        print("Subject remained in frame for the entire video.")

    cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame_index)
    ret, random_frame = cap.read()

    cap.release()
    cv2.destroyAllWindows()

    return {
        "frame": random_frame,
        "frame_width": frame_width,
        "frame_height": frame_height,
        "in_frame_count": in_frame_count  # Add this
    }


# Define the folder containing your videos
video_folder = Path(r"D:\Ricki\SPRING 2025\BE 7910\Project\Project Videos")
video_files = list(video_folder.glob("*.mov"))

# CSV log setup
log_path = video_folder / "pose_analysis_log.csv"
with open(log_path, mode='w', newline='') as log_file:
    csv_writer = csv.writer(log_file)
    csv_writer.writerow(["Video Name", "In-Frame Percentage"])

    first_video_data = None  # For visualization at the end

    for i, video_path in enumerate(video_files):
        print(f"\nProcessing video: {video_path.name}")
        video_data = get_video_metadata(video_path)

        # Estimate percentage in-frame from the result
        cap = cv2.VideoCapture(str(video_path))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        in_frame_percentage = None
        if frame_count:
            # Manually count how many frames the person was in-frame from the earlier print
            # But we can also store it directly in analyze_subject_in_frame and return it
            # So let's modify analyze_subject_in_frame slightly to return percentage too
            in_frame_percentage = (video_data["in_frame_count"] / frame_count) * 100
        else:
            in_frame_percentage = 0.0

        csv_writer.writerow([video_path.name, f"{in_frame_percentage:.2f}"])

        # Save first video's data for visualization
        if i < 2:
            # --- Visualize landmarks and bounding box for the first two videos ---
            original_frame = video_data["frame"]
            frame_width = video_data["frame_width"]
            frame_height = video_data["frame_height"]

            if original_frame is not None:
                rgb_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb_frame)
                frame_with_landmarks = rgb_frame.copy()
                frame_with_landmarks_and_box = rgb_frame.copy()

                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(frame_with_landmarks, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    mp_drawing.draw_landmarks(frame_with_landmarks_and_box, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                    x_min = min([landmark.x for landmark in results.pose_landmarks.landmark])
                    x_max = max([landmark.x for landmark in results.pose_landmarks.landmark])
                    y_min = min([landmark.y for landmark in results.pose_landmarks.landmark])
                    y_max = max([landmark.y for landmark in results.pose_landmarks.landmark])

                    x_min = int(x_min * frame_width)
                    x_max = int(x_max * frame_width)
                    y_min = int(y_min * frame_height)
                    y_max = int(y_max * frame_height)

                    cv2.rectangle(frame_with_landmarks_and_box, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                display_frames = {
                    f"Video {i+1}: Original Frame": original_frame,
                    f"Video {i+1}: RGB Frame": cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR),
                    f"Video {i+1}: Landmarks Drawn": cv2.cvtColor(frame_with_landmarks, cv2.COLOR_RGB2BGR),
                    f"Video {i+1}: Landmarks + Box": cv2.cvtColor(frame_with_landmarks_and_box, cv2.COLOR_RGB2BGR),
                }

                for title, img in display_frames.items():
                    cv2.imshow(title, img)

                print(f"\n📸 Displaying landmarks for Video {i+1}. Press any key to continue...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                print(f"⚠️ Could not extract random frame for visualization from video {i+1}.")


# Automatically open the CSV log file
if platform.system() == "Windows":
    os.startfile(log_path)
elif platform.system() == "Darwin":  # macOS
    subprocess.call(["open", log_path])
else:  # Linux and others
    subprocess.call(["xdg-open", log_path])
