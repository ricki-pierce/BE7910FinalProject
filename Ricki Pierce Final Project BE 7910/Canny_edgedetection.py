import cv2
import numpy as np
import os
import random
from pathlib import Path
import subprocess
import platform
import csv
from datetime import timedelta

# Define the folder where all the video files are stored
video_folder = Path(r"D:\Ricki\SPRING 2025\BE 7910\Project\Project Videos")
video_files = list(video_folder.glob("*.mov"))  # List all .mov files in the folder

# Create the background subtractor (works well for static backgrounds)
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)
#   ^OpenCV's MOG2 background subtraction algorithm builds a statistical model of the background over time
#   ^models each pixel with mixture of Gaussians to identify the moving objects (foreground) via comparing new frames to the learned background
#   ^"history=500" represents the number of frames used to learn what is considered the background
#   ^"varThreshold=50" represents the sensitivity to change; higher thresholds mean fewer false positive
#   ^detectShadows=True represents the system ignoring shadows in teh foreground mask 

# Function to analyze whether subject is in the video frame and visualize edges
def analyze_subject_in_frame(video_path, frame_width, frame_height, fps):
    cap = cv2.VideoCapture(str(video_path))  # Open video
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_of_frame_times = []  # Time intervals where subject is out of frame
    in_frame_count = 0  # Counter for frames where subject is in frame
    current_out_start = None  # Start time for out-of-frame segments

    # Adjust width and height due to rotation
    frame_width, frame_height = frame_height, frame_width
    #   ^the visualization results kept popping up sideways so i flipped the frame height and width dimensions since i wanted them rotated 90degrees

    while True:
        ret, frame = cap.read()  # Read next frame
        #   ^each frame is read
        if not ret:
            break

        # Rotate the frame 90 degrees clockwise
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        #   ^each frame is rotated 90 degrees

        # Convert to grayscale for background subtraction
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #   ^converted to grayscale to make it easier to detect motion vs background
        fg_mask = fgbg.apply(gray_frame)  # Apply background subtractor to get foreground mask
        #   ^detects moving objects which is considered the foregroudn

        # Show the foreground mask
        cv2.imshow("Foreground Mask", fg_mask)
        cv2.moveWindow("Foreground Mask", -50, 0)

        # Perform edge detection on the foreground mask
        edges = cv2.Canny(fg_mask, 50, 150)  # Adjust thresholds for better edge detection
        #   ^uses Canny edge detection which is a gradien based method to detect sharp intenisty changes
        #   ^these sharp intensity changes are considered the edges

        # Show edges
        cv2.imshow("Edges", edges)
        cv2.moveWindow("Edges", 600, 0)

        # Apply morphological operations to enhance edges (dilate to fill in gaps)
        kernel = np.ones((5, 5), np.uint8)
        #   ^applies 5x5 kernel to thicken edges
        dilated_edges = cv2.dilate(edges, kernel, iterations=2)
        #   ^bridges gaps using dilate 

        # Find contours based on edges
        contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #   ^attempts to find shapes (contours) from the dilated edge version of the frame image

        person_candidates = []  # List to store possible person contours
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 1000:
                continue
            #   ^area filter ignores small noise
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)
            if aspect_ratio < 0.2 or aspect_ratio > 1.5:
                continue
            if y + h > frame_height * 0.3:
                person_candidates.append(contour)
            #   ^keeps only the contours with realistic human shape

        main_contour = max(person_candidates, key=cv2.contourArea, default=None)
        #   ^removes objects that are not the person
        #    valid candidates are kept and the largest is determined to be the person

        if main_contour is not None:
            cv2.drawContours(frame, [main_contour], -1, (0, 255, 0), 2)
            in_frame_count += 1

            x, y, w, h = cv2.boundingRect(main_contour)
            margin = 20
            if x < margin or y < margin or x + w > frame_width - margin or y + h > frame_height - margin:
                if current_out_start is None:
                    current_out_start = cap.get(cv2.CAP_PROP_POS_FRAMES) / fps
                cv2.drawContours(frame, [main_contour], -1, (0, 0, 255), 2)
            else:
                if current_out_start is not None:
                    out_end = cap.get(cv2.CAP_PROP_POS_FRAMES) / fps
                    out_of_frame_times.append((current_out_start, out_end))
                    current_out_start = None
        else:
            if current_out_start is None:
                current_out_start = cap.get(cv2.CAP_PROP_POS_FRAMES) / fps
        #    ^ if a contour is found, its automatically green. when gets too close to the frame edge, turns red. timer is starte d to record out of frame intervals
        #    ^ if no contour found, new out of frame interval starts

        # Show the processed frame
        cv2.imshow("Frame", frame)
        cv2.moveWindow("Frame", 1200, 0)

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

    cap.release()
    cv2.destroyAllWindows()

    return {
        "frame": frame,
        "frame_width": frame_width,
        "frame_height": frame_height,
        "in_frame_count": in_frame_count
    }


# Path to log file for writing pose analysis results
log_path = video_folder / "pose_analysis_log.csv"
with open(log_path, mode='w', newline='') as log_file:
    csv_writer = csv.writer(log_file)
    csv_writer.writerow(["Video Name", "In-Frame Percentage"])

    first_video_data = None  # Store first video result for visualization later

    # Loop over all video files
    for i, video_path in enumerate(video_files):
        print(f"\n🎮 Processing video: {video_path.name}")
        cap = cv2.VideoCapture(str(video_path))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        video_data = analyze_subject_in_frame(video_path, frame_width, frame_height, fps)

        # Reopen video to get frame count for calculating percentage
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        in_frame_percentage = None

        if frame_count:
            in_frame_percentage = (video_data["in_frame_count"] / frame_count) * 100
        else:
            in_frame_percentage = 0.0

        # Write results to CSV
        csv_writer.writerow([video_path.name, f"{in_frame_percentage:.2f}"])

        if i == 0:
            first_video_data = video_data  # Save for final visualization

# Automatically open the CSV log file for review
if platform.system() == "Windows":
    os.startfile(log_path)
else:  # Linux and others
    subprocess.call(["xdg-open", log_path])
