import cv2
import os
import time
import json
import csv
import numpy as np

OPENCV_TRACKERS = {
    'MOSSE': cv2.legacy.TrackerMOSSE_create,
    'CSRT': cv2.legacy.TrackerCSRT_create,
    'MedianFlow': cv2.legacy.TrackerMedianFlow_create,
    'TLD': cv2.legacy.TrackerTLD_create,
    'KCF': cv2.legacy.TrackerKCF_create
}

# Colors for each tracker
TRACKER_COLORS = {
    'MOSSE': (255, 0, 0),
    'CSRT': (0, 255, 0),
    'MedianFlow': (0, 0, 255),
    'TLD': (255, 255, 0),
    'KCF': (255, 0, 255)
}

# Symbol for each tracker
TRACKER_SYMBOL = {
    'MOSSE': '●',  # Circle
    'CSRT': '●',  # Circle
    'MedianFlow': '●',  # Circle
    'TLD': '●',  # Circle
    'KCF': '●'  # Circle
}

# Loads video and its first frame
video_path = 'dataset\\drone_video.mp4'
cap = cv2.VideoCapture(video_path)

# Get video properties
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Load annotations
annotations_path = 'dataset\\annotations.json'
with open(annotations_path, 'r') as file:
    data = json.load(file)

# Access the first annotation
first_annotation = data['annotations'][0]
initial_bbox = tuple(first_annotation['bbox'])

# Initialize result storage
results = []

# To display tracking progress
progress_info = {}

for tracker_name, tracker_create in OPENCV_TRACKERS.items():
    # Reset video capture to the beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, frame = cap.read()

    # Initialize the tracker
    tracker = tracker_create()
    tracker.init(frame, initial_bbox)

    # Output video writer initializer
    out = cv2.VideoWriter(f'output\\drone_tracking_{tracker_name}.mp4', fourcc, 30, (width, height))

    # Initialize frame and time counters
    start_time = time.time()
    processed_frames = 0
    total_fps = 0
    total_frame_time = 0

    while True:
        # Initializes FPS counter
        timer = cv2.getTickCount()

        # Reads the next frame from the video
        ret, frame = cap.read()
        
        # Ends loop if there is any error
        if not ret or frame is None:
            break

        # Update tracker and get bbox
        ret, bbox = tracker.update(frame)
        if ret:
            x, y, w, h = [int(v) for v in bbox]
            bbox_center = (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))
        else:
            bbox_center = 'N/A'

        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        frame_time = 1000 / fps
        total_fps += fps
        total_frame_time += frame_time

        # Store results for later drawing
        results.append({
            'tracker_name': tracker_name,
            'bbox': bbox if ret else None,
            'bbox_center': bbox_center,
            'fps': fps,
            'frame_time': frame_time
        })

        # Draw tracker symbol
        symbol = TRACKER_SYMBOL[tracker_name]
        symbol_size = cv2.getTextSize(symbol, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)[0]
        symbol_color = TRACKER_COLORS[tracker_name]
        cv2.putText(frame, symbol, (box_x + 10, box_y + 30 + symbol_size[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.75, symbol_color, 2)

        # Update processed frames counter
        processed_frames += 1

        # Estimate time left
        elapsed_time = time.time() - start_time
        estimated_total_time = (elapsed_time / processed_frames) * total_frames
        remaining_time = estimated_total_time - elapsed_time
        
        # Estimate percentual progress
        progress_percentage = (processed_frames / total_frames) * 100
        
        # Clear screen and print progress
        os.system('cls')
        for previous_tracker, info in progress_info.items():
            print(f"{previous_tracker} - Processed Frames: {info['processed_frames']} - Elapsed Time: {info['elapsed_time']:.2f}s - Average FPS: {info['average_fps']:.2f} - Average Frame Time: {info['average_frame_time']:.2f} ms")
        
        print(f"Processing {tracker_name}: {processed_frames} processed frames from {total_frames} ({progress_percentage:.0f}%) - FPS: {fps:.0f} - Frametime: {frame_time:.2f} ms - Elapsed time: {elapsed_time:.2f}s - Estimated time: {remaining_time:.2f}s")

    # Calculate average FPS and frame time
    average_fps = total_fps / processed_frames
    average_frame_time = total_frame_time / processed_frames

    # Store progress information
    progress_info[tracker_name] = {
        'total_frames': total_frames,
        'processed_frames': processed_frames,
        'elapsed_time': elapsed_time,
        'estimated_total_time': estimated_total_time,
        'remaining_time': remaining_time,
        'progress_percentage': progress_percentage,
        'average_fps': average_fps,
        'average_frame_time': average_frame_time
    }

    print(f"{tracker_name} - Done")

# Release video capture and destroy all windows
cap.release()
cv2.destroyAllWindows()

# Display all progress information
os.system('cls')
for tracker_name, info in progress_info.items():
    print(f"{tracker_name}: {info['processed_frames']} processed frames from {info['total_frames']} ({info['progress_percentage']:.0f}%) - Elapsed time: {info['elapsed_time']:.2f}s    Estimated time: {info['remaining_time']:.2f}s - Average FPS: {info['average_fps']:.2f} - Average Frame Time: {info['average_frame_time']:.2f} ms")

# Generate CSV file with tracking results
output_csv_path = 'output\\tracking_results.csv'
with open(output_csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Tracker', 'Frame', 'Center', 'FPS', 'Frame Time'])
    for i, result in enumerate(results):
        writer.writerow([result['tracker_name'], i, result['bbox_center'], result['fps'], result['frame_time']])
