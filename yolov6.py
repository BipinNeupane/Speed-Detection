from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# variables
line1_start = (100, 300)  # Starting point of line 1
line1_end = (400, 300)    # Ending point of line 1
line2_start = (500, 300)  # Starting point of line 2
line2_end = (800, 300)

# Open the video file
video_path = "src/cars2.mp4"
cap = cv2.VideoCapture(video_path)

# Store the track history
track_history = defaultdict(lambda: [])

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)

        # Get the boxes and track IDs
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        print(results[0])

        for detect in results:
            for i in range(len(detect.boxes)):
                box_id = detect.boxes[i].id
                box_coordinates = detect.boxes[i].xyxy[0].cpu().numpy()
                # print(box_id)
                # print(box_coordinates)
# Now 'box_id' contains the ID corresponding to 'box_coordinates'
# You can use this ID to track the
        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # create two lines as area of interest
        
        
        # Plot the tracks
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))  # x, y center point
            if len(track) > 30:  # retain 90 tracks for 90 frames
                track.pop(0)

            # Draw the tracking lines
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

        image = cv2.line(annotated_frame, line1_start, line1_end, (0,0,0), 2) 
        
        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print(track)
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()