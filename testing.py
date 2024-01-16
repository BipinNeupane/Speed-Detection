
import numpy as np
import cv2
from ultralytics import YOLO
import random

# opening the file in read mode
my_file = open("coco.txt", "r")
# reading the file
data = my_file.read()
# replacing end splitting the text | when newline ('\n') is seen.
class_list = data.split("\n")
my_file.close()

# print(class_list)
tracked_objects = {}  # Dictionary to track objects across frames
frame_count = 0

def intersects_line(box, line_start, line_end):
    """Checks if a bounding box intersects a line segment."""
    for i in range(4):  # Check all four corners of the box
        corner1 = box[i]
        corner2 = box[(i + 1) % 4]
        if line_intersection(corner1, corner2, line_start, line_end):
            return True
    return False

def line_intersection(point1, point2, line_start, line_end):
    """Checks if two line segments intersect."""
    # ... (implementation of line intersection logic using vector cross products)

def track_object(tracked_objects, box, frame_count):
    """Tracks objects across frames and assigns unique IDs."""
    for id, tracked_box in tracked_objects.items():
        if distance(tracked_box[-1], box) < 50:  # Track objects within a threshold
            tracked_objects[id].append(box)
            return id
    new_id = len(tracked_objects) + 1  # Assign a new ID for untracked objects
    tracked_objects[new_id] = [box]
    return new_id

def calculate_speed(tracked_objects, id, frame_count):
    """Calculates speed based on distance and time."""
    positions = tracked_objects[id]
    if len(positions) >= 2:
        distance_traveled = distance(positions[-2], positions[-1])
        time_elapsed = frame_count - positions[-2][4]
        speed = distance_traveled / time_elapsed
        return speed
    return 0  # Speed cannot be calculated yet

def distance(point1, point2):
    """Calculates Euclidean distance between two points."""
    return np.sqrt(np.sum((point1[:2] - point2[:2])**2))


# Define lines of interest (adjust coordinates as needed)
line1_start = (100, 300)  # Starting point of line 1
line1_end = (400, 300)    # Ending point of line 1
line2_start = (500, 300)  # Starting point of line 2
line2_end = (800, 300)    # Ending point of line 2



# Generate random colors for class list
detection_colors = []
for i in range(len(class_list)):
    r = random.randint(0,255)
    g = random.randint(0,255)
    b = random.randint(0,255)
    detection_colors.append((b,g,r))

# load a pretrained YOLOv8n model
model = YOLO("yolov8n.pt", "v8") 

# Vals to resize video frames | small frame optimise the run 
frame_wid = 640
frame_hyt = 480

my_dict = {}
# cap = cv2.VideoCapture(1)
cap = cv2.VideoCapture("src/highway.mp4")

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    
    #  resize the frame | small frame optimise the run 
    # frame = cv2.resize(frame, (frame_wid, frame_hyt))

    # Predict on image
    detect_params = model.predict(source=[frame], conf=0.70, save=False)
    results = model.track(frame, persist=True)
   
    
    # print(f"DP :{DP}")

   
    for i in range(len(detect_params[0])):
            # print(i)
            track_ids = results[0].boxes.id.int().cpu().tolist()
            print(track_ids[i])
            boxes = detect_params[0].boxes
            box = boxes[i]  # returns one box
            clsID = box.cls.cpu()
            conf = box.conf.numpy()[0]
            bb = box.xyxy.numpy()[0]
            # print(conf)
            key = track_ids[i]
            value = conf
            my_dict[key] = value
       
       
            
            cv2.rectangle(
                frame,
                (int(bb[0]), int(bb[1])),
                (int(bb[2]), int(bb[3])),
                detection_colors[int(clsID)],
                3,
            )

            # Display class name and confidenceq
            font = cv2.FONT_HERSHEY_COMPLEX
            cv2.putText(
                frame,
                class_list[int(clsID)]
                + " "
                + str(track_ids[i])
                + " "
                + str(round(conf, 3))
                + "%",
                (int(bb[0]), int(bb[1]) - 10),
                font,
                1,
                (255, 255, 255),
                2,
            )

    # Display the resulting frame
    cv2.imshow('ObjectDetection', frame)

    # Terminate run when "Q" pressed
    if cv2.waitKey(1) == ord('q'):
        print(my_dict)
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()



