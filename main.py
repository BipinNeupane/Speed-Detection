# all necessary imports
import numpy as np
import cv2
import os
from ultralytics import YOLO
import random
import tkinter as tk
from tkinter import filedialog

# Main function that has all the code that is accesed with UI and location arguement is for the live camera or video input
def runDetection(location):
# opening the file in read mode which has all the class
    my_file = open("coco.txt", "r")
    # reading the file 
    data = my_file.read()
    # replacing end splitting the text | when newline ('\n') is seen.
    class_list = data.split("\n")
    # closing the file 
    my_file.close()

# setting the output directory for videos
    output_directory = "output_videos"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        
# setting the output directory for videos
    output_image = "output_image"
    if not os.path.exists(output_image):
        os.makedirs(output_image)
    
    # Setting variables
    
    # Setting a specific speed limit
    speed_limit = 24.0
    # all the vechile that are detected as speeding are stored in this dicitonary
    speeding_detected = {}
    # Dictionary to store collision times for each object
    line1_collision_times = {}  
    line2_collision_times = {}
    # distance assumption between each lines
    distance_between_lines = 10
    
    # Generate random colors for class list that takes classID to generate random color
    detection_colors = []
    for i in range(len(class_list)):
        r = random.randint(0,255)
        g = random.randint(0,255)
        b = random.randint(0,255)
        detection_colors.append((b,g,r))
    
    # dictionary for storing key value of id and its information
    my_dict = {}
    
    
    # here the option for location is used if it is live camera then it is 1 if it is a specific local location then it imports the video
    # cap = cv2.VideoCapture(1)
    cap = cv2.VideoCapture(location)
    
    # getting the current frame width and height of the frame for calculation purposes
    frame_wid = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # getting height
    frame_hyt = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # getting the Frames per Second
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    '''
    docs.ultralytics.com. (n.d.). YOLOv8 Documentation. [online] Available at: https://docs.ultralytics.com. (Accessed: 15 January 2024).
    '''
    # load a pretrained YOLOv8n model
    model = YOLO("yolov8n.pt", "v8") 
   
    

    # Vals to resize video frames | small frame optimise the run 
    overspeeding_video_writers = {}

    # Define the output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define video codec

    
    def calculate_speed(time1, time2, distance):
        '''
        Calculates the speed of an object based on two time points and a given distance.

    Parameters:
        time1 (float): The first time point.
        time2 (float): The second time point.
        distance (float): The distance traveled by the object.

    Returns:
        float or None: The calculated speed in kilometers per hour, or None if the time difference is zero or negative.
        '''
        # Ensure non-zero time difference to avoid division by zero
        if time2 > time1:
            # Calculate speed (speed = distance / time)
            speed = distance / (time2 - time1)
             # Convert speed to kilometers per hour
            speed_kilometers_per_hour = speed * 3.6
            return speed_kilometers_per_hour
        else:
            return None

    def check_collision_bbox(bbox, line1_start, line1_end, line2_start, line2_end):
        '''
        Checks if a bounding box collides with two horizontal lines.

        Parameters:
            bbox (tuple): Bounding box coordinates (x_min, y_min, x_max, y_max).
            line1_start (tuple): Starting point of line 1 (x, y).
            line1_end (tuple): Ending point of line 1 (x, y).
            line2_start (tuple): Starting point of line 2 (x, y).
            line2_end (tuple): Ending point of line 2 (x, y).

        Returns:
            tuple: A tuple of boolean values indicating collision with line 1 and line 2 (collision_line1, collision_line2).
        '''
        _, y_min, _, y_max = bbox

        # Check collision with line 1
        collision_line1 = y_min <= line1_start[1] and y_max >= line1_start[1]

        # Check collision with line 2
        collision_line2 = y_min <= line2_start[1] and y_max >= line2_start[1]

        return collision_line1, collision_line2

# testing fps
    print(f"FPS : {fps}")

# input validation for invalid input
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            break
        
        height, width, _ = frame.shape

        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        
        print(f"Overspeeding dict {speeding_detected}")
        
    # Define lines of interest across the entire width
        line1_start = (0, height // 2)           # Starting point of line 1 at the center of the frame
        line1_end = (width, height // 2)          # Ending point of line 1 at the center of the frame
        line2_start = (0, height // 2 + 200)      # Starting point of line 2 slightly below line 1
        line2_end = (width, height // 2 + 200)    
        
        # print(line1_start)
        
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break


        # Object detection code from Yolov8 documentation
        detect_params = model.predict(source=[frame], conf=0.70, save=False)
        # Object tracking code form yolov8 documentation
        '''
        Ultralytics (n.d.). Track. [online] docs.ultralytics.com. Available at: https://docs.ultralytics.com/modes/track/.
        '''
        results = model.track(frame, persist=True)
    
        # get all info about the detected object and their bounding box
        '''
        This code processes detected objects in a frame, extracting information like track IDs, bounding box coordinates, and confidence scores using a YOLOv8 model. It prints and stores this data in a dictionary. Additionally, it extracts and prints bounding box coordinates and generates cropped frames for each detected object. The purpose of the speed variable is not entirely clear in this specific part of the code.
        '''
        for i in range(len(detect_params[0])):
            # print(i)
            # getting id form model.track()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            print(track_ids[i])
            # getting the first index of the tensor and using its boxes method which contains all the information about a detected object
            boxes = detect_params[0].boxes
            # getting the box of iterated detection
            box = boxes[i]  # returns one box
            # classs id for classification
            clsID = box.cls.cpu()
            # getting the confidence score
            conf = box.conf.numpy()[0]
            # getting the xyxy value of the detected object to create the bounding box
            bb = box.xyxy.numpy()[0]
            # print(conf)
            # for testing
            key = track_ids[i]
            value = conf
            my_dict[key] = value
            # initianting a variable speed that is None because it will be filled later
            speed = None

            print(bb)
        
            '''
            This code snippet checks if a detected object's bounding box collides with two horizontal lines in the frame. If a collision occurs with the first line and the object's track ID is not already recorded, it captures the collision time. Similarly, if there's a collision with the second line and the track ID is not yet recorded, it logs the collision time. This information is stored in dictionaries (line1_collision_times and line2_collision_times) to track collision timings for each object with the respective lines.
            '''
            collision_line1, collision_line2 = check_collision_bbox(bb, line1_start, line1_end, line2_start, line2_end)
            
            if collision_line1 and key not in line1_collision_times:
                line1_collision_times[key] = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 

        # Check collision with line2
            if collision_line2 and key not in line2_collision_times:
                line2_collision_times[key] = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 
                
            

            '''
            This code calculates the speed of a detected object based on collision times with two lines in the frame. If the speed surpasses a set limit and the object isn't already marked as overspeeding, it triggers an alert. The script then crops the frame around the object, saves it as an image, and starts recording an overspeeding clip. Relevant details are stored, and if the object is already flagged, the current frame is added to the overspeeding clip. This functionality identifies and records instances of overspeeding in the video feed.
            '''
        # Calculate speed if both collision times are available
            if key in line1_collision_times and key in line2_collision_times:
                # Calculate speed based on collision times and distance between lines
                speed = calculate_speed(line1_collision_times[key], line2_collision_times[key], distance_between_lines)
                # Check if speed exceeds the limit and the object is not yet flagged as overspeeding    
                if speed is not None and speed > speed_limit and track_ids[i] not in speeding_detected:
                     # Print overspeeding alert and details
                    print(f"Over speeding detected for Object {track_ids[i]}! Speed: {speed:.2f} km/hr")
                    
                    # Set compression parameters for image storage
                    compression_params = [cv2.IMWRITE_JPEG_QUALITY, 90]
                    
                    # Crop the frame around the bounding box
                    crop_frame = frame[int(bb[1]):int(bb[3]), int(bb[0]):int(bb[2])]
                    
                    # Create paths for overspeeding clip and image
                    overspeeding_clip_path = os.path.join(output_directory, f'output_clip_{track_ids[i]}.mp4')
                    overspeeding_video_writers[track_ids[i]] = cv2.VideoWriter(overspeeding_clip_path, fourcc, 25.0, (frame_wid, frame_hyt))
                    
                    # Initialize overspeeding video writer for the object
                    image_path = os.path.join(output_image, f'speeding_frame_{track_ids[i]}.jpg')
                    # Save the overspeeding frame as an image
                    cv2.imwrite(image_path, crop_frame,compression_params)
                    # Set the flag to indicate overspeeding has been detected for this object
                    speeding_detected[track_ids[i]] = True, speed, line1_collision_times[track_ids[i]], line2_collision_times[track_ids[i]]

                # If the object is already flagged for overspeeding, write the frame to the overspeeding clip
                if track_ids[i] in overspeeding_video_writers:
                # Write the frame to the overspeeding clip
                    overspeeding_video_writers[track_ids[i]].write(frame)


            # creating bounding box around the detected object with its xyxy values
            cv2.rectangle(
                frame,
                (int(bb[0]), int(bb[1])),
                (int(bb[2]), int(bb[3])),
                detection_colors[int(clsID)],
                3,
            )

            # Display class name and confidenceq
            font = cv2.FONT_HERSHEY_COMPLEX
            speed_text = f"{class_list[int(clsID)]} {track_ids[i]}"
            # if speed is null then it does't show speed 
            if speed is not None:
                speed_text += f" Speed: {speed:.2f} km/hr"

            cv2.putText(
            frame,
            speed_text,
            (int(bb[0]), int(bb[1]) - 10),
            font,
            1,
            (255, 255, 255),
            2,
            )

            # Drawing line for distance calculation
            cv2.line(frame, line1_start, line1_end, (0, 255, 0), 2)  # Line 1 in green
            cv2.line(frame, line2_start, line2_end, (0, 0, 255), 2)  # Line 2 in red

        # Display the resulting frame
        cv2.imshow('ObjectDetection', frame)

        
        
        # Terminate run when "Q" pressed
        if cv2.waitKey(1) == ord('q'):
            # print(line1_collision_times)
            # print(line2_collision_times)
            
            print(speeding_detected)
        
            break

    # When everything done, release the capture
    cap.release()
    # overspeeding_clip_out.release()
    cv2.destroyAllWindows()


def center_window(window, width, height):
    '''
    This function, center_window, takes three parameters: window, width, and height, referring to a tkinter window and its desired dimensions. It calculates the screen width and height using winfo_screenwidth() and winfo_screenheight() methods. Then, it computes the x and y positions to center the window on the screen. Finally, it sets the window geometry using these values, effectively centering the window on the screen.
    '''
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    x_position = (screen_width - width) // 2
    y_position = (screen_height - height) // 2
    window.geometry(f"{width}x{height}+{x_position}+{y_position}")


def browse_file():
    file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi")])
    if file_path:
        # Process the video file (you need to implement the processing logic)
        runDetection(file_path)

def use_live_camera():
    # Process live camera stream (you need to implement the processing logic)
    runDetection(0)
    print("Using live camera")

# Create the main Tkinter window
root = tk.Tk()
root.title("Speed Detection App")

# Set the window size and position
center_window(root, 600, 400)

# opens the file explorer for displaying images
def browse_output_image_folder():
    output_folder_path = os.path.abspath("output_image")
    os.system("start explorer " + output_folder_path)
    
def browse_output_video_folder():
    # opens the file explorer for displaying videos
    output_folder_path = os.path.abspath("output_videos")
    os.system("start explorer " + output_folder_path)



# Label for "Speed Detection" centered at the top
label = tk.Label(root, text="Speed Detection", font=("Helvetica", 16))
label.pack(pady=20)

# Button to add a video file
add_video_button = tk.Button(root, text="Add Video", command=browse_file)
add_video_button.pack(pady=10)

# Button to use live camera
live_camera_button = tk.Button(root, text="Use Live Camera", command=use_live_camera)
live_camera_button.pack(pady=10)

# Button to browse the output images folder
browse_output_button = tk.Button(root, text="Browse Output Images", command=browse_output_image_folder)
browse_output_button.pack(pady=10)

browse_output_button = tk.Button(root, text="Browse Output Clips", command=browse_output_video_folder)
browse_output_button.pack(pady=10)

# Start the Tkinter event loop
root.mainloop()