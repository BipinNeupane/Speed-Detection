
import numpy as np
import cv2
import time
import os
from ultralytics import YOLO
import random
import tkinter as tk
from tkinter import filedialog





def runDetection(location):
# opening the file in read mode
    my_file = open("coco.txt", "r")
    # reading the file
    data = my_file.read()
    # replacing end splitting the text | when newline ('\n') is seen.
    class_list = data.split("\n")
    my_file.close()


    output_directory = "output_videos"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    output_image = "output_image"
    if not os.path.exists(output_image):
        os.makedirs(output_image)
        
    # print(class_list)
    tracked_objects = {}  # Dictionary to track objects across frames

    def calculate_speed(time1, time2, distance):
        # Ensure non-zero time difference to avoid division by zero
        if time2 > time1:
            # Calculate speed (speed = distance / time)
            speed = distance / (time2 - time1)
            speed_kilometers_per_hour = speed * 3.6
            return speed_kilometers_per_hour
        else:
            return None





    def check_collision_bbox(bbox, line1_start, line1_end, line2_start, line2_end):
        _, y_min, _, y_max = bbox

        # Check collision with line 1
        collision_line1 = y_min <= line1_start[1] and y_max >= line1_start[1]

        # Check collision with line 2
        collision_line2 = y_min <= line2_start[1] and y_max >= line2_start[1]

        return collision_line1, collision_line2


    line1_collision_times = {}  # Dictionary to store collision times for each object
    line2_collision_times = {}
    distance_between_lines = 10


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
    overspeeding_video_writers = {}

    speed_limit = 24.0
    speeding_detected = {}


    my_dict = {}
    # cap = cv2.VideoCapture(1)
    cap = cv2.VideoCapture(location)
    frame_wid = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_hyt = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define video codec


    print(f"FPS : {fps}")

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
            speed = None

            print(bb)
            crop_frame = frame[int(bb[1]):int(bb[3]), int(bb[0]):int(bb[2])]
            print(f'crop frame {crop_frame}')
            
            collision_line1, collision_line2 = check_collision_bbox(bb, line1_start, line1_end, line2_start, line2_end)
            
            if collision_line1 and key not in line1_collision_times:
                line1_collision_times[key] = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 

        # Check collision with line2
            if collision_line2 and key not in line2_collision_times:
                line2_collision_times[key] = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 

        # Calculate speed if both collision times are available
            if key in line1_collision_times and key in line2_collision_times:
                speed = calculate_speed(line1_collision_times[key], line2_collision_times[key], distance_between_lines)
                if speed is not None and speed > speed_limit and track_ids[i] not in speeding_detected:
                    print(f"Over speeding detected for Object {track_ids[i]}! Speed: {speed:.2f} km/hr")

                    
                    # Crop the frame around the bounding box
                    crop_frame = frame[int(bb[1]):int(bb[3]), int(bb[0]):int(bb[2])]
                    
                    overspeeding_clip_path = os.path.join(output_directory, f'output_clip_{track_ids[i]}.mp4')
                    overspeeding_video_writers[track_ids[i]] = cv2.VideoWriter(overspeeding_clip_path, fourcc, 25.0, (frame_wid, frame_hyt))
                    image_path = os.path.join(output_image, f'speeding_frame_{track_ids[i]}.jpg')
                    cv2.imwrite(image_path, crop_frame)
                    # Set the flag to indicate overspeeding has been detected for this object
                    speeding_detected[track_ids[i]] = True, speed, line1_collision_times[track_ids[i]], line2_collision_times[track_ids[i]]

                if track_ids[i] in overspeeding_video_writers:
                # Write the frame to the overspeeding clip
                    overspeeding_video_writers[track_ids[i]].write(frame)

                    
                    # out = None
                    # Save the frame as an image (optional)
                
                

                    # Set the flag to indicate over-speeding has been detected for this object
                    

            # Check if the car exceeds the speed limit
                # Check if the car exceeds the speed limit
    
                
                
        

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

            cv2.line(frame, line1_start, line1_end, (0, 255, 0), 2)  # Line 1 in green
            cv2.line(frame, line2_start, line2_end, (0, 0, 255), 2)  # Line 2 in red


            # for speedkey, speed_detected in speeding_detected.items():
            #     print(f"Current frame : {current_frame}")
            #     start_time = speed_detected[2]
            #     end_time = speed_detected[3]
            #     # if(end_time):
            #     #     print(f"End time: {end_time}")
            #     # else:
            #     #     print("no end time yet")
                
            #     start_frame = int((start_time * fps)+2)
            #     # -1 so it is always lower than the last frame
            #     end_frame = int((end_time * fps)-1)
                
            #     out.write(frame)
                
                # if current_frame == start_frame:
                #     # Start recording overspeeding clip
                #     print('found you')
                #     overspeeding_clip_out = cv2.VideoWriter(overspeeding_clip_path, fourcc, 25.0, (frame_wid, frame_hyt))

                # if overspeeding_clip_out is not None:
                #     # Write the frame to the overspeeding clip
                #     print("going on....")
                #     overspeeding_clip_out.write(frame)

                # if current_frame == end_frame:
                #     # Stop recording overspeeding clip and release the writer
                #     overspeeding_clip_out.release()
                #     # overspeeding_clip_out = None
                
                
                # print(f"Strart Frame: {start_frame}")
                # print(f"End Frame: {end_frame}")
            
        
        start_time = 2.0
        end_time = 6.0
            # if(end_time):
            #     print(f"End time: {end_time}")
            # else:
            #     print("no end time yet")
            
        start_frame = int((start_time * fps)+2)
            # -1 so it is always lower than the last frame          
        end_frame = int((end_time * fps)-1)
        
        if current_frame >= start_frame and current_frame <= end_frame:
            pass
        # print(f"current_frame: {current_frame}")
    
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


def browse_output_image_folder():
    output_folder_path = os.path.abspath("output_image")
    os.system("start explorer " + output_folder_path)
    
def browse_output_video_folder():
    output_folder_path = os.path.abspath("output_videos")
    os.system("start explorer " + output_folder_path)

# ... (remaining code)


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