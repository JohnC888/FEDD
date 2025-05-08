import cv2
import numpy as np
from ultralytics import YOLO


# Configuration flags
SHOW_STATIONARY_BOXES = False
SHOW_FLOOR_ZONE_BOXES = True
PRINT_LIGHTS_ON_OFF = True


# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Use yolov8s.pt or yolov8m.pt for better accuracy


# open webcam video stream
cap = cv2.VideoCapture(0)


# Video writer
out = cv2.VideoWriter(
    'output.avi',
    cv2.VideoWriter_fourcc(*'MJPG'),
    15.,
    (1280,960))


# Defined stationary red boxes here
# Format: [x1, y1, x2, y2] where (x1,y1) is top-left and (x2,y2) is bottom-right
stationary_boxes = [
    # Top row (y: 0 to 96)
    [0, 0, 128, 96],        # Top Left
    [128, 0, 256, 96],      # Top Middle-left
    [256, 0, 384, 96],      # Top Middle
    [384, 0, 512, 96],      # Top Middle-right
    [512, 0, 640, 96],      # Top Right
   
    # Second row (y: 96 to 192)
    [0, 96, 128, 192],      # Middle-left Top
    [128, 96, 256, 192],    # Middle-left Middle
    [256, 96, 384, 192],    # Center
    [384, 96, 512, 192],    # Middle-right Middle
    [512, 96, 640, 192],    # Middle-right Top
   
    # Third row (y: 192 to 288)
    [0, 192, 128, 288],     # Middle-left Bottom
    [128, 192, 256, 288],   # Middle-bottom Left
    [256, 192, 384, 288],   # Bottom Middle
    [384, 192, 512, 288],   # Middle-bottom Right
    [512, 192, 640, 288],   # Middle-right Bottom
   
    # Bottom row (y: 288 to 384)
    [0, 288, 128, 384],     # Bottom Left
    [128, 288, 256, 384],   # Bottom Middle-left
    [256, 288, 384, 384],   # Bottom Center
    [384, 288, 512, 384],   # Bottom Middle-right
    [512, 288, 640, 384],   # Bottom Right
   
    # Additional bottom area (y: 384 to 480)
    [0, 384, 128, 480],     # Far Bottom Left
    [128, 384, 256, 480],   # Far Bottom Middle-left
    [256, 384, 384, 480],   # Far Bottom Center
    [384, 384, 512, 480],   # Far Bottom Middle-right
    [512, 384, 640, 480]    # Far Bottom Right
]


# Define 4 stationary quadrant boxes (yellow)
# Format: [x1, y1, x2, y2] where (x1,y1) is top-left and (x2,y2) is bottom-right
quadrant_boxes = [
    [0, 0, 320, 240],       # Top-left quadrant
    [320, 0, 640, 240],     # Top-right quadrant
    [0, 240, 320, 480],     # Bottom-left quadrant
    [320, 240, 640, 480]    # Bottom-right quadrant
]


# Define stationary floor zone boxes (orange)
# Format: [x1, y1, x2, y2] where (x1,y1) is top-left and (x2,y2) is bottom-right
floor_zone_boxes = [
    [0, 220*2, 240*2, 480*2],     # floor zone 1
    [400*2, 220*2, 640*2, 480*2],   # floor zone 2
]


def is_point_in_box(point, box):
    # Check if a point (x,y) is inside a box [x1,y1,x2,y2]
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def get_occupied_zones(human_boxes):
    # Determine which zones have people present
    occupied_zones = set()
   

    for (xA, yA, xB, yB) in human_boxes:
        if confidence < 0.55:  # Skip detections below confidence threshold of 55% that its a person
            continue


        bottom_left = (xA, yB)
        bottom_right = (xB, yB)
       
        for zone_num, zone in enumerate(floor_zone_boxes, start=1):
            if is_point_in_box(bottom_left, zone) or is_point_in_box(bottom_right, zone):
                occupied_zones.add(zone_num)
               
    return occupied_zones



# For tracking zone states between frames
prev_zone_states = {}

print("Running")
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break
   
    # resizing for faster detection
    frame = cv2.resize(frame, (1280,960))


    # using a greyscale picture, also for faster detection
    # if SHOW_STATIONARY_BOXES:
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)


    # Draw stationary boxes if enabled
    if SHOW_STATIONARY_BOXES:
        for box in stationary_boxes:
            xA, yA, xB, yB = box
            cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 0, 255), 2)
        for box in quadrant_boxes:
            xA, yA, xB, yB = box
            cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 255), 2)


    # Draw floor zone boxes if enabled
    if SHOW_FLOOR_ZONE_BOXES:
        for zone_num, box in enumerate(floor_zone_boxes, start=1):
            xA, yA, xB, yB = box
            color = (0, 165, 255)  # Orange
            cv2.rectangle(frame, (xA, yA), (xB, yB), color, 2)
            cv2.putText(frame, f"Zone {zone_num}", (xA+5, yA+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)


    # YOLOv8 detection (only for 'person' class)
    results = model(frame, classes=[0], verbose=False)  # Class 0 = person
    human_boxes = []
   
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            human_boxes.append([x1, y1, x2, y2])
           
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
           
            # Add label
            confidence = box.conf[0]
            cv2.putText(frame, f"Person: {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    # Get current occupied zones
    occupied_zones = get_occupied_zones(human_boxes)
   
    # Print zone status if changed
    if PRINT_LIGHTS_ON_OFF:
        current_zone_states = {}
        for zone_num in range(1, len(floor_zone_boxes)+1):
            state = "ON" if zone_num in occupied_zones else "OFF"
            current_zone_states[zone_num] = state
           
            if zone_num not in prev_zone_states or prev_zone_states[zone_num] != state:
                print(f"Zone {zone_num}: {state}")
               
        prev_zone_states = current_zone_states


    # Write and display frame
    out.write(frame.astype('uint8'))
    cv2.imshow('Sitting Person Detection with Zones', frame)
   
    # finally, close the window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# When everything done, release the capture
cap.release()
# and release the output
out.release()
cv2.destroyAllWindows()