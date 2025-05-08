import cv2
import numpy as np
from ultralytics import YOLO

# Configuration flags 
SHOW_STATIONARY_BOXES = False
SHOW_FLOOR_ZONE_BOXES = False
PRINT_LIGHTS_ON_OFF = False

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Use yolov8s.pt or yolov8m.pt for better accuracy

# Open webcam video stream
cap = cv2.VideoCapture(0)

# Defined stationary boxes (same as V2 just condensed for presentation)
stationary_boxes = [
    [0, 0, 128, 96], [128, 0, 256, 96], [256, 0, 384, 96], [384, 0, 512, 96], [512, 0, 640, 96],
    [0, 96, 128, 192], [128, 96, 256, 192], [256, 96, 384, 192], [384, 96, 512, 192], [512, 96, 640, 192],
    [0, 192, 128, 288], [128, 192, 256, 288], [256, 192, 384, 288], [384, 192, 512, 288], [512, 192, 640, 288],
    [0, 288, 128, 384], [128, 288, 256, 384], [256, 288, 384, 384], [384, 288, 512, 384], [512, 288, 640, 384],
    [0, 384, 128, 480], [128, 384, 256, 480], [256, 384, 384, 480], [384, 384, 512, 480], [512, 384, 640, 480]
]

# (same as V2 just condensed for presentation)
quadrant_boxes = [ 
    [0, 0, 320, 240], [320, 0, 640, 240],
    [0, 240, 320, 480], [320, 240, 640, 480]
]

# (same as V2 just condensed for presentation)
floor_zone_boxes = [
    [0, 220, 240, 480], [400, 220, 640, 480]
]

def is_point_in_box(point, box):
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2

def get_occupied_zones(human_boxes):
    occupied_zones = set()
    for (xA, yA, xB, yB, confidence) in human_boxes:
        if confidence < 0.55:
            continue
        bottom_left = (xA, yB)
        bottom_right = (xB, yB)
        for zone_num, zone in enumerate(floor_zone_boxes, start=1):
            if is_point_in_box(bottom_left, zone) or is_point_in_box(bottom_right, zone):
                occupied_zones.add(zone_num)
    return occupied_zones

# Set up to be able to keep a frame on screen (for presentation)
prev_zone_states = {}
print("Press SPACE to analyze frame, Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Always show live feed
    display_frame = frame.copy()
    cv2.imshow('Live Feed - Press SPACE to analyze', display_frame)
    
    key = cv2.waitKey(1)
    
    if key == ord(' '):  # Space pressed - process this frame
        processed_frame = frame.copy()
        
        # Draw boxes if enabled
        if SHOW_STATIONARY_BOXES:
            for box in stationary_boxes + quadrant_boxes:
                color = (0, 0, 255) if box in stationary_boxes else (0, 255, 255)
                cv2.rectangle(processed_frame, (box[0], box[1]), (box[2], box[3]), color, 2)
        
        if SHOW_FLOOR_ZONE_BOXES:
            for zone_num, box in enumerate(floor_zone_boxes, start=1):
                cv2.rectangle(processed_frame, (box[0], box[1]), (box[2], box[3]), (0, 165, 255), 2)
                cv2.putText(processed_frame, f"Zone {zone_num}", (box[0]+5, box[1]+20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        # Run detection
        results = model(processed_frame, classes=[0], verbose=False)
        human_boxes = []
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                human_boxes.append((x1, y1, x2, y2, confidence))
                cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(processed_frame, f"Person: {confidence:.2f}", 
                           (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Zone detection
        occupied_zones = get_occupied_zones(human_boxes)
        if PRINT_LIGHTS_ON_OFF:
            for zone_num in range(1, len(floor_zone_boxes)+1):
                state = "ON" if zone_num in occupied_zones else "OFF"
                if zone_num not in prev_zone_states or prev_zone_states[zone_num] != state:
                    print(f"Zone {zone_num}: {state}")
                    prev_zone_states[zone_num] = state
        
        cv2.imshow('Processed Frame', processed_frame)
    
    elif key == ord('q'):  # Quit
        break

cap.release()
cv2.destroyAllWindows()