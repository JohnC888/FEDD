# import the necessary packages
import numpy as np
import cv2

# Configuration flags 
SHOW_STATIONARY_BOXES = False  # Set this to False to disable stationary boxes initially
PRINT_HUMAN_BOX_BOTTOM_VALUES = False  # Set this to False to disable printing bottom values of human boxes
SHOW_FLOOR_ZONE_BOXES = True  # Set this to False to disable floor zone boxes initially
PRINT_LIGHTS_ON_OFF = True  # Controls printing of zone occupancy status

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cv2.startWindowThread()

# open webcam video stream
cap = cv2.VideoCapture(0)

# the output will be written to output.avi
out = cv2.VideoWriter(
    'output.avi',
    cv2.VideoWriter_fourcc(*'MJPG'),
    15.,
    (640,480))

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
    [0, 220, 240, 480],     # floor zone 1
    [400, 220, 640, 480],   # floor zone 2
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
        bottom_left = (xA, yB)
        bottom_right = (xB, yB)
        
        for zone_num, zone in enumerate(floor_zone_boxes, start=1):
            if is_point_in_box(bottom_left, zone) or is_point_in_box(bottom_right, zone):
                occupied_zones.add(zone_num)
                
    return occupied_zones

# For tracking zone states between frames
prev_zone_states = {}

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # resizing for faster detection
    frame = cv2.resize(frame, (640, 480))

    # using a greyscale picture, also for faster detection
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)



    # Draw stationary red and yellow boxes if enabled via the flag
    if SHOW_STATIONARY_BOXES:
        # Draw quadrant boxes (yellow) - always visible
        for box in stationary_boxes:
            xA, yA, xB, yB = box
            cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 0, 255), 2)  # Red color (BGR format)
        # Draw quadrant boxes (yellow) - always visible
        for box in quadrant_boxes:
            xA, yA, xB, yB = box
            cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 255), 2)  # yellow color (BGR)

    # Draw floor zone boxes if enabled
    if SHOW_FLOOR_ZONE_BOXES:
        for zone_num, box in enumerate(floor_zone_boxes, start=1):
            xA, yA, xB, yB = box
            color = (0, 165, 255)  # Orange
            cv2.rectangle(frame, (xA, yA), (xB, yB), color, 2)
            cv2.putText(frame, f"Zone {zone_num}", (xA+5, yA+20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    # detect people in the image
    boxes, weights = hog.detectMultiScale(frame, winStride=(8,8))
    human_boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

    # Draw human bounding boxes
    for (xA, yA, xB, yB) in human_boxes:
        cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)

    # Get current occupied zones
    occupied_zones = get_occupied_zones(human_boxes)
    
    # Print zone status if changed
    if PRINT_LIGHTS_ON_OFF:
        current_zone_states = {}
        for zone_num in range(1, len(floor_zone_boxes)+1):
            state = "ON" if zone_num in occupied_zones else "OFF"
            current_zone_states[zone_num] = state
            
            # Only print if state changed from previous frame
            if zone_num not in prev_zone_states or prev_zone_states[zone_num] != state:
                print(f"Zone {zone_num}: {state}")
                
        prev_zone_states = current_zone_states

    # Write the output video 
    out.write(frame.astype('uint8'))
    
    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
# and release the output
out.release()
# finally, close the window
cv2.destroyAllWindows()
cv2.waitKey(1)