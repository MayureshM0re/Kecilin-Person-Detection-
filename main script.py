import cv2
import time
import pandas as pd
from ultralytics import YOLO

# Load your YOLO model (trained or pre-trained model)
model = YOLO('best.pt')  # Replace with your trained YOLOv8 model path

# Parameters
video_path = 'video.mp4'  # Path to your video file or 0 for webcam
alert_threshold_people = 3  # Number of people to trigger alert
alert_threshold_time = 60  # Time in seconds to sustain alert condition
alert_records = []

# Variables for ROI selection
roi_selected = False
roi_coords = None  # This will store the coordinates of the selected ROI
start_point = None
end_point = None
alert_start_time = None
alert_triggered = False

# Mouse callback function to draw ROI
def select_roi(event, x, y, flags, param):
    global start_point, end_point, roi_selected, roi_coords
    if event == cv2.EVENT_LBUTTONDOWN:  # When the left mouse button is pressed
        start_point = (x, y)
        end_point = None
    elif event == cv2.EVENT_MOUSEMOVE and start_point:  # As the mouse is moved
        end_point = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:  # When the left mouse button is released
        end_point = (x, y)
        roi_selected = True
        roi_coords = (start_point[0], start_point[1], end_point[0], end_point[1])

# Open the video feed
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Could not open video {video_path}")
    exit()

# Set up a named window and bind the mouse callback function to it
cv2.namedWindow("Video", cv2.WINDOW_NORMAL)  # Allow window to be resizable
cv2.resizeWindow("Video", 1920, 1080)  # Set window size to match video resolution
cv2.setMouseCallback("Video", select_roi)

# Start video processing
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video or failed to read frame.")
        break

    # Run YOLOv8 detection on the entire frame
    results = model(frame, verbose=False)

    # Draw detections on the frame
    for result in results[0].boxes.data:
        x1, y1, x2, y2, conf, cls = map(int, result)
        label = f"Person {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Allow user to select the ROI dynamically
    if not roi_selected:
        # Draw the box being selected
        if start_point and end_point:
            cv2.rectangle(frame, start_point, end_point, (255, 0, 0), 2)  # Draw ROI rectangle
    else:
        # Extract the ROI from the selected coordinates
        x1, y1, x2, y2 = roi_coords
        if x1 < x2 and y1 < y2:  # Ensure ROI is valid
            # Count the number of people detected in the ROI
            people_count = sum(1 for result in results[0].boxes.data if x1 <= result[0] <= x2 and y1 <= result[1] <= y2 and result[5] == 0)  # Class '0' is for 'person'

            # Check if people count exceeds the threshold
            if people_count >= alert_threshold_people:
                if alert_start_time is None:  # Start timing if threshold is exceeded
                    alert_start_time = time.time()  # Record the time when the threshold is first exceeded

                # Check if the time condition is met for sustained alert
                if (time.time() - alert_start_time) >= alert_threshold_time:
                    alert_triggered = True  # Set alert triggered
                    alert_records.append({"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "alert": f"ALERT! More than {alert_threshold_people} people detected."})
            else:
                # Reset alert conditions if count drops below threshold
                alert_start_time = None  
                alert_triggered = False  

            # Draw the ROI rectangle on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Draw ROI rectangle

            # Display the alert message on the video frame if triggered
            if alert_triggered:
                cv2.putText(frame, f"ALERT! More than {alert_threshold_people} people detected.", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)  # Yellow color for alert text
            else:
                # If no alert, show the people count for debugging
                cv2.putText(frame, f"Count: {people_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)  # White color for count

    # Display the video frame
    cv2.imshow("Video", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save alert records to CSV
if alert_records:
    df = pd.DataFrame(alert_records)
    df.to_csv('alerts.csv', index=False)

# Release resources
cap.release()
cv2.destroyAllWindows()
