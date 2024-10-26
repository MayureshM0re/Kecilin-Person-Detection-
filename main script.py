import cv2
import numpy as np
from ultralytics import YOLO
import time
from datetime import datetime
import csv
import os

# Configure your paths here
MODEL_PATH = "best.pt"    
VIDEOP = "video.mp4"  

class PeopleCounter:
    def __init__(self, threshold_count=4, threshold_time=120):
        self.model = YOLO(MODEL_PATH)
        self.threshold_count = threshold_count
        self.threshold_time = threshold_time
        self.roi_start = None
        self.roi_end = None
        self.drawing = False
        self.roi_rectangle = None
        self.alert_start_time = None
        self.alert_active = False
        self.continuous_detection_start = None
        self.frame_time = None
        self.processing_time = 0
        
        # Colors for different detections (BGR format)
        self.roi_color = (0, 0, 255)      # Red for people in ROI
        self.non_roi_color = (255, 165, 0) # Blue for people outside ROI
        
        # Create CSV file for logging with simplified columns
        self.csv_filename = f"alerts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(self.csv_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp', 'People Count In ROI', 'Alert Type'])

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.roi_start = (x, y)
            self.roi_end = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.roi_end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            if self.roi_start and self.roi_end:
                x1 = min(self.roi_start[0], self.roi_end[0])
                y1 = min(self.roi_start[1], self.roi_end[1])
                x2 = max(self.roi_start[0], self.roi_end[0])
                y2 = max(self.roi_start[1], self.roi_end[1])
                self.roi_rectangle = (x1, y1, x2, y2)

    def draw_roi(self, frame):
        clone = frame.copy()
        cv2.namedWindow('Draw ROI')
        cv2.setMouseCallback('Draw ROI', self.mouse_callback)

        print("\nInstructions:")
        print("1. Click and drag to draw rectangular ROI")
        print("2. Press 'c' to confirm ROI")
        print("3. Press 'r' to reset and redraw")

        while True:
            display = clone.copy()
            if self.drawing and self.roi_start and self.roi_end:
                cv2.rectangle(display, self.roi_start, self.roi_end, (0, 255, 0), 2)
            elif self.roi_rectangle:
                x1, y1, x2, y2 = self.roi_rectangle
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.imshow('Draw ROI', display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('c') and self.roi_rectangle:
                break
            elif key == ord('r'):
                self.roi_start = None
                self.roi_end = None
                self.roi_rectangle = None

        cv2.destroyWindow('Draw ROI')

    def is_in_roi(self, bbox):
        if self.roi_rectangle is None:
            return False

        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2

        roi_x1, roi_y1, roi_x2, roi_y2 = self.roi_rectangle
        return (roi_x1 <= center_x <= roi_x2) and (roi_y1 <= center_y <= roi_y2)

    def log_alert(self, roi_count, alert_type):
        try:
            timestamp = datetime.now()
            
            with open(self.csv_filename, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, roi_count, alert_type])
            
        except Exception as e:
            print(f"Error logging alert: {str(e)}")

    def process_frame(self, frame, frame_time):
        if self.frame_time is None:
            self.frame_time = frame_time
            
        time_delta = frame_time - self.frame_time
        self.frame_time = frame_time
        
        processed_frame = frame.copy()
        
        # Draw ROI if exists
        if self.roi_rectangle:
            x1, y1, x2, y2 = self.roi_rectangle
            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(processed_frame, "ROI", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Run YOLOv8 inference
        results = self.model(frame, half=True)[0]
        people_count_roi = 0

        # Process detections
        for detection in results.boxes.data:
            x1, y1, x2, y2, conf, cls = detection
            
            if int(cls) == 0 and conf > 0.3:  # Person class with confidence threshold
                bbox = [int(x1), int(y1), int(x2), int(y2)]
                
                if self.is_in_roi(bbox):
                    people_count_roi += 1
                    color = self.roi_color
                    label = "Person (ROI)"
                else:
                    color = self.non_roi_color
                    label = "Person"
                
                # Draw bounding box and label
                cv2.rectangle(processed_frame, 
                            (bbox[0], bbox[1]), 
                            (bbox[2], bbox[3]), 
                            color, 2)
                cv2.putText(processed_frame, label, 
                          (bbox[0], bbox[1]-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Update alert logic
        if people_count_roi >= self.threshold_count:
            if self.continuous_detection_start is None:
                self.continuous_detection_start = frame_time
                self.log_alert(people_count_roi, "WARNING: People count threshold exceeded in ROI")
            elif frame_time - self.continuous_detection_start >= self.threshold_time:
                if not self.alert_active:
                    self.alert_active = True
                    self.alert_start_time = frame_time
                    self.log_alert(people_count_roi, "ALERT: Continuous threshold violation")
        else:
            self.continuous_detection_start = None
            if self.alert_active:
                self.log_alert(people_count_roi, "RESOLVED: People count returned to normal")
            self.alert_active = False
            self.alert_start_time = None

        # Draw ROI count on frame
        cv2.putText(processed_frame, f"People in ROI: {people_count_roi}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, self.roi_color, 2)

        if self.alert_active:
            cv2.putText(processed_frame, "ALERT: Too many people in ROI!", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            duration = frame_time - self.continuous_detection_start
            cv2.putText(processed_frame, f"Duration: {duration:.1f}s", 
                       (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif self.continuous_detection_start is not None:
            time_left = self.threshold_time - (frame_time - self.continuous_detection_start)
            cv2.putText(processed_frame, f"Warning: {time_left:.1f}s to alert", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        return processed_frame, people_count_roi, self.alert_active

def main():
    try:
        cap = cv2.VideoCapture(VIDEOP)
        
        if not cap.isOpened():
            print(f"Error: Could not open video file: {VIDEOP}")
            return

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_time = 1/fps
        
        # Initialize video writer directly in the current directory
        output_path = f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        # Initialize counter
        counter = PeopleCounter()

        # Get first frame for ROI drawing
        ret, frame = cap.read()
        if ret:
            counter.draw_roi(frame)

        print("\nMonitoring started. Press 'q' to quit.")
        
        start_time = time.time()
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            current_frame_time = frame_count * frame_time
            
            # Process frame
            processed_frame, roi_count, alert = counter.process_frame(frame, current_frame_time)
            
            # Display FPS
            elapsed_time = time.time() - start_time
            fps_actual = frame_count / elapsed_time if elapsed_time > 0 else 0
            cv2.putText(processed_frame, f"FPS: {fps_actual:.1f}", 
                       (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Write frame to output video
            out.write(processed_frame)
            
            cv2.imshow('Frame', processed_frame)
            frame_count += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"An error occurred: {str(e)}")
    
    finally:
        if 'cap' in locals():
            cap.release()
        if 'out' in locals():
            out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()