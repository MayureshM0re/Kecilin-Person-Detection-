Hello, please kindly read all the instructions. Thank you!

I have trained the dataset using my laptop GPU and YOLOv8n NANO model is being used
GPU specification: NVIDIA GeForce RTX 3050 Ti 4gb DDR6.

The input video I have used is in a loop originally it was 11 seconds I have made a loop of to achieve the target of recording more than 2 minutes.


Pre-requisites to run the code :

      1) VS Code 
      
      2) Python version 3.10 ( minimum)
      
      3) pip install ultralytics
      
      4) pip install opencv-python
      
      
Optional Pre-requsites (if you wish to train the datasets on your GPU):

      1) NVIDIA CUDA 12.4 Toolkit ( only if you plan to train the model, i have already provided "best.pt" model which is trained on my GPU ) 
      
      2) pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 (only if you plan to train the model by yourself )



### Google Drive Links to Output Video, Original Video and Dataset :

## 1) Output Video = 
      
https://drive.google.com/file/d/1y8_R-pkyOK2CC2zHkzABeQcZqpPWuKBj/view?usp=sharing
      
## 2) Original Video = 
      
https://drive.google.com/file/d/14ksZTSkY801unbWeGry3pcYyoC52XRZm/view?usp=drive_link
      
## 3) Dataset =
       
https://drive.google.com/drive/folders/1ryPoPAExnU47_p5yQhY9qIsxEpRB_bfv?usp=sharing

---

Instructions for the outputs :

      1) video output.mp4 = it is the output video which demonstrates that persons are getting detected with the help of best.pt model and user creating an region of interest (ROI) and getting alerts .
      
      2) alerts.csv = it is a CSV file it contains the real world time of detection and an alert message
      
      3) best.pt = the best.pt is the trained model


Instructions to run my code :

      1) training.py = it is the code to train the YOLOv8n model
      
      2) main script.py = it is the main code to detect persons in a video frame and to select an region of interest (ROI) and send an alert message if the certain number of peopel continously stay in that ROI for more than x amoutn of minutes. 


---

## 1. Collect Data:

---

 1) I acquired the data in YOLO format from Roboflow website the images are already annotated and the data is being split in train, valid and test, I have added the Google drive link to access the dataset             and this is the reference from where i acquired the dataset = https://universe.roboflow.com/titulacin/person-detection-9a6mk/dataset/16. The Google drive links to dataset are available at the beginign of          README FILE.
    
  ---

## 2. Train Data :

---
   
 1) To train the data we need to create a data.yaml file, since it was already present with dataset from Roboflow website i didnt need to write data.yaml file 


![Screenshot (27)](https://github.com/user-attachments/assets/b47edb68-a385-4fe9-8ee6-2b3c165c5f02)


names = names is the labelled data name

nc = nc defines no.of classes to detect

test, train and val contains images and annotation in .txt format (YOLO format)

Below you can see I have uplaoded a screenshot after training the dataset,  I have to train on 50 epochs with 8 batch size and you can see it is getting trained on my laptops GPU NVIDIA 3050 TI 4gb

![person detetcion training](https://github.com/user-attachments/assets/d9e5a6aa-9396-4f6a-a586-22385acea2c5)

As you can mAP50 values is 0.818 which is 81.8% average precision which is good enough

---


# Code Logic Walkthrough :
 ---
   
## 1)
   
   ![1](https://github.com/user-attachments/assets/0fe622f2-b4f4-4d58-b772-7e0def883fc8)


The necessary libraries are imported:
cv2: Provides OpenCV functions for computer vision tasks.
numpy: Used here for numerical operations.
YOLO (from ultralytics): Loads the YOLO model for person detection.
time, datetime: Used for timing and timestamping events.
csv, os: Handle logging of detections and file operations.

threshold_count: The minimum number of detected people required to trigger an alert.
threshold_time: Time in seconds that this threshold must be met for an alert.
roi_rectangle: Stores coordinates of the Region of Interest (ROI) drawn by the user.
alert_active: Keeps track if an alert is currently active.
The YOLO model is loaded using self.model = YOLO(MODEL_PATH).

a CSV file with a unique timestamped filename to log alerts. It initializes the file with headers: 'Timestamp', 'People Count In ROI', 'Alert Type'.

---

## 2)    

![2](https://github.com/user-attachments/assets/190f8195-7719-489d-a340-13a1a403d461)

This method enables the user to draw a rectangular ROI by clicking and dragging with the mouse on the video frame:

EVENT_LBUTTONDOWN: Begins the drawing, setting roi_start and roi_end to the initial click position.
EVENT_MOUSEMOVE: Updates roi_end as the mouse is dragged.
EVENT_LBUTTONUP: Ends the drawing and calculates the bounding box coordinates (roi_rectangle).

The mouse_callback function is used to allow users to draw a ROI rectangle on the frame.
The c key confirms the selection, and the r key resets the drawing process.

---


## 3) 

![3](https://github.com/user-attachments/assets/4708d7fe-d18f-41c5-b67f-eb9ec1e9ff41)

if a person detected in a bounding box (bbox) is within the ROI. The bounding box’s center coordinates are compared with the ROI boundaries.

def log_alert appends a new entry to the CSV file with the current timestamp, count of people in the ROI, and the alert type (WARNING, ALERT, RESOLVED).

def process_frame
This function processes each frame to:

Draw the ROI rectangle.
Run YOLO detection (results = self.model(frame, half=True)[0]) and process detections.

---


## 4)

![4](https://github.com/user-attachments/assets/9258f509-9835-44d5-a345-73930f7242a3)


This part iterates through each detection, filtering only those classified as persons with high confidence (above 0.3). If the detected person is within the ROI, people_count_roi is incremented.

If the count of people in the ROI exceeds the threshold, a warning is logged. If this count persists beyond threshold_time, an alert is activated and logged.

---

## 5) 

![5](https://github.com/user-attachments/assets/ed648c0e-70ef-4d3a-8407-8978445db83c)

This is the main loop that:

Opens the video and reads properties.
Instantiates PeopleCounter.
Reads each frame and processes it to detect and annotate people, while writing the output to an MP4 file.






