Following are the links to dataset for training, video for script and video output (demo) :





I have trained the dataset using my laptops GPU and YOLOv8n NANO model is being used
GPU specification : NVIDIA GeForce RTX 3050 Ti 4gb DDR6 

Pre-requisites to run the code :
      1) VS Code 
      2) Python version 3.10 ( minimum)
      3) pip install ultralytics
      4) pip install opencv-python
      5) pip install pandas
      
Optional Pre-requsites (if you wish to train the datasets on your GPU):
      1) NVIDIA CUDA 12.4 Toolkit ( only if you plan to train the model, i have already provided "best.pt" model which is trained on my GPU ) 
      2) pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 (only if you plan to train the model by yourself )

Instructions to run my code :
      1) training.py = it is the code to train the YOLOv8n model
      2) main script.py = it is the main code to detect persons in a video frame and to select an region of interest (ROI) and send an alert message if the certain number of peopel continously stay in that ROI for more than x amoutn of minutes. 

Instructions for the outputs :
      1) video output.mp4 = it is the output video which demonstrates that persons are getting detected with the help of best.pt model and user creating an region of interest (ROI) and getting alerts .
      2) alerts.csv = it is a CSV file it contains the real world time of detection and an alert message
      3) best.pt = the best.pt is the trained model 

---


    
