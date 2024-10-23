Hello, please follow the below instructions to run my code. Thank you!

I have trained the dataset using my laptops GPU and YOLOv8n NANO model is being used
GPU specification: NVIDIA GeForce RTX 3050 Ti 4gb DDR6 

Pre-requisites to run the code :

      1) VS Code 
      
      2) Python version 3.10 ( minimum)
      
      3) pip install ultralytics
      
      4) pip install opencv-python
      
      5) pip install pandas
      
Optional Pre-requsites (if you wish to train the datasets on your GPU):

      1) NVIDIA CUDA 12.4 Toolkit ( only if you plan to train the model, i have already provided "best.pt" model which is trained on my GPU ) 
      
      2) pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 (only if you plan to train the model by yourself )



Following are the links to dataset for training, video for script and video output (demo) :

      1) Dataset = https://drive.google.com/drive/folders/1YadKIxFp-HEukfQOGu9ZfCsw4raF0OUV?usp=drive_link 

      2) Original Video = https://drive.google.com/file/d/14ksZTSkY801unbWeGry3pcYyoC52XRZm/view?usp=drive_link

      3) Video demo/output = 
---

Instructions to run my code :

      1) training.py = it is the code to train the YOLOv8n model
      
      2) main script.py = it is the main code to detect persons in a video frame and to select an region of interest (ROI) and send an alert message if the certain number of peopel continously stay in that ROI for more than x amoutn of minutes. 

Instructions for the outputs :

      1) video output.mp4 = it is the output video which demonstrates that persons are getting detected with the help of best.pt model and user creating an region of interest (ROI) and getting alerts .
      
      2) alerts.csv = it is a CSV file it contains the real world time of detection and an alert message
      
      3) best.pt = the best.pt is the trained model 

---

1. Collect Data:
   
      1) I acquired the data in YOLO format from Roboflow website the images are already annotated and the data is being split in train, valid and test, I have added the Google drive link to access the dataset             and this is the reference from where i acquired the dataset = https://universe.roboflow.com/titulacin/person-detection-9a6mk/dataset/16. The Google drive links to dataset are available at the beginign of          README FILE.
  ---

2. Train Data :
   
     1) To train the data we need to create a data.yaml file, since it was already present with dataset from Roboflow website i didnt need to write data.yaml file 


![Screenshot (27)](https://github.com/user-attachments/assets/b47edb68-a385-4fe9-8ee6-2b3c165c5f02)


names = names is the labelled data name

nc = nc defines no.of classes to detect

test, train and val contains images and annotation in .txt format (YOLO format)

Below you can see I have uplaoded a screenshot after training the dataset,  I have to train on 50 epochs with 8 batch size and you can see it is getting trained on my laptops GPU NVIDIA 3050 TI 4gb

![person detetcion training](https://github.com/user-attachments/assets/d9e5a6aa-9396-4f6a-a586-22385acea2c5)

As you can mAP50 values is 0.818 which is 81.8% average precision which is good enough

---

3. Code Logic explained :

   The main code is present in 'main script.py' file please check mentioned above as well 


      

      
         
      


    
