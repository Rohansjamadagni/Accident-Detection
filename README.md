# Accident-Detection

Real Time Accident Detection using Yolo object detection, openCV and CNN (keras and Tensorflow).

Trained using Google Images, 1200 pictures of accidents occuring on roads and 1300 pictures of roads and traffic.
GSM module is used to send a text message to a particular number for alerting.

The file CNN for accident detection is the CNN model trained with the accident image data. If you would like to train the model on your own data, specify the path accordingly in the 'CNN for accident detection' file and changes can be made to the CNN in this file.

The file Code.ipynb has the Yolo object detection code, weights for the yolo object detection (yolov3.weights)  is down below.


Accident Data : in the folder MyData : [Drive Link](https://drive.google.com/drive/folders/10r6RJOEJ0hl9fuzhShBl84I1LH6mt9K7?usp=sharing)




## Steps to run the file:-

**Firstly, install dependencies mentioned in requirements.txt**

> pip install -r requirements.txt


1. Download model and yolov3.weights(model) from the link https://drive.google.com/drive/folders/10r6RJOEJ0hl9fuzhShBl84I1LH6mt9K7?usp=sharing
2. Place the "yolov3.weights" file in "yolo-coco" directry and the file "model" in root directory.
3. Run Code.ipynb 
4. Run all the cells and enter path of the video as an input when prompted in one of the cells. The files demo.mp4 and demo2.mp4 are examples that can be used.
5. If you wish to run it through webcam, enter 0 as the input.
6. Go to localhost:5000 on windows and 0.0.0.0:5000 on linux
7. The image of the occured accident is stored in the directory images/final_image
8. The templates folder contains the index.html which is the front end implimentation.


```
Note : pip install any packages if they are msising or not mentioned
```
