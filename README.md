# Computer Pointer Controller

This project controls a computer's pointer using the gaze of the eyes. To accomplish this, four pre-trained 
models from the [Intel Pre-trained Models Zoo](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit/pretrained-models.html) have to be used.

- [Face Detection](https://docs.openvinotoolkit.org/latest/_models_intel_face_detection_adas_binary_0001_description_face_detection_adas_binary_0001.html)
- [Head Pose Estimation](https://docs.openvinotoolkit.org/latest/_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html)
- [Facial Landmarks Detection](https://docs.openvinotoolkit.org/latest/_models_intel_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html)
- [Gaze Estimation Model](https://docs.openvinotoolkit.org/latest/_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html)

![Flow of Data](bin/images/Flow_of_data.png)

## Project Set Up and Installation

The installation instructions for the [OpenVINO Toolkit](https://docs.openvinotoolkit.org/latest/index.html), 
can be found [here](https://docs.openvinotoolkit.org/latest/index.html)

This project utilizes the InferenceEngine API from Intel's OpenVino ToolKit to build the project. 
The gaze estimation model requires three inputs:
 - Image of the right eye
 - Image of the left eye
 - The head pose
 
Start by cloning [this GitHub Repo](https://github.com/sam505/Computer_pointer_controller). This repository contains
 all the pre-trained models required to run the project on your machine locally. In addition to that, 
 it contains the required packages that you will have to install to run this project without encountering errors.
    
Luckily, for this project all the required packages have been set up in the virtual environment. 
There are four main directories for this project
- [bin](bin) ~ This contains the video input video to test the app and also has a directory containing images 
illustrating how the project works
- [models](models) ~ This contains all the pre-trained models required to run the project. They have been downloaded 
from the [Intel Pre-trained Models Zoo](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit/pretrained-models.html)
   
    - [Face Detection](https://docs.openvinotoolkit.org/latest/_models_intel_face_detection_adas_binary_0001_description_face_detection_adas_binary_0001.html)
    - [Head Pose Estimation](https://docs.openvinotoolkit.org/latest/_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html)
    - [Facial Landmarks Detection](https://docs.openvinotoolkit.org/latest/_models_intel_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html)
    - [Gaze Estimation Model](https://docs.openvinotoolkit.org/latest/_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html)
    
- [src](src) ~ Contains all the python files required to control the flow of data from either the video file, webcam 
feed or an image through the four models and obtain results that will be used to control the computer pointer.
There are 7 python files all performing different tasks. Running the app, loading the four different models, 
loading the input file and the last one controls the computer pointer.
- [venv](venv) ~ The virtual environment folder that contains all additional packages required while running the project

- There is a [Readme](README.md) and a [requirements file](requirements.txt)

    To install the required packages listen in the requirements file use
    
     ```commandline
    pip install PACKAGE_NAME
    ```


## Demo

There are two ways to run this project. 
- Running the [main](src/main.py) python file using a command line argument.
Before that, assuming that you have already installed the [OpenVINO Toolkit](https://docs.openvinotoolkit.org/latest/index.html),
source the OpenVINO Environment using
```commandline
source /opt/intel/openvino/bin/setupvars.sh -pyver 3.6
```
Afterwards, run the main python file using either of the following command lines

   - To run the project using the webcam as the input type, use the following the command line. 
   The command line argument specifying the input type is required to tun the main python file
        ```commandline
        python src/main.py --input_type cam
        ```
        While using the video or an image as the input type, a command line argument specifying the directory 
   path of the input video or input image has to be specified by using one of the following command lines. 
   For an image file specify the command line argument `--input_type` as video and add the path to the video in the `--input_type` command line argument
   
        ```commandline
        python src/main.py --input_type video --input_file path_to_video_file
        ```
        
        For an image file specify the command line argument `--input_type` as image and add the path to the image in the `--input_type` command line argument
        
        ```commandline
        python src/main.py --input_type image --input_file path_to_image_file
        ```
## Documentation

The project requires different command line arguments depending on your desired input file to the project. 
- While using the webcam as the input feed source, to run the main python file, a command line argument `--input_type` 
has to be added with the parameter `cam` to specify the input type as webcam.

- To use a video file as the source of the input feed, running the main python file though the command line, a command
 line argument `--input_type` has to be added with the parameter `video` to specify the input type as video. In addition to that, 
 to use a different video or the one in the project directory, add the path to the video file to the `--input_file` command line argument.
 
- To use a image file as the source of the input feed, running the main python file though the command line, a command
 line argument `--input_type` has to be added with the parameter `image` to specify the input type as an image. In addition to that, 
 add the path to the image file to the `--input_file` command line argument.


## Benchmarks
*TODO:* Include the benchmark results of running your model on multiple hardwares and multiple model precisions. 
Your benchmarks can include: model loading time, input/output processing time, model inference time etc.
### FP32 Results
The face detection model uses one precision, that is, FP32-INT1. The loading time and the inference time for the
 model remains the same.
 
- Results for all the models have been calculated while running the project and are as shown below.
 
     ![FP32 results](bin/images/fp32_results.png)

   - Face Detection Model Deep Learning Workbench Results using FP32-INT1 as the model precision.
 
     ![Face Detection Model results](bin/images/fd_results.png)

   - Facial Landmarks Detection Model Deep Learning Workbench results using FP32 as the model precision
     ![Facial Landmarks FP32 DL Workbench Results](bin/images/fld_fp32_results.png)
   
   - Head Pose Estimation Model Deep Learning Workbench Results using FP32 as the model precision
   
     ![Head Pose Estimation FP32 DL Workbench results](bin/images/hpe_fp32_results.png)
   
   - Gaze Estimation Model Deep Learning Workbench Results using FP32 as the model precision
 
     ![Gaze Estimation FP32 DL Workbench results](bin/images/ge_fp32_results.png)
 
### FP16 Results
- Below are the results for running the project using FP16 as the model precision
 
    ![FP16 results](bin/images/fp16_results.png)
    
    - Facial Landmarks Detection Model Deep Learning Workbench results using FP32 as the model precision
      ![Facial Landmarks FP16 DL Workbench Results](bin/images/fld_fp16_results.png)
    
    - Head Pose Estimation Model Deep Learning Workbench Results using FP32 as the model precision
      ![Head Pose Estimation FP16 DL Workbench results](bin/images/hpe_fp16_results.png)
      
    - Gaze Estimation Model Deep Learning Workbench Results using FP32 as the model precision
      ![Gaze Estimation FP16 DL Workbench results](bin/images/ge_fp16_results.png)

### FP16-INT8 Results
- Below are the results for running the project using FP16-INT8 as the model precision
    ![FP16-INT8 Results](bin/images/fp16-int8_results.png)
    
    - Facial Landmarks Detection Model Deep Learning Workbench results using FP32 as the model precision
       ![Facial Landmarks FP16-INT8 DL Workbench Results](bin/images/fld_fp16-int8_results.png)
       
    - Head Pose Estimation Model Deep Learning Workbench Results using FP32 as the model precision
      ![Head Pose Estimation FP16-INT8 DL Workbench results](bin/images/hpe_fp16-int8_results.png)
      
    - Gaze Estimation Model Deep Learning Workbench Results using FP32 as the model precision
      ![Gaze Estimation FP16-INT8 DL Workbench results](bin/images/ge_fp16-int8_results.png)

## Results
*TODO:* Discuss the benchmark results and explain why you are getting the results you are getting. 
For instance, explain why there is difference in inference time for FP32, FP16 and INT8 models.

## Stand Out Suggestions
This is where you can provide information about the stand out suggestions that you have attempted.

### Async Inference
If you have used Async Inference in your code, benchmark the results and explain its effects on power and 
performance of your project.

### Edge Cases
There will be certain situations that will break your inference flow. For instance, 
lighting changes or multiple people in the frame. Explain some of the edge cases you encountered in your project 
and how you solved them to make your project more robust.
