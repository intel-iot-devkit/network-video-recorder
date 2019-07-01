# Network Video Recorder with the Intel® Distribution of OpenVINO™ Toolkit Integrated

The need to accommodate high resolution video for management, storage, and viewing are essential for smart video surveillance systems ensuring the highest standard of security. This solution demonstrates how to implement and take advantage of Intel® hardware platforms for end to end video analytics involving decoding, encoding, and optimization using various media stacks. 

Target Operating System	Ubuntu* 16.04 LTS
Time to Complete	45 minutes

[button] GitHub (Python*)

[Action Links]
What You Will Learn | How It Works | What You Need | Tools We Used

[SECTION]
WHAT YOU WILL LEARN
This application uses Intel® Media SDK for encoding and decoding high resolution video streams into a server machine for storage and management
Gain insight into the following solutions:
•	Create and run inference workloads that provide low latency video processing 
•	Encode and Decode Multiple Video Stream
•	Gather data and send back to the edge and or datacenter
Learn to build and run an application with these capabilities:
1.	Decode and Decode using a multimedia framework library
2.	Transform and resize frames using computer vision
3.	Enable Deep Learning to classify frames
4.	Run filters and count matches in the post-process
5.	Create a detector processor
6.	Optimize solution on various Intel® hardware

[SECTION ]
HOW IT WORKS
This application uses 2 video producing terminals in which their streams are decoded into a server machine.
1.	The video streams are captured using the multimedia frameworks and encoded into a format that can be accommodated by the server machine
2.	The stored video in the server machine is then streamed into a computer vision application which performs inference detecting people.
3.	This inferred video is then sent to an additional system for further analysis.

[SECTION - 2 COLUMNS]
WHAT YOU NEED

Hardware Requirements
Choose one of the following kits:
2 - UP Squared* AI Vision X Development Kit + USB webcam

Software Requirements
Ubuntu 16.04 LTS (preinstalled on the hardware)
imutils
TOOLS WE USED 
Intel® Distribution of OpenVINO™ toolkit (Release 1)
A multiplatform computer vision solution.
[button] Free Download [button] Get Started [button] Training

Gstreamer

A framework for creating media applications.

[button] Free Download [button] Get Started




[SECTION]
INSTRUCTIONS
Install imutils library:

pip install imutils

Stream camera with Gstreamer on UP Squared* board 1:

$ gst-launch-1.0 -v v4l2src device=/dev/video0 ! "image/jpeg,width=640, height=480,framerate=30/1" ! rtpjpegpay ! udpsink host=<host_ip> port=5000

Stream camera with Gstreamer on UP Squared board 2:

$ gst-launch-1.0 -v v4l2src device=/dev/video0 ! "image/jpeg,width=640, height=480,framerate=30/1" ! rtpjpegpay ! udpsink host=<host_ip> port=5001

Run detection script on host machine:
$ source /opt/intel/openvino/bin/setupvars.sh

$ python3 multicam_detection.py -i gstreamer -i2 gstreamer -m /home/harichand/openvino_models/ir/FP32/Retail/object_detection/pedestrian/rmnet_ssd/0013/dldt/person-detection-retail-0013.xml -l /home/harichand/inference_engine_samples_build/intel64/Release/lib/libcpu_extension.so


[SECTION]
OUTPUT
