#!/usr/bin/env python
"""
 Copyright (C) 2018-2019 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

from __future__ import print_function
import sys
import os
from argparse import ArgumentParser, SUPPRESS
import cv2
import time
import logging as log
from openvino.inference_engine import IENetwork, IEPlugin

#Added here
import numpy as np
import imutils
from imutils import build_montages


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-m", "--model", help="Required. Path to an .xml file with a trained model.",
                      required=True, type=str)
    args.add_argument("-i", "--input",
                      help="Required. Path to video file or image. 'cam' for capturing video stream from camera",
                      required=True, type=str)
    #Added here
    args.add_argument("-i2", "--input2",
                      help="Optional. Path to second video file or image. 'cam' for capturing video stream from camera",
                      default=None, type=str)
    args.add_argument("-l", "--cpu_extension",
                      help="Optional. Required for CPU custom layers. Absolute path to a shared library with the "
                           "kernels implementations.", type=str, default=None)
    args.add_argument("-pp", "--plugin_dir", help="Optional. Path to a plugin folder", type=str, default=None)
    args.add_argument("-d", "--device",
                      help="Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL or MYRIAD is "
                           "acceptable. The demo will look for a suitable plugin for device specified. "
                           "Default value is CPU", default="CPU", type=str)
    args.add_argument("--labels", help="Optional. Path to labels mapping file", default=None, type=str)
    args.add_argument("-pt", "--prob_threshold", help="Optional. Probability threshold for detections filtering",
                      default=0.5, type=float)

    return parser


def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()
    model_xml = args.model
    model_bin = os.path.splitext(model_xml)[0] + ".bin"
    # Plugin initialization for specified device and load extensions library if specified
    log.info("Initializing plugin for {} device...".format(args.device))
    plugin = IEPlugin(device=args.device, plugin_dirs=args.plugin_dir)
    if args.cpu_extension and 'CPU' in args.device:
        plugin.add_cpu_extension(args.cpu_extension)
    # Read IR
    log.info("Reading IR...")
    net = IENetwork(model=model_xml, weights=model_bin)

    if plugin.device == "CPU":
        supported_layers = plugin.get_supported_layers(net)
        not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
        if len(not_supported_layers) != 0:
            log.error("Following layers are not supported by the plugin for specified device {}:\n {}".
                      format(plugin.device, ', '.join(not_supported_layers)))
            log.error("Please try to specify cpu extensions library path in demo's command line parameters using -l "
                      "or --cpu_extension command line argument")
            sys.exit(1)
    assert len(net.inputs.keys()) == 1, "Demo supports only single input topologies"
    assert len(net.outputs) == 1, "Demo supports only single output topologies"
    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))
    
    #Added here
    input_blob2 = next(iter(net.inputs))
    out_blob2 = next(iter(net.outputs))
    
    log.info("Loading IR to the plugin...")
    exec_net = plugin.load(network=net, num_requests=2)
    # Read and pre-process input image
    n, c, h, w = net.inputs[input_blob].shape
    #Added here
    n2,c2,h2,w2 = net.inputs[input_blob2].shape
    del net
    #added from here
    if args.input == 'cam':
        input_stream = 0
    elif args.input == 'gstreamer':
        #gst rtp sink
        input_stream = 'udpsrc port=5000 caps = " application/x-rtp, encoding-name=JPEG,payload=26" ! rtpjpegdepay ! decodebin ! videoconvert ! appsink'
    else:
        input_stream = args.input
        assert os.path.isfile(args.input), "Specified input file doesn't exist"
    if args.labels:
        with open(args.labels, 'r') as f:
            labels_map = [x.strip() for x in f]
    else:
        labels_map = None
    
    if input_stream == 'gstreamer':
      cap = cv2.VideoCapture(input_stream,cv2.CAP_GSTREAMER)
    else:
      cap = cv2.VideoCapture(input_stream)

    if args.input2 == 'cam':
        input_stream2 = 0
    elif args.input2 == 'gstreamer':
        #gst rtp sink
        input_stream2 = 'udpsrc port=5001 caps = " application/x-rtp, encoding-name=JPEG,payload=26" ! rtpjpegdepay ! decodebin ! videoconvert ! appsink'
    else:
        input_stream2 = args.input2
        assert os.path.isfile(args.input2), "Specified input file doesn't exist"
    if args.labels:
        with open(args.labels, 'r') as f:
            labels_map = [x.strip() for x in f]
    else:
        labels_map = None

    if input_stream2 == 'gstreamer':
      cap2 = cv2.VideoCapture(input_stream2,cv2.CAP_GSTREAMER)
    else:
      cap2 = cv2.VideoCapture(input_stream2)
      
    cur_request_id = 0
    next_request_id = 1
    
    #Added here
    cur_request_id2 = 1
    next_request_id2 = 0

    log.info("Starting inference in async mode...")
    log.info("To switch between sync and async modes press Tab button")
    log.info("To stop the demo execution press Esc button")
    
    #Async doesn't work if True
    #Request issues = Runtime Error: [REQUEST BUSY]
    is_async_mode = False
    render_time = 0
    ret, frame = cap.read()
    ret2, frame2 = cap2.read()
    
    #Montage width and height
    #In this case means 2x1 boxes
    mW = 2
    mH = 1

    frameList = []

    print("To close the application, press 'CTRL+C' or any key with focus on the output window")
    while cap.isOpened() or cap2.isOpened():
        if is_async_mode:
            ret, next_frame = cap.read()
            ret2, next_frame2 = cap2.read()
        else:
            ret, frame = cap.read()
            ret2, frame2 = cap2.read()
        if not ret and ret2:
            break
        initial_w = cap.get(3)
        initial_h = cap.get(4)
        initial_w2 = cap2.get(3)
        initial_h2 = cap2.get(4)
        # Main sync point:
        # in the truly Async mode we start the NEXT infer request, while waiting for the CURRENT to complete
        # in the regular mode we start the CURRENT request and immediately wait for it's completion
        inf_start = time.time()
        if is_async_mode:
            in_frame = cv2.resize(next_frame, (w, h))
            in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
            in_frame = in_frame.reshape((n, c, h, w))
            exec_net.start_async(request_id=next_request_id, inputs={input_blob: in_frame})

            in_frame2 = cv2.resize(next_frame2, (w2, h2))
            in_frame2 = in_frame2.transpose((2, 0, 1))  # Change data layout from HWC to CHW
            in_frame2 = in_frame2.reshape((n2, c2, h2, w2))
            exec_net.start_async(request_id=next_request_id2, inputs={input_blob2: in_frame2})
            
        else:
            in_frame = cv2.resize(frame, (w, h))
            in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
            in_frame = in_frame.reshape((n, c, h, w))
            exec_net.start_async(request_id=cur_request_id, inputs={input_blob: in_frame})

            in_frame2 = cv2.resize(frame2, (w2, h2))
            in_frame2 = in_frame2.transpose((2, 0, 1))  # Change data layout from HWC to CHW
            in_frame2 = in_frame2.reshape((n2, c2, h2, w2))
            exec_net.start_async(request_id=cur_request_id2, inputs={input_blob2: in_frame2})
            
        if exec_net.requests[cur_request_id].wait(-1) == 0 and exec_net.requests[cur_request_id2].wait(-1) == 0:
            inf_end = time.time()
            det_time = inf_end - inf_start

            # Parse detection results of the current request
            res = exec_net.requests[cur_request_id].outputs[out_blob]
            res2 = exec_net.requests[cur_request_id2].outputs[out_blob2]
            
            for obj in res[0][0]:
                # Draw only objects when probability more than specified threshold
                if obj[2] > args.prob_threshold:
                    xmin = int(obj[3] * initial_w)
                    ymin = int(obj[4] * initial_h)
                    xmax = int(obj[5] * initial_w)
                    ymax = int(obj[6] * initial_h)
                    class_id = int(obj[1])
                    # Draw box and label\class_id
                    color = (min(class_id * 12.5, 255), min(class_id * 7, 255), min(class_id * 5, 255))
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                    det_label = labels_map[class_id] if labels_map else str(class_id)
                    cv2.putText(frame, det_label + ' ' + str(round(obj[2] * 100, 1)) + ' %', (xmin, ymin - 7),
                                cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)
            for obj in res2[0][0]:
                # Draw only objects when probability more than specified threshold
                if obj[2] > args.prob_threshold:
                    xmin = int(obj[3] * initial_w2)
                    ymin = int(obj[4] * initial_h2)
                    xmax = int(obj[5] * initial_w2)
                    ymax = int(obj[6] * initial_h2)
                    class_id = int(obj[1])
                    # Draw box and label\class_id
                    color = (min(class_id * 12.5, 255), min(class_id * 7, 255), min(class_id * 5, 255))
                    cv2.rectangle(frame2, (xmin, ymin), (xmax, ymax), color, 2)
                    det_label = labels_map[class_id] if labels_map else str(class_id)
                    cv2.putText(frame2, det_label + ' ' + str(round(obj[2] * 100, 1)) + ' %', (xmin, ymin - 7),
                                cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)

            # Draw performance stats
            inf_time_message = "Inference time: N\A for async mode" if is_async_mode else \
                "Inference time: {:.3f} ms".format(det_time * 1000)
            render_time_message = "OpenCV rendering time: {:.3f} ms".format(render_time * 1000)
            async_mode_message = "Async mode is on. Processing request {}".format(cur_request_id) if is_async_mode else \
                "Async mode is off. Processing request {}".format(cur_request_id)

            cv2.putText(frame, inf_time_message, (15, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)
            cv2.putText(frame, render_time_message, (15, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (10, 10, 200), 1)
            cv2.putText(frame, async_mode_message, (10, int(initial_h - 20)), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                        (10, 10, 200), 1)

            cv2.putText(frame2, inf_time_message, (15, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)
            cv2.putText(frame2, render_time_message, (15, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (10, 10, 200), 1)
            cv2.putText(frame2, async_mode_message, (10, int(initial_h - 20)), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                        (10, 10, 200), 1)

        
        render_start = time.time()
        
        #Added here
        #Add the frame into a list
        frameList.append(frame)
        frameList.append(frame2)
        
        #build_montages function from imutils to display 2 cameras on a single dashboard
        
        montages = build_montages(frameList, (640,480), (mW, mH))
    
        for montage in montages:
          cv2.imshow("Detection results" , montage)

        
        #cv2.imshow("Detection Results", frame)
        render_end = time.time()
        render_time = render_end - render_start

        if is_async_mode:
            cur_request_id, next_request_id = next_request_id, cur_request_id
            
            frame = next_frame
            frame2 = next_frame2
        key = cv2.waitKey(1)
        if key == 27:
            break
        if (9 == key):
            is_async_mode = not is_async_mode
            log.info("Switched to {} mode".format("async" if is_async_mode else "sync"))
    
    cap.release()
    cap2.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    sys.exit(main() or 0)
