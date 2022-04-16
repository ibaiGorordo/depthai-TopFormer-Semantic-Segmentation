#!/usr/bin/env python3
from time import monotonic
import cv2
import numpy as np
import depthai as dai
import pafy

from utils import util_draw_seg, FpsUpdater

def createPipeline(nn_path):

    # Start defining a pipeline
    pipeline = dai.Pipeline()

    pipeline.setOpenVINOVersion(version = dai.OpenVINO.VERSION_2021_4)

    # Define Input node
    xinFrame = pipeline.create(dai.node.XLinkIn)
    xinFrame.setStreamName("inFrame")

    # Define a neural network that will make predictions based on the source frames
    nnNode = pipeline.create(dai.node.NeuralNetwork)
    nnNode.setBlobPath(nn_path)
    nnNode.setNumPoolFrames(4)
    nnNode.input.setBlocking(False)
    nnNode.setNumInferenceThreads(2)

    # Define model output node
    nnOut = pipeline.create(dai.node.XLinkOut)
    nnOut.setStreamName("nn")
    nnOut.input.setBlocking(False)

    # Link nodes
    xinFrame.out.link(nnNode.input)
    nnNode.out.link(nnOut.input)

    return pipeline

def to_planar(arr: np.ndarray, shape: tuple) -> np.ndarray:
    return cv2.resize(arr, (shape[1],shape[0])).transpose(2, 0, 1).flatten()

if __name__ == '__main__':
        
    nn_path = "models/topformerb_openvino_2021.4_8shave_512x896.blob"
    nn_shape = (512,896) # Height, Width
    num_of_classes = 150 # define the number of classes in the dataset

    pipeline = createPipeline(nn_path)
    img = dai.ImgFrame()

    # Pipeline defined, now the device is assigned and pipeline is started
    with dai.Device() as device:

        device.startPipeline(pipeline)

        # Input queue will be used to send video frames to the device.
        qIn = device.getInputQueue(name="inFrame")

        # Output queue will be used to get nn data from the video frames.
        qSeg = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

        # Initialize video
        videoUrl = 'https://youtu.be/fzZEylhZTbI'
        videoPafy = pafy.new(videoUrl)
        print(videoPafy.streams)
        cap = cv2.VideoCapture(videoPafy.streams[-1].url)
        # cap = cv2.VideoCapture("input.mp4")

        # skip first {start_time} seconds
        start_time = 12 
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_time*30)

        fpsUpdate = FpsUpdater()
        
        cv2.namedWindow("Semantic Sementation", cv2.WINDOW_NORMAL)  
        while cap.isOpened():
            # Press key q to stop
            if cv2.waitKey(1) == ord('q'):
                break

            try:
                # Read frame from the video
                ret, frame = cap.read()
                if not ret: 
                    break
            except:
                continue
 
            img = dai.ImgFrame()
            img.setData(to_planar(frame, nn_shape))
            img.setTimestamp(monotonic())
            img.setWidth(nn_shape[1])
            img.setHeight(nn_shape[0])
            qIn.send(img)

            inSeg = qSeg.get()

            if inSeg is not None:

                # Get segmentation map
                seg_map = np.array(inSeg.getFirstLayerInt32()).reshape(nn_shape[0]//8,nn_shape[1]//8)

                # Update fps
                fps = fpsUpdate()
                fps_text = f"FPS: {int(fps)}"

                # Draw segmentation
                combined_img = util_draw_seg(seg_map, frame, alpha=0.5)

                cv2.putText(combined_img, fps_text, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.imshow("Semantic Sementation", combined_img)


            # Press key q to stop
            if cv2.waitKey(1) == ord('q'):
                break