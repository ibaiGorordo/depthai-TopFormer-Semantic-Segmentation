#!/usr/bin/env python3
import cv2
import numpy as np
import depthai as dai

from utils import util_draw_seg, FpsUpdater

def createPipeline(nn_path, nn_shape, cam_source='rgb'):

    # Start defining a pipeline
    pipeline = dai.Pipeline()

    pipeline.setOpenVINOVersion(version = dai.OpenVINO.VERSION_2021_4)

    # Define a neural network that will make predictions based on the source frames
    detection_nn = pipeline.create(dai.node.NeuralNetwork)
    detection_nn.setBlobPath(nn_path)

    detection_nn.setNumPoolFrames(4)
    detection_nn.input.setBlocking(False)
    detection_nn.setNumInferenceThreads(2)

    cam=None
    # Define a source - color camera
    if cam_source == 'rgb':
        cam = pipeline.create(dai.node.ColorCamera)
        cam.setPreviewSize(nn_shape[1],nn_shape[0])
        cam.setInterleaved(False)
        cam.setPreviewKeepAspectRatio(True)
        cam.preview.link(detection_nn.input)
    elif cam_source == 'left':
        cam = pipeline.create(dai.node.MonoCamera)
        cam.setBoardSocket(dai.CameraBoardSocket.LEFT)
    elif cam_source == 'right':
        cam = pipeline.create(dai.node.MonoCamera)
        cam.setBoardSocket(dai.CameraBoardSocket.RIGHT)

    if cam_source != 'rgb':
        manip = pipeline.create(dai.node.ImageManip)
        manip.setResize(nn_shape[1],nn_shape[0])
        manip.setKeepAspectRatio(True)
        manip.setFrameType(dai.RawImgFrame.Type.BGR888p)
        cam.out.link(manip.inputImage)
        manip.out.link(detection_nn.input)

    cam.setFps(20)

    # Create outputs
    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("nn_input")
    xout_rgb.input.setBlocking(False)

    detection_nn.passthrough.link(xout_rgb.input)

    xout_nn = pipeline.create(dai.node.XLinkOut)
    xout_nn.setStreamName("nn")
    xout_nn.input.setBlocking(False)

    detection_nn.out.link(xout_nn.input)

    return pipeline


if __name__ == '__main__':
        
    nn_path = "models/topformers_openvino_2021.4_6shave.blob"
    nn_shape = (512,512) # Height, Width
    num_of_classes = 150 # define the number of classes in the dataset

    pipeline = createPipeline(nn_path, nn_shape, cam_source='rgb')

    # Pipeline defined, now the device is assigned and pipeline is started
    with dai.Device() as device:

        device.startPipeline(pipeline)

        # Output queues will be used to get the rgb frames and nn data from the outputs defined above
        q_nn_input = device.getOutputQueue(name="nn_input", maxSize=4, blocking=False)
        q_nn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

        fpsUpdate = FpsUpdater()

        cv2.namedWindow("Semantic Sementation", cv2.WINDOW_NORMAL)  
        while True:
            
            # Read processed image frame
            in_nn_input = q_nn_input.get()
            frame = in_nn_input.getCvFrame()

            # Read segmentation map
            in_nn = q_nn.get()
            seg_map = np.array(in_nn.getFirstLayerInt32()).reshape(nn_shape[0]//8,nn_shape[1]//8)

            # Update fps
            fps = fpsUpdate()
            fps_text = f"FPS: {int(fps)}"

            # Draw combined image
            combined_img = util_draw_seg(seg_map, frame, alpha=0.5)
            cv2.putText(combined_img, fps_text, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            
            cv2.imshow("Semantic Sementation", combined_img)
            # Press key q to stop
            if cv2.waitKey(1) == ord('q'):
                break
