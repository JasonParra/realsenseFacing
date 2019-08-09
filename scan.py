

## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################

import pyrealsense2 as rs
import numpy as np
import cv2

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

cascPath = "./src/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame and not depth_frame:
            continue

          # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())

        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        far=""
        farColor=(0,0,0)
        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            dist = depth_frame.get_distance(x, y)
            if(dist>=1):
                far = "Lejos"
                farColor=(0,255,0)
            elif (dist>=0.7 and dist<1):
                far = "normal"
                farColor=(0,255,255)
            else:
                far = "cerca"
                farColor=(0,0,255)

            cv2.putText(color_image,far,(5,15),5, 1,farColor, lineType=cv2.LINE_AA)
            cv2.rectangle(color_image, (x, y), (x+w, y+h),farColor, 2)
            cv2.putText(color_image, "(" + str(x) +","+str(y) +","+ str(round(dist,2)) +")",(x,y - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.6, farColor, lineType=cv2.LINE_AA)
      

        # Show images
        cv2.imshow('RealSense', color_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:

    # Stop streaming
    pipeline.stop()