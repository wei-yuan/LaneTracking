#!/usr/bin/python
from __future__ import division

import cv2

import track
import detect

vidoe_path = '/home/alex504/img_video_file/kalman/road_view.mp4'

def main(input_vidoe_path):     
    # input video
    cap = cv2.VideoCapture(input_vidoe_path)    
    # initialize detector and tracker here
    lt = track.LaneTracker(2, 0.1, 500)
    ld = detect.LaneDetector(180)
    # initialize time tick
    ticks = 0
    
    while cap.isOpened():
        # read frame
        ret, frame = cap.read()
        # start count tick
        precTick = ticks
        ticks = cv2.getTickCount()
        dt = (ticks - precTick) / cv2.getTickFrequency()        
        # start predict and detect lane
        # Kalman filter predict step
        predicted = lt.predict(dt)
        lanes = ld.detect(frame)

        if predicted is not None:
            cv2.line(frame, (predicted[0][0], predicted[0][1]), (predicted[0][2], predicted[0][3]), (0, 0, 255), 5)
            cv2.line(frame, (predicted[1][0], predicted[1][1]), (predicted[1][2], predicted[1][3]), (0, 0, 255), 5)
        # Kalman filter update step
        lt.update(lanes)

        cv2.imshow('', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main(vidoe_path)