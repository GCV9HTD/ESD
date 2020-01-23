# python detect_blinks.py --shape shape_predictor_68_face_landmarks.dat
# python detect_blinks.py --shape shape_predictor_68_face_landmarks.dat --video blink_detection_demo.mp4

from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
# from time import time, ctime


def calc_ear(eye):
    # Compute distance between two vertical landmarks
    X_V = dist.euclidean(eye[1], eye[5])
    Y_V = dist.euclidean(eye[2], eye[4])

    # Compute the distance between two horizontal eye landmarks
    Z = dist.euclidean(eye[0], eye[3])

    # Eye Aspect Ratio
    ear = (X_V + Y_V) / (2.0 * Z)

    return ear


ap = argparse.ArgumentParser()
ap.add_argument("-s", "--shape", required=True,
                help="path to facial landmark predictor")

# ap.add_argument("-v", "--video", type=str, default="",
#   help="./blink_detection_demo.mp4")
args = vars(ap.parse_args())

EAR_thresh = 0.3  # EAR threshold
EAR_CONSEC_FRAMES = 3  # No. of consecutive frames
COUNT = 0
TOT = 0

print("loading...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape"])

(leftStart, leftEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]  # left eye landmarks
(rightStart, rightEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]  # right eye landmarks

# print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
# time.sleep(1.0)
t = time.time()


# vs = FileVideoStream(args["video"]).start()
# fileStream = True

while True:
    #   t = ctime(time()).split(" ")[3]
    #   s = t.split(":")[2]
    # t = time.time()
    #   print(ctime(time()).split(" ")[3])
    frame = vs.read()

    # detecting the face
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # convert frame to gray scale
    rects = detector(gray, 0)  # detect faces
    # print(rects)
    # -- rectangles[[(300, 120) (424, 245)], [(93, 141) (179, 227)]]

    for rect in rects:
        sx = rect.left()
        # sy = rect.top()
        w = rect.width()
        # h  = rect.height()
        # remove the unwanted faces
        if (sx < 140 or sx + w > 370):
            break
        # draw ROI face (only for development)
        # cv2.rectangle(frame,(sx,sy) , (sx+w,sy+h), (0, 255, 0), 1  )
        # print(rect.left() , rect.right() )
        # cv2.putText(frame, "#start: {}".format(sx), (sx, sy ),
        #   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        # cv2.putText(frame, "{}".format(sx+w), (sx+w, sy ),
        #   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # get facial landmarks
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        # print(shape)

        # Compute EAR values
        lEye = shape[leftStart:leftEnd]
        rEye = shape[rightStart:rightEnd]
        leftEAR = calc_ear(lEye)
        rightEAR = calc_ear(rEye)
        ear = (leftEAR + rightEAR) / 2.0

        # Drow the eay on the video frame (only for development)
        leftEyeHull = cv2.convexHull(lEye)  # convex hull for left eye
        rightEyeHull = cv2.convexHull(rEye)  # convex hull for right eye
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        if ear < EAR_thresh:
            COUNT += 1
            if COUNT > 50:
                print("sleep")
        else:
            if COUNT >= EAR_CONSEC_FRAMES:
                TOT += 1
                if (time.time() - t) > 20:
                    print(time.time() - t)
                    if TOT > 15:
                        print("Warn")
                        TOT = 0

                    else:
                        t = time.time()
            COUNT = 0

        # Print text on the video frame (only for development)
        cv2.putText(frame, "#Count: {}".format(COUNT), (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "#Blinks: {}".format(TOT), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):  # press q to exit
        break

cv2.destroyAllWindows()
vs.stop()
