import cv2 
from cvzone.PoseModule import PoseDetector
# Use numpy to show the connection between different landmarks in the blank window
import numpy as np

detector = PoseDetector()

# Take live camera  input for pose detection
# cap = cv2.VideoCapture(0)

# Put here video of your choice
cap = cv2.VideoCapture("../video_test/videoyoutube.mp4")
# cap = cv2.VideoCapture("../video_test/test2.mp4")
# cap = cv2.VideoCapture("../video_test/test3.mp4")

while True:
    success, img = cap.read()
    if success:
        # Pose detection
        imgDetected = detector.findPose(img)
        # resize
        imgDetected = cv2.resize(imgDetected, (1200, 1200))
        LmList, bboxInfo = detector.findPosition(imgDetected,draw=True, bboxWithHands=False)
        cv2.imshow("Detect", imgDetected)

        # Blank window
        opImg = np.zeros([1000, 1000, 3])
        opImg.fill(255)

        opImg = detector.findPoseBlankWindow(img, opImg)
        LmList, bboxInfo = detector.findPosition(opImg, draw=True, bboxWithHands=False)
        cv2.imshow("Extracted Pose", opImg)

        print(LmList)
        if cv2.waitKey(1) == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()