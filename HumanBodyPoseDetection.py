# Sử dụng render bằng cpu >> gpu 
# (theo như nhận xét của tác giả trong video, giảm độ trễ, snooth hơn)
import cv2
import mediapipe as mp
# Use numpy to show the connection between different landmarks in the blank window
import numpy as np

from cvzone.PoseModule import PoseDetector

detector = PoseDetector()

# initialize mediapipe pose solution
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# ===============================================================================================================================================
# Take video input for pose detection

# Put here video of your choice
# cap = cv2.VideoCapture("../production_id_4057407 (2160p).mp4")
cap = cv2.VideoCapture("../video_test/IMG_1448.MOV")

# Take live camera  input for pose detection
# cap = cv2.VideoCapture(0)
# ===============================================================================================================================================

# read each frame/image from capture object
while True:
    # Read video input <==> function read()
    ret, img = cap.read()
    # Resize image/frame 
   #  img = cv2.resize(img, (600, 400))
    img = cv2.resize(img, (1200, 1200))

    # Do Pose detection
    results = pose.process(img)
    # Draw the detected pose on original video/ live stream
    # Function draw_landmarks dùng để vẽ các pose sau khi sử dụng lib cv2 để đọc file image/video
    # Sử dụng hàm POSE_CONNECTIONS có sẵn trong thư viện để thực hiện nối các landmarks
    # lại với nhau (32 landmarks)
    mp_draw.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                        # Chỉnh màu của các điểm pose (màu RGB, thickness, cirle radius)
                           mp_draw.DrawingSpec((0, 0, 255), 2, 2),
                        # Chỉnh màu của các lines connect pose (màu RGB, thickness, cirle radius)
                           mp_draw.DrawingSpec((0, 255, 0), 1, 2)
                           )
    
    # Display pose on original video/live stream
    cv2.imshow("Pose Estimation", img)
    # =============================================================================================================================================

    # Extract and draw pose on the blank window
    # Create the blank window
    height, width, channel = img.shape   # get shape of original frame

    # Open blank window with pose only as same width anh height
    opImg = np.zeros([height, width, channel])  # create blank image with original frame size

    opImg.fill(255)  # set white background. put 0 if you want to make it black

    # Draw extracted pose (only pose) on black window
    mp_draw.draw_landmarks(opImg, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                           mp_draw.DrawingSpec((0, 0, 255), 2, 2),
                           mp_draw.DrawingSpec((255, 0, 255), 1, 2)
                           )
    # display extracted pose on blank images
    cv2.imshow("Extracted Pose", opImg)

    # print all landmarks
    print(results.pose_landmarks)

    cv2.waitKey(1)
