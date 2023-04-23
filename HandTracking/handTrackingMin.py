import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    #print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                #print(id,lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h) # multiplying x co-ordinate with width and y co-ordinate with height
                print(id, cx, cy)
                if id==0 :
                    # if landmark = 0, then we are detecting it, there are total 20 landmarks on your hand, we are targetting 0th one
                    # 25 is basically the radius of the circle
                    cv2.circle(img, (cx,cy), 25, (255,0,255), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)),(10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
    # textColor = purple (255,0,255)
    (255,0,255),3)

    cv2.imshow('Image',img)
    cv2.waitKey(1)

