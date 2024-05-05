import cv2
import time
import mediapipe as mp
import math
import numpy as np

cap = cv2.VideoCapture(0)

mpHand = mp.solutions.hands
hands = mpHand.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

def ucgen_olma(kose1 , kose2 , kose3):
    kose_list = [kose1, kose2, kose3]
    for i in range(len(kose_list)):
        for j in range(len(kose_list)):
            for k in range(len(kose_list)):
                if i!=j and j!=k and i!=k:
                    if np.abs(kose_list[i] - kose_list[j]) <= kose_list[k] <= kose_list[i] + kose_list[j]:
                        continue
                    else:
                        return False
                        break 
    
    return True

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #mediapipe kütüphanesi RGB renk formatını kabul ediyor bu yüzden RGB'ye çeviriyoruz.
    
    results = hands.process(imgRGB)
    
    lm_list = []
    if results.multi_hand_landmarks:
        for handLandmarks in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLandmarks, mpHand.HAND_CONNECTIONS)
            
            for id, lm in enumerate(handLandmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                
                lm_list.append([id, cx, cy])
                
                if id == 4:
                    cv2.circle(img, (cx,cy), 9, (255,0,0), cv2.FILLED)
                if id == 8:
                    cv2.circle(img, (cx,cy), 9, (0,255,0), cv2.FILLED)
                if id == 12:
                    cv2.circle(img, (cx,cy), 9, (0,0,255), cv2.FILLED)
                
    if len(lm_list) != 0:
        x1, y1 = lm_list[4][1], lm_list[4][2]
        x2, y2 = lm_list[8][1], lm_list[8][2]
        x3, y3 = lm_list[12][1], lm_list[12][2]
        
        cv2.line(img, (x1,y1), (x2,y2), (255,0,0), 3)
        cv2.line(img, (x1,y1), (x3,y3), (255,0,0), 3)
        cv2.line(img, (x2,y2), (x3,y3), (255,0,0), 3)
        
        kose_1 = int(math.sqrt((x2-x1) ** 2 + (y2-y1)**2))
        kose_2 = int(math.sqrt((x3-x2) ** 2 + (y3-y2)**2))
        kose_3 = int(math.sqrt((x3-x1) ** 2 + (y3-y1)**2))
        if ucgen_olma(kose_1, kose_2, kose_3) == True:

            angle_1 = math.degrees(math.acos((kose_1 ** 2 + kose_2 ** 2 - kose_3 ** 2) / (2 * kose_1 * kose_2)))
            angle_2 = math.degrees(math.acos((kose_2 ** 2 + kose_3 ** 2 - kose_1 ** 2) / (2 * kose_2 * kose_3)))
            angle_3 = math.degrees(math.acos((kose_1 ** 2 + kose_3 ** 2 - kose_2 ** 2) / (2 * kose_1 * kose_3)))
            
           
    
            cv2.putText(img, str(int(angle_3)), (x1, y1), cv2.FONT_HERSHEY_PLAIN, 1.3, (0,255,0), 3, cv2.LINE_AA)
            cv2.putText(img, str(int(angle_1)), (x2, y2), cv2.FONT_HERSHEY_PLAIN, 1.3, (0,255,0), 3, cv2.LINE_AA)
            cv2.putText(img, str(int(angle_2)), (x3, y3), cv2.FONT_HERSHEY_PLAIN, 1.3, (0,255,0), 3, cv2.LINE_AA)
                        
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    
    cv2.putText(img, "FPS: "+str(int(fps)), (10,75), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 5)
    
    cv2.imshow("img", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
