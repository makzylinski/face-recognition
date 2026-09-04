
import cv2
import numpy as np

cap = cv2.VideoCapture(0)
rectangleImg = np.zeros((512, 512, 3), dtype=np.uint8)
topLeft = (100, 100)
topRight = (400, 400)
colorRed = (0, 0, 255)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    cv2.rectangle(frame, topLeft, topRight, colorRed)
    cv2.imshow("Kamera", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
