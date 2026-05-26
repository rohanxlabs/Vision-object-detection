import cv2
cap = cv2.VideoCapture(0)
cv2.namedWindow("Test",cv2.WINDOW_NORMAL)

while True:
    ret,frame = cap.read()
    if not ret:
        break
    cv2.imshow("Test",frame)
    if cv2.waitKey(30) == ord('q'):
        break

cap.release
cv2.destroyAllWindows