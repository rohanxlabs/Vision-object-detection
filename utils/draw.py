import cv2 

def draw_boxes(frame, detections):
    for det in detections:
        x1,y1,x2,y2 = map(int, det["xyxy"])
        label = f'{det["class_name"]}{det["confidence"]:.2f}'

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255,0),2)
        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,0.6,
            (0, 255, 0),
            2

        )
    return frame