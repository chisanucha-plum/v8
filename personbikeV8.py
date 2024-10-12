import cv2
from ultralytics import YOLO

# โหลดโมเดล YOLOv8n
model = YOLO('yolov8n.pt')

# ตั้งค่าการตรวจจับเฉพาะคน และมอเตอร์ไซค์
# Person = class 0, Motorbike = class 3
classes_to_detect = [0, 3]

# เปิดไฟล์วิดีโอ MP4
video_path = 'test.mp4'
cap = cv2.VideoCapture(video_path)

# ตั้งค่าการบันทึกวิดีโอ
output_path = 'output_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # ทำการตรวจจับวัตถุ
    results = model(frame)

    # ดึงข้อมูลผลการตรวจจับ
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls)
            conf = float(box.conf)
            if cls in classes_to_detect:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = f"{model.names[cls]} {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # บันทึกเฟรมที่มีการตรวจจับลงในไฟล์วิดีโอ
    out.write(frame)

    # แสดงผลภาพ (ถ้าต้องการ)
    cv2.imshow('YOLOv8 Detection', frame)

    # กด 'q' เพื่อออกจากลูป
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ปิดการใช้งาน
cap.release()
out.release()
cv2.destroyAllWindows()
