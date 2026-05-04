import cv2
import os
import glob
from ultralytics import YOLO

def main():
    # 1. LOAD MO HINH 1: NHAN DIEN NGUOI (Detector)
    print("Dang load Mo hinh 1 (Phat hien nguoi)...")
    detector = YOLO('yolo11n.pt') 

    # 2. LOAD MO HINH 2: PHAN LOAI CAM XUC (Classifier)
    print("Dang tim kiem file mo hinh 2 (best.pt)...")
    
    # Tim de quy giong het file Evaluate va Demo
    best_pt_files = glob.glob(r'runs/**/best.pt', recursive=True)
    if not best_pt_files:
        best_pt_files = glob.glob(r'../runs/**/best.pt', recursive=True)
        
    if not best_pt_files:
        print("LOI: Khong tim thay file best.pt! Ban kiem tra lai xem da train xong chua.")
        return
        
    best_model_path = max(best_pt_files, key=os.path.getmtime)
    print(f"Da tim thay! Dang load Mo hinh 2 tu: {best_model_path}")
    classifier = YOLO(best_model_path)

    # 3. MO WEBCAM (Da doi thanh so 1 de uu tien webcam laptop)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("LOI: Khong mo duoc webcam so 1! Ban co the thu doi lai thanh so 0 hoac so 2 nhe.")
        return

    print("Da san sang! (Nhan 'q' tren cua so camera de thoat)")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # --- B1: Tim tat ca sinh vien trong khung hinh ---
        det_results = detector.predict(frame, classes=[0], verbose=False)
        boxes = det_results[0].boxes

        # --- B2: Duyet qua tung nguoi de phan tich thai do ---
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Cat hinh tung nguoi ra
            student_crop = frame[y1:y2, x1:x2]
            
            if student_crop.size == 0: continue
                
            # Dua vao Mo hinh 2
            cls_results = classifier.predict(student_crop, verbose=False)
            
            top1_idx = cls_results[0].probs.top1
            state = cls_results[0].names[top1_idx]
            conf = cls_results[0].probs.top1conf.item()
            
            # --- B3: Ve khung len man hinh ---
            color = (0, 255, 0) if state in ['Focused'] else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            label = f"{state} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1-25), (x1 + len(label)*15, y1), color, -1)
            cv2.putText(frame, label, (x1+5, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        cv2.imshow("Smart Classroom Real-time Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()