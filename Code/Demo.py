import cv2
import os
import glob
from ultralytics import YOLO

def main():
    print("Dang tim kiem file mo hinh (best.pt)...")
    
    # Tim de quy tat ca cac file best.pt nam trong thu muc runs (du cho no bi long vao may lop thu muc)
    best_pt_files = glob.glob(r'runs/**/best.pt', recursive=True)
    
    # Du phong neu ban co lo de folder runs o ngoai folder me
    if not best_pt_files:
        best_pt_files = glob.glob(r'../runs/**/best.pt', recursive=True)
        
    if not best_pt_files:
        print("LOI: Khong tim thay mo hinh nao! Ban da chay file Train.py chua?")
        return
        
    # Lay file best.pt moi duoc tao gan day nhat
    best_model_path = max(best_pt_files, key=os.path.getmtime)

    print(f"Da tim thay! Dang load AI tu: {best_model_path}")
    model = YOLO(best_model_path)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("LOI: Khong mo duoc webcam!")
        return

    print("Bat webcam thanh cong. Nhan phim 'q' tren cua so camera de tat.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        results = model.predict(frame, verbose=False) 
        
        top1_idx = results[0].probs.top1
        class_name = results[0].names[top1_idx]
        conf = results[0].probs.top1conf.item()

        # Doi mau khung vien: Xanh la neu Focused, Do neu cac trang thai khac
        color = (0, 255, 0) if class_name in ['Focused'] else (0, 0, 255)
        label = f"{class_name}: {conf:.2%}"
        
        # Ve giao dien
        cv2.rectangle(frame, (10, 10), (frame.shape[1]-10, frame.shape[0]-10), color, 3)
        cv2.rectangle(frame, (15, 15), (350, 60), (0, 0, 0), -1) 
        cv2.putText(frame, label, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        cv2.imshow("Student Engagement Tracker (Nhan 'q' de thoat)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()