from ultralytics import YOLO

def main():
    print("Bắt đầu khởi tạo mô hình YOLO11-cls...")
    model = YOLO('yolo11n-cls.pt') 

    print("Đang tiến hành huấn luyện trên GPU...")
    results = model.train(
        data='Processed_Dataset',
        project='runs',
        epochs=50,         
        imgsz=224,         
        batch=32,          
        name='Student_Engagement_Project',
        device=0           
    )
    print("Quá trình huấn luyện đã xong! Kết quả lưu trong thư mục 'runs/classify/'")

if __name__ == '__main__':
    main()