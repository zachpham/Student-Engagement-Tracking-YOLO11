import os
import glob
import shutil
from ultralytics import YOLO

def main():
    print("Đang tìm kiếm file mô hình và kết quả...")
    
    # 1. Tìm đường dẫn file best.pt mới nhất
    best_pt_files = glob.glob(r'runs\classify\runs\Student_Engagement_Project\weights\best.pt', recursive=True)
    if not best_pt_files:
        print("LỖI: Không tìm thấy file best.pt!")
        return

    best_model_path = max(best_pt_files, key=os.path.getmtime)
    
    # Thư mục chứa kết quả train (thư mục cha của thư mục chứa best.pt)
    train_run_dir = os.path.dirname(os.path.dirname(best_model_path))
    
    # 2. Tạo thư mục để lưu kết quả đánh giá (nằm ngoài task2)
    eval_out_dir = r'Evaluate_Results'
    os.makedirs(eval_out_dir, exist_ok=True)

    # 3. Load mô hình và chạy đánh giá trên tập Test
    print(f"Load AI từ: {best_model_path}")
    model = YOLO(best_model_path)
    
    print("\n--- ĐANG KIỂM TRA TRÊN TẬP TEST (10%) ---")
    metrics = model.val(data=r'Processed_Dataset', split='test') 
    accuracy = metrics.top1 * 100

    # 4. SAO CHÉP CÁC BIỂU ĐỒ QUAN TRỌNG VÀO THƯ MỤC EVALUATE
    print(f"\nĐang gom các biểu đồ vào thư mục: {eval_out_dir}")
    

    important_files = [
        'results.png',              # Biểu đồ Loss và Accuracy qua các Epoch
        'confusion_matrix.png',     # Ma trận nhầm lẫn (cực kỳ quan trọng)
        'confusion_matrix_normalized.png',
        'val_batch0_labels.jpg',    # Ảnh mẫu dự đoán thực tế
        'val_batch0_pred.jpg'
    ]

    for f_name in important_files:
        src = os.path.join(train_run_dir, f_name)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(eval_out_dir, f_name))
            print(f"  -> Đã copy: {f_name}")

    # 5. Lưu kết quả Accuracy vào một file text để sau này khỏi quên
    with open(os.path.join(eval_out_dir, 'final_accuracy.txt'), 'w', encoding='utf-8') as f:
        f.write(f"Mô hình sử dụng: {best_model_path}\n")
        f.write(f"Độ chính xác chung cuộc (Top-1 Accuracy): {accuracy:.2f}%")

    print(f"\nXong! Độ chính xác: {accuracy:.2f}%")
    print(f"Mời bạn vào thư mục 'Evaluate_Results' ở ngoài để xem các biểu đồ.")

if __name__ == '__main__':
    main()