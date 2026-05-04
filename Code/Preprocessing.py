import os
import shutil
import random

def main():
    base_dir = 'Student_Dataset' 
    output_dir = 'Processed_Dataset'

    mapping = {
        'Engaged/Focused': 'Focused',
        'Engaged/Confused': 'Confused',
        'Engaged/Frustrated': 'Frustrated',
        'Not Engaged/Bored': 'Bored',
        'Not Engaged/Drowsy': 'Drowsy',
        'Not Engaged/Looking Away': 'Looking_Away'
    }

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    for split in ['train', 'val', 'test']:
        for target_cls in mapping.values():
            os.makedirs(os.path.join(output_dir, split, target_cls), exist_ok=True)

    print("Đang xử lý và chia dữ liệu (70-20-10)...")
    for sub_path, target_cls in mapping.items():
        full_src_path = os.path.join(base_dir, sub_path)
        if not os.path.exists(full_src_path): continue
            
        files = [f for f in os.listdir(full_src_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(files)
        
        train_idx = int(len(files) * 0.7)
        val_idx = int(len(files) * 0.9) 
        
        for i, f in enumerate(files):
            if i < train_idx: split = 'train'
            elif i < val_idx: split = 'val'
            else: split = 'test'
                
            shutil.copy(os.path.join(full_src_path, f), os.path.join(output_dir, split, target_cls, f))
        print(f"Xong class: {target_cls}")

    print("\nPreprocessing hoàn tất! Dữ liệu đã lưu ở 'Processed_Dataset'")

if __name__ == '__main__':
    main()