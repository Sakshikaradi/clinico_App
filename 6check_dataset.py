import os

# Set your dataset path
dataset_path = r"C:\Users\Sakshi\OneDrive\Desktop\clinico_App\dataset\png_data"

def check_dataset(path):
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(path, split)
        if not os.path.exists(split_path):
            print(f"❌ {split} folder not found at {split_path}")
            continue

        print(f"\nChecking {split} folder:")
        for cls in ['NORMAL', 'PNEUMONIA']:
            cls_path = os.path.join(split_path, cls)
            if not os.path.exists(cls_path):
                print(f"❌ Class folder missing: {cls_path}")
                continue

            num_files = len([f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            if num_files == 0:
                print(f"⚠️ No images found in {cls_path}")
            else:
                print(f"✅ {cls}: {num_files} images")

check_dataset(dataset_path)
