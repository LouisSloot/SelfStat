import os
import json
import cv2
from pathlib import Path
import shutil

# prepares the MultiSubjects dataset for use with MMAction2

def prepare_multisubjects_dataset(dataset_root):

    dataset_root = Path(dataset_root)
    print(f"Preparing MultiSubjects dataset from: {dataset_root}")
    
    # output directory
    output_dir = Path("data/multisubjects")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    organize_video_files(dataset_root, output_dir)
    
    train_list, val_list, test_list = process_annotation_files(dataset_root, output_dir)
    
    return train_list, val_list, test_list

def organize_video_files(dataset_root, output_dir):
    
    videos_output = output_dir / "videos"
    videos_output.mkdir(exist_ok=True)
    
    # process by split
    for split in ['train', 'val', 'test']:
        source_dir = dataset_root / f"videos_{split}"
        
        if not source_dir.exists():
            print(f"Warning: {source_dir} not found, skipping...")
            continue
            
        print(f"Processing {split} videos from {source_dir}")
        video_files = list(source_dir.glob("*.mp4"))
        print(f"Found {len(video_files)} videos in {split}")
        
        
        for video_file in video_files:
            dest_file = videos_output / video_file.name
            
            # symlink to save disk space
            try:
                if not dest_file.exists():
                    os.symlink(video_file.absolute(), dest_file.absolute())
            except OSError:
                #  copy if symlinks not supported
                if not dest_file.exists():
                    shutil.copy2(video_file, dest_file)

def process_annotation_files(dataset_root, output_dir):
    
    train_annotations = read_annotation_file(dataset_root / "labels_train.txt")
    val_annotations = read_annotation_file(dataset_root / "labels_val.txt") 
    test_annotations = read_annotation_file(dataset_root / "labels_test.txt")
    
    print(f"Loaded annotations:")
    print(f"  Train: {len(train_annotations)} videos")
    print(f"  Val: {len(val_annotations)} videos")
    print(f"  Test: {len(test_annotations)} videos")
    
    videos_dir = output_dir / "videos"
    
    train_list = verify_and_format_annotations(train_annotations, videos_dir, "train")
    val_list = verify_and_format_annotations(val_annotations, videos_dir, "val")
    test_list = verify_and_format_annotations(test_annotations, videos_dir, "test")
    
    # save in MMAction2 format
    save_annotation_file(train_list, output_dir / "train_list.txt")
    save_annotation_file(val_list, output_dir / "val_list.txt")
    save_annotation_file(test_list, output_dir / "test_list.txt")
    
    return train_list, val_list, test_list

def read_annotation_file(annotation_path):

    annotations = []
    
    if not annotation_path.exists():
        print(f"Warning: {annotation_path} not found")
        return annotations
        
    with open(annotation_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                parts = line.split()
                if len(parts) != 2:
                    print(f"Warning: Invalid format in {annotation_path} line {line_num}: {line}")
                    continue
                    
                filename, label = parts
                annotations.append({
                    'filename': filename,
                    'label': int(label)
                })
            except ValueError as e:
                print(f"Warning: Error parsing {annotation_path} line {line_num}: {line} - {e}")
                continue
    
    return annotations

def verify_and_format_annotations(annotations, videos_dir, split_name):

    verified_list = []
    missing_count = 0
    
    for ann in annotations:
        video_path = videos_dir / ann['filename']
        
        if video_path.exists():
            verified_list.append({
                'filename': f"videos/{ann['filename']}",
                'label': ann['label'],
                'total_frames': get_video_frame_count(video_path) if video_path.exists() else None
            })
        else:
            missing_count += 1
            if missing_count <= 5:
                print(f"Warning: Video not found: {video_path}")
    
    if missing_count > 5:
        print(f"Warning: {missing_count - 5} more videos missing from {split_name} split")
    
    print(f"Verified {len(verified_list)} videos for {split_name} split")
    return verified_list

def get_video_frame_count(video_path):
    
    try:
        cap = cv2.VideoCapture(str(video_path))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return frame_count
    except:
        return None

def save_annotation_file(video_list, output_path):

    with open(output_path, 'w') as f:
        for video_info in video_list:
            f.write(f"{video_info['filename']} {video_info['label']}\n")

def analyze_dataset_distribution(train_list, val_list, test_list):

    class_names = {0: 'dribbling', 1: 'layup', 2: 'shooting'}
    
    def count_classes(data_list):
        counts = {0: 0, 1: 0, 2: 0}
        for item in data_list:
            counts[item['label']] += 1
        return counts
    
    train_counts = count_classes(train_list)
    val_counts = count_classes(val_list)
    test_counts = count_classes(test_list)
    
    print("\n=== Dataset Statistics ===")
    print(f"{'Split':<10} {'Total':<8} {'Dribbling':<12} {'Layup':<8} {'Shooting':<10}")
    print("-" * 50)
    print(f"{'Train':<10} {len(train_list):<8} {train_counts[0]:<12} {train_counts[1]:<8} {train_counts[2]:<10}")
    print(f"{'Val':<10} {len(val_list):<8} {val_counts[0]:<12} {val_counts[1]:<8} {val_counts[2]:<10}")
    print(f"{'Test':<10} {len(test_list):<8} {test_counts[0]:<12} {test_counts[1]:<8} {test_counts[2]:<10}")
    
    total_videos = len(train_list) + len(val_list) + len(test_list)
    total_counts = {
        0: train_counts[0] + val_counts[0] + test_counts[0],
        1: train_counts[1] + val_counts[1] + test_counts[1], 
        2: train_counts[2] + val_counts[2] + test_counts[2]
    }
    
    print(f"{'Total':<10} {total_videos:<8} {total_counts[0]:<12} {total_counts[1]:<8} {total_counts[2]:<10}")
    
    print("\n=== Class Distribution (%) ===")
    for label, name in class_names.items():
        percentage = (total_counts[label] / total_videos) * 100
        print(f"{name}: {percentage:.1f}%")
    
    return {
        'total_videos': total_videos,
        'class_distribution': total_counts,
        'splits': {
            'train': len(train_list),
            'val': len(val_list),
            'test': len(test_list)
        }
    }

def create_class_mapping():
    """Create class index mapping file"""
    classes = {
        0: 'dribbling',
        1: 'layup', 
        2: 'shooting'
    }
    
    output_path = Path('data/multisubjects/class_mapping.json')
    with open(output_path, 'w') as f:
        json.dump(classes, f, indent=2)
    
    return classes

if __name__ == "__main__":
    
    dataset_root = Path("path/to/multisubjects") # replace with local path
    
    print("MultiSubjects Dataset Preparation")
    print("=" * 40)
    
    if not dataset_root.exists():
        print(f"Dataset not found at: {dataset_root}")
        print("Please check the path and try again.")
        exit(1)
    
    
    try:
        train_list, val_list, test_list = prepare_multisubjects_dataset(dataset_root)
        
       
        stats = analyze_dataset_distribution(train_list, val_list, test_list)
       
        create_class_mapping()
        
        print("\nDataset preparation completed successfully!")
        print(f"Output directory: data/multisubjects/")
        print("Files created:")
        print("   - videos/ (organized video files)")
        print("   - train_list.txt")
        print("   - val_list.txt") 
        print("   - test_list.txt")
        print("   - class_mapping.json")
        
        print(f"\nReady for training with {stats['total_videos']} total videos!")
        
    except Exception as e:
        print(f"Error during dataset preparation: {e}")
        import traceback
        traceback.print_exc()