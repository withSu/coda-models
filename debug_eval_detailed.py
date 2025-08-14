import pickle
import numpy as np
import os

# Find the most recent result.pkl file
base_dir = "/home/a/ros2_ws/coda-models/output/cfgs/da_models/second_coda32_oracle_3class/defaultLR0.010000OPTadam_onecycle/"
result_files = []

for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file == "result.pkl":
            full_path = os.path.join(root, file)
            result_files.append((full_path, os.path.getmtime(full_path)))

if result_files:
    result_files.sort(key=lambda x: x[1], reverse=True)
    most_recent = result_files[0][0]
    print(f"Loading most recent result file: {most_recent}")
    
    with open(most_recent, "rb") as f:
        det_annos = pickle.load(f)
    
    print(f"\n=== DETECTION RESULTS ===")
    print(f"Total samples: {len(det_annos)}")
    
    # Check first few samples in detail
    for i in range(min(5, len(det_annos))):
        print(f"\n--- Sample {i} ---")
        if 'name' in det_annos[i]:
            names = det_annos[i]['name']
            print(f"Number of detections: {len(names)}")
            if len(names) > 0:
                unique, counts = np.unique(names, return_counts=True)
                for n, c in zip(unique, counts):
                    print(f"  '{n}': {c} detections")
                # Show first few raw detection names
                print(f"  First 5 raw names: {names[:5] if len(names) > 0 else []}")
                print(f"  First 5 scores: {det_annos[i]['score'][:5] if len(det_annos[i]['score']) > 0 else []}")
        else:
            print("  No 'name' field in detection")
            print(f"  Available keys: {det_annos[i].keys()}")
else:
    print("No result.pkl files found")

# Check ground truth data
print("\n\n=== GROUND TRUTH DATA ===")

# Find the correct info file path
info_paths = [
    "/home/a/ros2_ws/coda-models/data/coda32_allclass_full/coda_infos_val.pkl",
    "/home/a/ros2_ws/coda-models/data/coda32_3class/coda_infos_val.pkl",
]

for info_path in info_paths:
    if os.path.exists(info_path):
        print(f"Loading GT from: {info_path}")
        with open(info_path, "rb") as f:
            infos = pickle.load(f)
        
        print(f"Total GT samples: {len(infos)}")
        
        # Check first few GT samples
        for i in range(min(5, len(infos))):
            if 'annos' in infos[i]:
                annos = infos[i]['annos']
                print(f"\n--- GT Sample {i} ---")
                if 'name' in annos:
                    names = annos['name']
                    print(f"Number of GT objects: {len(names)}")
                    if len(names) > 0:
                        unique, counts = np.unique(names, return_counts=True)
                        for n, c in zip(unique, counts):
                            print(f"  '{n}': {c} objects")
                        # Show first few raw GT names
                        print(f"  First 5 raw names: {names[:5] if len(names) > 0 else []}")
                else:
                    print("  No 'name' field in GT")
            else:
                print(f"  No 'annos' field in GT sample")
        break
else:
    print("No GT info file found")

# Check if the class names match exactly
print("\n\n=== CLASS NAME COMPARISON ===")
if result_files and infos:
    # Get all unique class names from detections
    det_classes = set()
    for det in det_annos[:100]:  # Check first 100 samples
        if 'name' in det and len(det['name']) > 0:
            det_classes.update(det['name'])
    
    # Get all unique class names from GT
    gt_classes = set()
    for info in infos[:100]:  # Check first 100 samples
        if 'annos' in info and 'name' in info['annos']:
            gt_classes.update(info['annos']['name'])
    
    print(f"Unique detection classes: {sorted(det_classes)}")
    print(f"Unique GT classes: {sorted(gt_classes)}")
    print(f"Common classes: {sorted(det_classes & gt_classes)}")
    print(f"Detection-only classes: {sorted(det_classes - gt_classes)}")
    print(f"GT-only classes: {sorted(gt_classes - det_classes)}")