import pickle
import numpy as np
import os

# Load the most recent result file
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
    print(f"Loading: {most_recent}")
    
    with open(most_recent, "rb") as f:
        det_annos = pickle.load(f)

# Load GT data
info_path = "/home/a/ros2_ws/coda-models/data/coda32_allclass_full/coda_infos_val.pkl"
with open(info_path, "rb") as f:
    infos = pickle.load(f)

print("\n=== CHECKING DATA FORMAT ===")

# Check detection format
print("\nDetection format (first sample):")
if len(det_annos) > 0:
    sample = det_annos[0]
    print(f"  Keys: {sample.keys()}")
    if 'bbox' in sample:
        print(f"  bbox shape: {sample['bbox'].shape}")
        print(f"  bbox dtype: {sample['bbox'].dtype}")
        if len(sample['bbox']) > 0:
            print(f"  First bbox: {sample['bbox'][0]}")
    if 'location' in sample:
        print(f"  location shape: {sample['location'].shape}")
        if len(sample['location']) > 0:
            print(f"  First location: {sample['location'][0]}")
    if 'dimensions' in sample:
        print(f"  dimensions shape: {sample['dimensions'].shape}")
        if len(sample['dimensions']) > 0:
            print(f"  First dimensions: {sample['dimensions'][0]}")
    if 'rotation_y' in sample:
        print(f"  rotation_y shape: {sample['rotation_y'].shape}")
        if len(sample['rotation_y']) > 0:
            print(f"  First rotation_y: {sample['rotation_y'][0]}")
    if 'truncated' in sample:
        print(f"  truncated shape: {sample['truncated'].shape}")
    if 'occluded' in sample:
        print(f"  occluded shape: {sample['occluded'].shape}")
    if 'alpha' in sample:
        print(f"  alpha shape: {sample['alpha'].shape}")
        if len(sample['alpha']) > 0:
            print(f"  First alpha: {sample['alpha'][0]}")

# Check GT format
print("\nGT format (first sample with annos):")
for info in infos[:5]:
    if 'annos' in info:
        annos = info['annos']
        print(f"  Keys: {annos.keys()}")
        if 'bbox' in annos:
            print(f"  bbox shape: {annos['bbox'].shape}")
            print(f"  bbox dtype: {annos['bbox'].dtype}")
            if len(annos['bbox']) > 0:
                print(f"  First bbox: {annos['bbox'][0]}")
        if 'location' in annos:
            print(f"  location shape: {annos['location'].shape}")
            if len(annos['location']) > 0:
                print(f"  First location: {annos['location'][0]}")
        if 'dimensions' in annos:
            print(f"  dimensions shape: {annos['dimensions'].shape}")
            if len(annos['dimensions']) > 0:
                print(f"  First dimensions: {annos['dimensions'][0]}")
        if 'rotation_y' in annos:
            print(f"  rotation_y shape: {annos['rotation_y'].shape}")
            if len(annos['rotation_y']) > 0:
                print(f"  First rotation_y: {annos['rotation_y'][0]}")
        if 'truncated' in annos:
            print(f"  truncated shape: {annos['truncated'].shape}")
        if 'occluded' in annos:
            print(f"  occluded shape: {annos['occluded'].shape}")
        break

print("\n=== CHECKING BBOX VALUES ===")

# Check if bbox values are reasonable (2D image coordinates)
det_bbox_stats = []
gt_bbox_stats = []

for det in det_annos[:100]:
    if 'bbox' in det and len(det['bbox']) > 0:
        det_bbox_stats.extend(det['bbox'].flatten())

for info in infos[:100]:
    if 'annos' in info and 'bbox' in info['annos'] and len(info['annos']['bbox']) > 0:
        gt_bbox_stats.extend(info['annos']['bbox'].flatten())

if det_bbox_stats:
    det_bbox_stats = np.array(det_bbox_stats)
    print(f"\nDetection bbox statistics:")
    print(f"  Min: {det_bbox_stats.min():.2f}")
    print(f"  Max: {det_bbox_stats.max():.2f}")
    print(f"  Mean: {det_bbox_stats.mean():.2f}")

if gt_bbox_stats:
    gt_bbox_stats = np.array(gt_bbox_stats)
    print(f"\nGT bbox statistics:")
    print(f"  Min: {gt_bbox_stats.min():.2f}")
    print(f"  Max: {gt_bbox_stats.max():.2f}")
    print(f"  Mean: {gt_bbox_stats.mean():.2f}")

# Check for specific Car samples
print("\n=== CHECKING CAR SAMPLES ===")
car_det_count = 0
car_gt_count = 0

for i, det in enumerate(det_annos[:100]):
    if 'name' in det:
        car_indices = [j for j, name in enumerate(det['name']) if name == 'Car']
        if car_indices:
            car_det_count += len(car_indices)
            if i < 3:  # Show first 3 samples with Car
                print(f"\nDet sample {i} has {len(car_indices)} Car detections")
                for idx in car_indices[:2]:  # Show first 2 cars
                    print(f"  Car {idx}:")
                    print(f"    bbox: {det['bbox'][idx]}")
                    print(f"    score: {det['score'][idx]}")
                    if 'location' in det:
                        print(f"    location: {det['location'][idx]}")

for i, info in enumerate(infos[:100]):
    if 'annos' in info and 'name' in info['annos']:
        car_indices = [j for j, name in enumerate(info['annos']['name']) if name == 'Car']
        if car_indices:
            car_gt_count += len(car_indices)
            if i < 3:  # Show first 3 samples with Car
                print(f"\nGT sample {i} has {len(car_indices)} Car objects")
                for idx in car_indices[:2]:  # Show first 2 cars
                    print(f"  Car {idx}:")
                    print(f"    bbox: {info['annos']['bbox'][idx]}")
                    if 'location' in info['annos']:
                        print(f"    location: {info['annos']['location'][idx]}")

print(f"\nTotal Car detections in first 100 samples: {car_det_count}")
print(f"Total Car GT objects in first 100 samples: {car_gt_count}")