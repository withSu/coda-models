import pickle
import numpy as np

# Load GT data
info_path = "/home/a/ros2_ws/coda-models/data/coda32_allclass_full/coda_infos_val.pkl"
with open(info_path, "rb") as f:
    infos = pickle.load(f)

# Collect all GT class names
all_gt_classes = []
for info in infos:
    if 'annos' in info and 'name' in info['annos']:
        all_gt_classes.extend(info['annos']['name'])

# Count occurrences of each class
unique_classes, counts = np.unique(all_gt_classes, return_counts=True)

print("GT class distribution in validation set:")
print("-" * 50)
for cls, count in sorted(zip(unique_classes, counts), key=lambda x: x[1], reverse=True):
    print(f"{cls:30s}: {count:6d}")

print(f"\nTotal objects: {len(all_gt_classes)}")
print(f"Total unique classes: {len(unique_classes)}")

# Check for Car-related classes
print("\n" + "="*50)
print("Car-related classes in GT:")
car_related = [cls for cls in unique_classes if 'Car' in cls or 'car' in cls or 'Vehicle' in cls or 'vehicle' in cls or 'Truck' in cls]
if car_related:
    for cls in car_related:
        count = counts[np.where(unique_classes == cls)[0][0]]
        print(f"  {cls}: {count}")
else:
    print("  No Car/Vehicle/Truck classes found in GT!")

# Check if we're using the wrong dataset
print("\n" + "="*50)
print("Checking dataset configuration:")
print(f"GT file path: {info_path}")
print(f"Expected classes for 3-class model: ['Car', 'Pedestrian', 'Cyclist']")
print(f"Actual classes in GT that match: {[c for c in unique_classes if c in ['Car', 'Pedestrian', 'Cyclist']]}")