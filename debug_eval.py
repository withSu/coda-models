import pickle
import numpy as np

# Load evaluation results
result_path = "/home/a/ros2_ws/coda-models/output/cfgs/da_models/second_coda32_oracle_3class/defaultLR0.010000OPTadam_onecycle/eval/eval_all_default/epoch_23/val/result.pkl"

try:
    with open(result_path, "rb") as f:
        det_annos = pickle.load(f)
        
    print(f"Number of detection annotations: {len(det_annos)}")
    
    # Check first few detection results
    for i in range(min(3, len(det_annos))):
        print(f"\n--- Detection {i} ---")
        print(f"Number of detections: {len(det_annos[i]['name'])}")
        if len(det_annos[i]['name']) > 0:
            print(f"First few class names: {det_annos[i]['name'][:5]}")
            print(f"First few scores: {det_annos[i]['score'][:5]}")
            
            # Count detections per class
            unique_names, counts = np.unique(det_annos[i]['name'], return_counts=True)
            for name, count in zip(unique_names, counts):
                print(f"  {name}: {count} detections")
                
except FileNotFoundError:
    print(f"Result file not found at {result_path}")
    print("Looking for available result files...")
    
    import os
    base_dir = "/home/a/ros2_ws/coda-models/output/cfgs/da_models/second_coda32_oracle_3class/defaultLR0.010000OPTadam_onecycle/"
    
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file == "result.pkl":
                print(f"Found: {os.path.join(root, file)}")

# Load ground truth
info_path = "/home/a/ros2_ws/coda-models/data/coda32_3class/coda_infos_val.pkl"
with open(info_path, "rb") as f:
    infos = pickle.load(f)
    
print(f"\n\nNumber of GT samples: {len(infos)}")

# Check GT annotations
for i in range(min(3, len(infos))):
    if 'annos' in infos[i]:
        annos = infos[i]['annos']
        print(f"\n--- GT Sample {i} ---")
        print(f"Number of GT objects: {len(annos['name'])}")
        if len(annos['name']) > 0:
            print(f"First few GT names: {annos['name'][:5]}")
            
            # Count GT per class
            unique_names, counts = np.unique(annos['name'], return_counts=True)
            for name, count in zip(unique_names, counts):
                print(f"  {name}: {count} objects")