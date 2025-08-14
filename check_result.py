import pickle

# result.pkl 경로
pkl_path = "/home/a/ros2_ws/coda-models/output/cfgs/da_models/second_coda32_oracle_3class/defaultLR0.010000OPTadam_onecycle/eval/eval_all_default_ORIGINAL/epoch_27/val/result.pkl"

with open(pkl_path, "rb") as f:
    data = pickle.load(f)

# 타입과 키 확인
print(type(data))
if isinstance(data, dict):
    print(data.keys())
elif isinstance(data, list):
    print(f"총 {len(data)}개의 항목")
    print("첫 번째 항목:", data[0])
else:
    print(data)
