import json
import random
import pandas as pd
import os

# 1. Cấu hình đường dẫn
project = "NVD"
input_path = f'/kaggle/input/graphfvd-nvd-dataset/NVD/NVD_PrVCs.json'
output_dir = f'/kaggle/working/GraphFVD/dataset/{project}'

# 2. Tạo thư mục đầu ra
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 3. Đọc dữ liệu
print(f"Reading data from: {input_path}")
try:
    # Thử đọc theo kiểu dataframe
    js_all = pd.read_json(input_path)
    js_all = js_all.to_dict('records')
except Exception as e:
    print(f"Standard read failed, trying line-by-line: {e}")
    # Nếu file JSON không phải dạng array (JSONL), dùng cách này:
    with open(input_path, 'r') as f:
        js_all = [json.loads(line) for line in f]

# 4. Chia dữ liệu Train/Valid/Test
total_num = len(js_all)
train_num = int(total_num * 0.8)
valid_num = int(total_num * 0.9)
total_idx = [i for i in range(total_num)]

random.seed(42)
random.shuffle(total_idx)

train_index = set(total_idx[:train_num])
valid_index = set(total_idx[train_num:valid_num])
test_index = set(total_idx[valid_num:])

# 5. Hàm lưu file
def save_jsonl(indices, filename):
    file_path = os.path.join(output_dir, filename)
    with open(file_path, 'w') as f:
        for idx in indices:
            js = js_all[idx]
            js['idx'] = idx
            f.write(json.dumps(js) + '\n')
    print(f"Saved to {file_path}")

save_jsonl(train_index, 'my_train.jsonl')
save_jsonl(valid_index, 'my_valid.jsonl')
save_jsonl(test_index, 'my_test.jsonl')

print("Preprocessing completed!")
