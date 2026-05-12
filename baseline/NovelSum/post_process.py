import json


with open('SFT_novel_10000/novelselect_10000_dense_0.5_dist_1.0_indices.json', 'r') as f:
    indices = json.load(f)

with open('../../data/SFT.json', 'r') as f:
    data = json.load(f)

selected_data = [data[i] for i in indices]

with open('../../data/SFT_novel_10000.json', 'w') as f:
    json.dump(selected_data, f, indent=2, ensure_ascii=False)