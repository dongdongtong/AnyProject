from utils import read_json
import numpy as np

json_path = "/dingshaodong/projects/hematoma_meta/data/preprocessed/resampled/hematoma_fingerprint_grouped.json"
data = read_json(json_path)

train_data = data["training"] + data["validation"]

w_size = []
h_size = []
d_size = []

for data_dict in train_data:
    baseline_shape = data_dict['baseline_resample_shape']
    followup_shape = data_dict['24h_resample_shape']
    
    w_size.append(baseline_shape[0])
    h_size.append(baseline_shape[1])
    d_size.append(baseline_shape[2])
    
    w_size.append(followup_shape[0])
    h_size.append(followup_shape[1])
    d_size.append(followup_shape[2])
    
# print the mean and std of the w, h, d size
print(f"w_size: {np.mean(w_size)}, {np.std(w_size)}")
print(f"h_size: {np.mean(h_size)}, {np.std(h_size)}")
print(f"d_size: {np.mean(d_size)}, {np.std(d_size)}")

# print quantile
print(f"w_size: 25% {np.quantile(w_size, 0.25)}, 50% {np.quantile(w_size, 0.5)}, 75% {np.quantile(w_size, 0.75)}, 95% {np.quantile(w_size, 0.95)}, 99% {np.quantile(w_size, 0.99)}")
print(f"h_size: 25% {np.quantile(h_size, 0.25)}, 50% {np.quantile(h_size, 0.5)}, 75% {np.quantile(h_size, 0.75)}, 95% {np.quantile(h_size, 0.95)}, 99% {np.quantile(h_size, 0.99)}")
print(f"d_size: 25% {np.quantile(d_size, 0.25)}, 50% {np.quantile(d_size, 0.5)}, 75% {np.quantile(d_size, 0.75)}, 95% {np.quantile(d_size, 0.95)}, 99% {np.quantile(d_size, 0.99)}")

# print max and min
print(f"w_size: {np.max(w_size)}, {np.min(w_size)}")
print(f"h_size: {np.max(h_size)}, {np.min(h_size)}")
print(f"d_size: {np.max(d_size)}, {np.min(d_size)}")