import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from task_tracker.training.dataset import ActivationsDatasetDynamicPrimaryText
from task_tracker.training.utils.constants import (
    TEST_ACTIVATIONS_DIR_PER_MODEL,
    TEST_CLEAN_FILES_PER_MODEL,
    TEST_POISONED_FILES_PER_MODEL,
)

MODEL = 'mistral'
BATCH_SIZE = 256
TEST_ACTIVATIONS_DIR = TEST_ACTIVATIONS_DIR_PER_MODEL[MODEL]
LAYERS = 32
N_PER_CLASS = 10000
FILES_CHUNCK = 10

# 出力先
OUT_DIR = '/home/k705456/ml/TaskTracker/task_tracker/evaluation/visualizations/tsne_out'
os.makedirs(OUT_DIR, exist_ok=True)

# 5000点確保に十分なファイル数だけ読む（1ファイル約1000件 → 6ファイル）
clean_files = TEST_CLEAN_FILES_PER_MODEL[MODEL][:11]
poisoned_files = TEST_POISONED_FILES_PER_MODEL[MODEL][:11]
print(f'{len(clean_files)} clean files, {len(poisoned_files)} poisoned files (subset)')


def compute_activations_residuals(evaluate_files):
    emb_diffs = [[] for _ in range(LAYERS)]
    for i in range(0, len(evaluate_files), FILES_CHUNCK):
        files = evaluate_files[i:i + FILES_CHUNCK]
        dataset = ActivationsDatasetDynamicPrimaryText(files, LAYERS, TEST_ACTIVATIONS_DIR)
        data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
        for data in data_loader:
            primary, primary_with_text = data
            for layer in range(primary.size(1)):
                # bf16対策: .float() を挟んでからnumpy化
                d = (primary_with_text[:, layer, :] - primary[:, layer, :]).float()
                emb_diffs[layer].append(d.numpy())
    stacked = [np.vstack(emb_diffs[layer]) for layer in range(LAYERS)]
    return stacked


print("Computing poisoned deltas...")
diff_poisoned = compute_activations_residuals(poisoned_files)
print("Computing clean deltas...")
diff_clean = compute_activations_residuals(clean_files)

# 正確にN_PER_CLASS点へ切り出し
diff_clean = [d[:N_PER_CLASS] for d in diff_clean]
diff_poisoned = [d[:N_PER_CLASS] for d in diff_poisoned]
print(f"clean layer0 shape: {diff_clean[0].shape}, poisoned layer0 shape: {diff_poisoned[0].shape}")

# t-SNE（sklearn）
from sklearn.manifold import TSNE

all_tsne = []
for layer in range(LAYERS):
    print(f"[t-SNE] layer {layer} ...", flush=True)
    combined = np.vstack((diff_clean[layer], diff_poisoned[layer]))
    tsne = TSNE(n_components=2, random_state=42)
    reduced = tsne.fit_transform(combined)
    all_tsne.append(reduced)
    # 各層ごとに逐次保存（途中で落ちても既済み層は残る）
    np.save(os.path.join(OUT_DIR, f'tsne_layer_{layer:02d}.npy'), reduced)
    print(f"   saved layer {layer}", flush=True)

# 全層まとめて保存
np.save(os.path.join(OUT_DIR, 'tsne_all_layers.npy'), np.array(all_tsne))
# ラベル（前半clean, 後半poisoned）も保存
labels = np.array(['Clean'] * N_PER_CLASS + ['Poisoned'] * N_PER_CLASS)
np.save(os.path.join(OUT_DIR, 'labels.npy'), labels)
print("All t-SNE coordinates saved to", OUT_DIR)

# 描画（32層を16x2で）
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axs = plt.subplots(16, 2, figsize=(12, 80))
fig.tight_layout(h_pad=2)
for layer in range(LAYERS):
    ax = axs[layer // 2][layer % 2]
    emb = all_tsne[layer]
    for label, color in zip(['Clean', 'Poisoned'], ['#4CB2F9', '#FA7988']):
        cond = labels == label
        ax.scatter(emb[cond, 0], emb[cond, 1], c=color, label=label, alpha=0.3, s=5)
    ax.set_title(f'layer {layer}', fontsize=10)
    ax.legend(fontsize=8)

out_png = os.path.join(OUT_DIR, 'tsne_all32_layers.png')
plt.savefig(out_png, dpi=100, bbox_inches='tight')
print("Figure saved to", out_png)
