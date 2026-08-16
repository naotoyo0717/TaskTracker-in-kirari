import os
from typing import Dict

import torch

from task_tracker.models.model import Model

# Set the cache directory for Hugging Face transformers
# cache_dir = "/disk1/"
cache_dir = "/home/k705456/hf_cache/"  
os.environ["TRANSFORMERS_CACHE"] = cache_dir
os.environ["HF_HOME"] = cache_dir

# Directory where model activation data will be stored
# activation_parent_dir = "/disk3/activations/"
activation_parent_dir = "/home/k705456/ml/activations/" 

# Directory where the dataset text files are stored
# text_dataset_parent_dir = (
#     "/home/saabdelnabi/TaskTracker/task_tracker/dataset_creation/dataset_sampled"
# )
text_dataset_parent_dir = (
    "/home/k705456/ml/TaskTracker/task_tracker/dataset_creation/dataset_sampled"
)

# Paths to dataset files
data = {
    "train": os.path.join(text_dataset_parent_dir, "train_subset.json"),
    "val_clean": os.path.join(text_dataset_parent_dir, "dataset_out_clean.json"),
    "val_poisoned": os.path.join(
        text_dataset_parent_dir, "dataset_out_poisoned_v1.json"
    ),
    "test_clean": os.path.join(text_dataset_parent_dir, "dataset_out_clean_v2.json"),
    "test_poisoned": os.path.join(
        text_dataset_parent_dir, "dataset_out_poisoned_v2.json"
    ),
}

# Initialize models with specific configurations
llama_3_70B = Model(
    name="meta-llama/Meta-Llama-3-70B-Instruct",
    output_dir=os.path.join(activation_parent_dir, "llama3_70b"),
    data=data,
    subset="train",
    torch_dtype=torch.bfloat16,
)

llama_3_8B = Model(
    name="meta-llama/Meta-Llama-3-8B-Instruct",
    output_dir=os.path.join(activation_parent_dir, "llama3_8b"),
    data=data,
    subset="train",
    torch_dtype=torch.float32,
)

# mistral_7B = Model(
#     name="mistralai/Mistral-7B-Instruct-v0.2",
#     output_dir=os.path.join(activation_parent_dir, "mistral_test"),
#     data=data,
#     subset="train",
#     torch_dtype=torch.float32,
# )

mistral_7B = Model(
    name="mistralai/Mistral-7B-Instruct-v0.2",
    output_dir=os.path.join(activation_parent_dir, "mistral_test"),
    data=data,
    subset="train",
    torch_dtype=torch.bfloat16,
)

phi3 = Model(
    name="microsoft/Phi-3-mini-4k-instruct",
    output_dir=os.path.join(activation_parent_dir, "phi3"),
    data=data,
    subset="train",
    torch_dtype=torch.bfloat16,
)

mixtral = Model(
    name="mistralai/Mixtral-8x7B-Instruct-v0.1",
    output_dir=os.path.join(activation_parent_dir, "mixtral"),
    data=data,
    subset="train",
    torch_dtype=torch.float16,
)

# ===== StruQ models (本命: 構造化クエリ防御の比較) =====
struq_base = "/home/k705456/ml/StruQ/mistralai"

# 条件①: 防御なし (TextTextText, フィルタなし)
struq_text = Model(
    name=os.path.join(struq_base, "Mistral-7B-v0.1_TextTextText_None_manual"),
    output_dir=os.path.join(activation_parent_dir, "struq_text"),
    data=data,
    subset="train",
    torch_dtype=torch.bfloat16,
    struq_type="TextTextText",
    apply_filter=False,
)

# 条件②: 防御あり・フィルタなし (SpclSpclSpcl, フィルタなし) — テンプレート単独の効果
struq_spcl_nofilter = Model(
    name=os.path.join(struq_base, "Mistral-7B-v0.1_SpclSpclSpcl_NaiveCompletion"),
    output_dir=os.path.join(activation_parent_dir, "struq_spcl_nofilter"),
    data=data,
    subset="train",
    torch_dtype=torch.bfloat16,
    struq_type="SpclSpclSpcl",
    apply_filter=False,
)

# 条件③: 防御あり・フィルタあり (SpclSpclSpcl, フィルタあり) — 完全な防御
struq_spcl_filter = Model(
    name=os.path.join(struq_base, "Mistral-7B-v0.1_SpclSpclSpcl_NaiveCompletion"),
    output_dir=os.path.join(activation_parent_dir, "struq_spcl_filter"),
    data=data,
    subset="train",
    torch_dtype=torch.bfloat16,
    struq_type="SpclSpclSpcl",
    apply_filter=True,
)

# Dictionary of models for easy access
models: Dict[str, Model] = {
    "llama3_70b": llama_3_70B,
    "llama3_8b": llama_3_8B,
    "mistral": mistral_7B,
    "phi3": phi3,
    "mixtral": mixtral,
    "struq_text": struq_text,
    "struq_spcl_nofilter": struq_spcl_nofilter,
    "struq_spcl_filter": struq_spcl_filter,
}
