#!/bin/bash
# Llama 3条件: 活性化デルタ全件(forward) + 生成文字列100件(generate) を続けて実行
set -e
cd ~/ml/TaskTracker
mkdir -p logs

CONDS="llama_text llama_spcl_nofilter llama_spcl_filter"

echo "########## PART 1: 活性化デルタ全件生成 (forward, 3条件) ##########"
for cond in $CONDS; do
    echo "===== [$cond] 全件 activation 生成開始 $(date) ====="
    sed -i "s/^model_name: str = .*/model_name: str = \"$cond\"/" task_tracker/activations/generate.py
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        python task_tracker/activations/generate.py 2>&1 | tee "logs/${cond}_actgen.log"
    echo "===== [$cond] 全件 activation 完了 $(date) ====="
done

echo ""
echo "########## PART 2: 生成文字列ログ 100件 (generate, 3条件) ##########"
for cond in $CONDS; do
    echo "===== [$cond] 生成文字列100件 開始 $(date) ====="
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        python task_tracker/activations/gen_with_acts_tt.py --cond "$cond" --n 100 2>&1 | tee "logs/${cond}_gen100.log"
    echo "===== [$cond] 生成文字列100件 完了 $(date) ====="
done

echo ""
echo "########## 全処理完了 $(date) ##########"
