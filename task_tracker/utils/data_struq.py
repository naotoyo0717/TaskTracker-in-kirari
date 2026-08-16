# -*- coding: utf-8 -*-
"""
StruQモデル用のプロンプト整形。
TaskTrackerのformat_promptsと同じ3リスト(primary/clean/poisoned)を返す。
StruQのconfig.pyからPROMPT_FORMATとFILTERED_TOKENSを直接importして使用。
"""
import sys

# StruQのconfig.pyを読み込む
STRUQ_PATH = "/home/k705456/ml/StruQ"
if STRUQ_PATH not in sys.path:
    sys.path.insert(0, STRUQ_PATH)

from config import PROMPT_FORMAT, FILTERED_TOKENS


def recursive_filter(s):
    """StruQ test.py の recursive_filter を移植。
    input部から特殊トークンと'##'を、消えなくなるまで繰り返し除去する。
    防御ありモデル(SpclSpclSpcl)のinput部にのみ適用する運用。"""
    filtered = False
    while not filtered:
        for f in FILTERED_TOKENS:
            s = s.replace(f, '')
        filtered = True
        for f in FILTERED_TOKENS:
            if f in s:
                filtered = False
                break
    return s


def format_prompts_struq(dataset_items, model_type, apply_filter):
    """
    Parameters
    ----------
    dataset_items : list[dict]
        TaskTrackerのデータセット項目。sep_prompt, primary_task_prompt,
        orig_text, final_text_paragraph を使用。
    model_type : str
        "SpclSpclSpcl"(防御あり) or "TextTextText"(防御なし)
    apply_filter : bool
        Trueならinput部にrecursive_filterを適用(防御あり運用)。

    Returns
    -------
    batch_primary, batch_primary_clean, batch_primary_poisoned : list[str]
        3文脈のプロンプト文字列リスト。インデックス対応。
    """
    assert model_type in PROMPT_FORMAT, f"unknown model_type: {model_type}"
    prompt_input = PROMPT_FORMAT[model_type]["prompt_input"]

    batch_primary = []
    batch_primary_clean = []
    batch_primary_poisoned = []

    for item in dataset_items:
        # TaskTrackerのキーをStruQのスロットにマッピング
        instruction = item["sep_prompt"] + " " + item["primary_task_prompt"]
        clean_input = item["orig_text"]
        poisoned_input = item["final_text_paragraph"]

        # 防御ありモデルはinput部をフィルタ(clean/poisoned両方)
        # instruction部にはフィルタしない(StruQの運用通り)
        if apply_filter:
            clean_input = recursive_filter(clean_input)
            poisoned_input = recursive_filter(poisoned_input)

        # 3文脈をStruQテンプレートで組む
        # x_pri は input を空にして統制(clean/poisonedとの差はデータ本文のみ)
        x_pri = prompt_input.format_map({"instruction": instruction, "input": ""})
        x_cln = prompt_input.format_map({"instruction": instruction, "input": clean_input})
        x_pois = prompt_input.format_map({"instruction": instruction, "input": poisoned_input})

        batch_primary.append(x_pri)
        batch_primary_clean.append(x_cln)
        batch_primary_poisoned.append(x_pois)

    return batch_primary, batch_primary_clean, batch_primary_poisoned
