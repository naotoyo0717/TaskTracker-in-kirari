# -*- coding: utf-8 -*-
"""
TaskTrackerデータ用: 生成+活性値の統合取得（clean/poisoned両方generate）。
先頭からN件を処理し、t-SNE用の活性化デルタと、入力・出力を対応づけて記録する。

出力:
  <out>/primary_acts.pt      [N,32,4096]  primary活性値
  <out>/clean_acts.pt        [N,32,4096]  clean活性値（generateのprefillから）
  <out>/poisoned_acts.pt     [N,32,4096]  poisoned活性値
  <out>/records.json         各サンプルの入力・出力・sample_id
  <out>/readable_log.txt     入力と出力を一目で見られる人間可読ログ

使い方:
  python gen_with_acts_tt.py --cond llama_text --n 10000
"""
import os
import json
import argparse
import torch
from tqdm import tqdm

from task_tracker.config.models import cache_dir, models, activation_parent_dir
from task_tracker.utils.model import load_model
from task_tracker.utils.data_struq import format_prompts_struq

TOKEN = -1
MAX_NEW_TOKENS = 200


def forward_last_token(model, text):
    ids = model.tokenizer(text, return_tensors="pt").input_ids.to(model.model.device)
    with torch.no_grad():
        out = model.model(ids, output_hidden_states=True)
    hs = out.hidden_states
    return torch.stack([hs[i][:, TOKEN].cpu() for i in range(1, len(hs))]).squeeze(1)


def generate_with_acts(model, text):
    tok = model.tokenizer
    ids = tok(text, return_tensors="pt").input_ids.to(model.model.device)
    prompt_len = ids.shape[1]
    with torch.no_grad():
        gen = model.model.generate(
            ids,
            attention_mask=torch.ones_like(ids),
            do_sample=False,
            max_new_tokens=MAX_NEW_TOKENS,
            output_hidden_states=True,
            return_dict_in_generate=True,
            pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
        )
    response = tok.decode(gen.sequences[0][prompt_len:], skip_special_tokens=True)
    step0 = gen.hidden_states[0]
    acts = torch.stack([step0[i][:, -1].cpu() for i in range(1, len(step0))]).squeeze(1)
    return response, acts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cond', type=str, required=True)
    parser.add_argument('--n', type=int, default=10000)
    args = parser.parse_args()

    model = models[args.cond]
    print(f"=== {args.cond} モデルロード ===", flush=True)
    loaded = load_model(model.name, cache_dir=cache_dir, torch_dtype=model.torch_dtype)
    model.tokenizer = loaded["tokenizer"]
    model.model = loaded["model"]
    model.model.eval()

    # clean と poisoned のデータ（同じ順・同じsample_id対応）
    clean_data = json.load(open(model.data["test_clean"]))[:args.n]
    poisoned_data = json.load(open(model.data["test_poisoned"]))[:args.n]
    n = min(len(clean_data), len(poisoned_data))
    print(f"処理件数: {n}", flush=True)

    out_dir = os.path.join(activation_parent_dir, args.cond, "gen_acts_tt")
    os.makedirs(out_dir, exist_ok=True)

    primary_acts, clean_acts, poisoned_acts = [], [], []
    records = []
    readable = open(os.path.join(out_dir, "readable_log.txt"), "w")

    for i in tqdm(range(n)):
        cs = clean_data[i]
        ps = poisoned_data[i]

        # clean側の3文脈（primaryはcleanのprimary_task_promptから）
        p_prompts, c_prompts, _ = format_prompts_struq([cs], model.struq_type, model.apply_filter)
        # poisoned側
        _, _, po_prompts = format_prompts_struq([ps], model.struq_type, model.apply_filter)

        primary_prompt = p_prompts[0]
        clean_prompt = c_prompts[0]
        poisoned_prompt = po_prompts[0]

        # primary: forward（生成不要）
        p_act = forward_last_token(model, primary_prompt)
        # clean: generate（応答も記録）
        clean_resp, c_act = generate_with_acts(model, clean_prompt)
        # poisoned: generate（応答も記録）
        pois_resp, po_act = generate_with_acts(model, poisoned_prompt)

        primary_acts.append(p_act)
        clean_acts.append(c_act)
        poisoned_acts.append(po_act)

        rec = {
            "id": i,  # t-SNEの点と対応するID（先頭からの順）
            "condition": args.cond,
            "primary_task_prompt": ps.get("primary_task_prompt", ""),
            "secondary_task_prompt": ps.get("secondary_task_prompt", ""),  # 注入された別タスク
            "trigger": ps.get("trigger", ""),                              # 注入の冒頭
            "clean_input": cs.get("orig_text", ""),
            "poisoned_input": ps.get("final_text_paragraph", ""),
            "clean_response": clean_resp,
            "poisoned_response": pois_resp,
        }
        records.append(rec)

        # 人間可読ログ: 入力と出力を一目で
        readable.write(f"{'='*80}\n")
        readable.write(f"[ID {i}] condition={args.cond}\n")
        readable.write(f"--- 主タスク（命令）---\n{ps.get('primary_task_prompt','')}\n\n")
        readable.write(f"--- 注入された別タスク（secondary）---\n{ps.get('secondary_task_prompt','')}\n\n")
        readable.write(f"--- clean入力 ---\n{cs.get('orig_text','')[:300]}\n")
        readable.write(f"--- clean出力 ---\n{clean_resp[:300]}\n\n")
        readable.write(f"--- poisoned入力 ---\n{ps.get('final_text_paragraph','')[:400]}\n")
        readable.write(f"--- poisoned出力 ---\n{pois_resp[:400]}\n\n")
        readable.flush()

    readable.close()
    torch.save(torch.stack(primary_acts), os.path.join(out_dir, "primary_acts.pt"))
    torch.save(torch.stack(clean_acts), os.path.join(out_dir, "clean_acts.pt"))
    torch.save(torch.stack(poisoned_acts), os.path.join(out_dir, "poisoned_acts.pt"))
    with open(os.path.join(out_dir, "records.json"), "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"\n=== {args.cond} 完了: {n}件 ===", flush=True)
    print(f"保存先: {out_dir}", flush=True)
    print(f"  primary_acts.pt / clean_acts.pt / poisoned_acts.pt", flush=True)
    print(f"  records.json（構造化）, readable_log.txt（目視用）", flush=True)


if __name__ == '__main__':
    main()
