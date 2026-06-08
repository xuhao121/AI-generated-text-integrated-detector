"""
统一 baseline 测试脚本：Qwen / ModernBERT / TF-IDF
在用户提供的测试文件上一次性评测三个 baseline。

用法：
    python baseline.py --test_file <path>

测试文件含标签则同时输出指标，不含标签则只写预测文件。
"""
import os, gc, argparse
import torch, joblib
from torch.utils.data import DataLoader

from shared import (CONFIG, init, build_bert, load_qwen_from_ckpt,
                    predict, TextDataset, tokenize, evaluate,
                    load_pan_jsonl, write_predictions)


# ---------- 通用工具 ----------

def make_loader(texts, tokenizer, max_len, labels=None, batch_size=None, shuffle=False):
    """texts -> DataLoader，使用 shared.tokenize + shared.TextDataset。"""
    input_ids, attention_mask = tokenize(texts, tokenizer, max_len)
    ds = TextDataset(input_ids, attention_mask, labels)
    bs = batch_size or CONFIG.get("batch_size", 16)
    return DataLoader(ds, batch_size=bs, shuffle=shuffle)


def load_test_set(path):
    """从文件加载测试集；labels 可能为空。"""
    print(f"=== 加载测试集 {path} ===")
    ids, texts, labels = load_pan_jsonl(path)
    has_labels = labels is not None and len(labels) == len(texts) and any(l is not None for l in labels)
    print(f"样本数: {len(texts)}，{'含标签（将评估）' if has_labels else '无标签（仅写预测）'}")
    return ids, texts, (labels if has_labels else None)


def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------- 每个 baseline 的差异：loader -> predict_fn(texts) ----------

def _qwen_loader(ckpt):
    tok, mod = load_qwen_from_ckpt(ckpt)
    max_len = CONFIG["qwen_max_len"]
    return lambda texts: predict(mod, make_loader(texts, tok, max_len), CONFIG["device"])


def _bert_loader(ckpt):
    tok, mod = build_bert()
    mod.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    mod.to(CONFIG["device"])
    max_len = CONFIG["bert_max_len"]
    return lambda texts: predict(mod, make_loader(texts, tok, max_len), CONFIG["device"])


def _tfidf_loader(ckpt):
    tfidf = joblib.load(ckpt)
    return lambda texts: tfidf.predict_proba(texts)   # shared.TfidfDetector 用 LightGBM，方法名是 predict_proba


BASELINES = [
    dict(ckpt="qwen_lora_eval",    title="Qwen2.5-1.5B + QLoRA",          out="qwen_predictions_test.jsonl",  loader=_qwen_loader),
    dict(ckpt="bert_eval.pt",      title="ModernBERT",                    out="bert_predictions_test.jsonl",  loader=_bert_loader),
    dict(ckpt="tfidf_eval.joblib", title="TF-IDF + LightGBM",              out="tfidf_predictions_test.jsonl", loader=_tfidf_loader),
]


# ---------- 主流程 ----------

def run_one(cfg, ids, texts, labels):
    ck = os.path.join(CONFIG["ckpt_dir"], cfg["ckpt"])
    if not os.path.exists(ck):
        print(f"\n[跳过 {cfg['title']}] 缺少检查点 {ck}（请先运行 train.py）")
        return

    print(f"\n=== 加载 {cfg['title']} ({ck}) ===")
    predict_fn = cfg["loader"](ck)

    print(f"=== 推理 ({len(texts)} 条) ===")
    probs = predict_fn(texts)
    preds = (probs >= CONFIG["threshold"]).astype(int)

    if labels is not None:
        print(f"\n[基线：{cfg['title']}（单模型, 阈值 0.5）]")
        evaluate(labels, preds, probs)

    out_path = f"./{cfg['out']}"
    write_predictions(ids, probs, out_path)
    print(f"预测已写入 {out_path}")

    del predict_fn
    free_gpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_file", required=True, help="测试集 jsonl 路径")
    args = ap.parse_args()

    init()
    ids, texts, labels = load_test_set(args.test_file)

    for cfg in BASELINES:
        run_one(cfg, ids, texts, labels)


if __name__ == "__main__":
    main()
