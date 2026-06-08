"""
共用模块：模型构建、训练、推理、数据加载
=========================================
被 train_and_eval.py 和 train_all.py 共同引用。
"""
from typing import Any

import torch
import torch.nn as nn
import numpy as np
import json, os, glob, random, joblib

from numpy import ndarray, dtype
try:
    from numpy._core.multiarray import _ScalarT
except ImportError:
    from typing import TypeVar
    import numpy as np
    _ScalarT = TypeVar("_ScalarT", bound=np.generic)
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup, BitsAndBytesConfig,
)
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, brier_score_loss
import lightgbm as lgb

# ============================================================
# 配置
# ============================================================
CONFIG = {
    "seed": 42,
    "data_dir": "./pan25-data",
    "ckpt_dir": "./checkpoints",

    "qwen_model": "Qwen/Qwen2.5-1.5B",
    "qwen_max_len": 512,
    "qwen_lora_r": 16,
    "qwen_lora_alpha": 32,
    "qwen_lr": 2e-4,
    "qwen_epochs": 3,

    "bert_model": "answerdotai/ModernBERT-base",
    "bert_max_len": 512,
    "bert_lr": 2e-5,
    "bert_epochs": 3,

    "tfidf_max_features": 50000,
    "tfidf_ngram_range": (1, 3),

    "batch_size": 2,
    "warmup_ratio": 0.1,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "threshold":0.025
}

def init():
    """初始化随机种子和显存优化"""
    random.seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    torch.manual_seed(CONFIG["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(CONFIG["seed"])
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.makedirs(CONFIG["ckpt_dir"], exist_ok=True)


# ============================================================
# 数据加载
# ============================================================
def load_pan_jsonl(path):
    ids, texts, labels = [], [], []

    # 尝试多种编码：UTF-8 BOM > UTF-8 > Latin-1（Latin-1 能读任何单字节文件）
    content = None
    for enc in ["utf-8-sig", "utf-8", "latin-1"]:
        try:
            with open(path, "r", encoding=enc) as f:
                content = f.read()
            break
        except UnicodeDecodeError:
            continue

    if content is None:
        print(f"  警告：无法读取 {path}，跳过")
        return ids, texts, labels

    for i, line in enumerate(content.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            print(f"  警告：{os.path.basename(path)} 第 {i+1} 行 JSON 解析失败，跳过")
            continue
        ids.append(item["id"])
        texts.append(item["text"])
        labels.append(item.get("label"))
    return ids, texts, labels


def load_dir(d):
    ids, texts, labels = [], [], []
    for fpath in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
        i, t, l = load_pan_jsonl(fpath)
        ids.extend(i); texts.extend(t); labels.extend(l)
        print(f"  {os.path.basename(fpath)}: {len(i)} 条")
    return ids, texts, labels


# ============================================================
# Dataset & 分词
# ============================================================
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, input_ids, attention_mask, labels=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels
    def __len__(self): return len(self.input_ids)
    def __getitem__(self, idx):
        item = {"input_ids": self.input_ids[idx], "attention_mask": self.attention_mask[idx]}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def tokenize(texts, tokenizer, max_len):
    enc = tokenizer(texts, max_length=max_len, padding="max_length",
                    truncation=True, return_tensors="pt")
    return enc["input_ids"], enc["attention_mask"]


# ============================================================
# 模型构建
# ============================================================
def build_bert():
    tok = AutoTokenizer.from_pretrained(CONFIG["bert_model"])
    model = AutoModelForSequenceClassification.from_pretrained(
        CONFIG["bert_model"], num_labels=2)
    model.gradient_checkpointing_enable()
    return tok, model


def build_qwen():
    tok = AutoTokenizer.from_pretrained(CONFIG["qwen_model"], trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id

    # ★ 关键修复：decoder 模型必须左填充
    # 右填充时，pad_token == eos_token 导致模型无法区分文本结尾和 padding，
    # 取到错误位置的隐藏状态，分类头等于在随机向量上做判断。
    # 左填充后，真实文本始终在序列最右端，最后一个 token 一定是有效 token。
    tok.padding_side = "left"

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        CONFIG["qwen_model"], num_labels=2,
        quantization_config=bnb, device_map="auto", trust_remote_code=True)
    model.config.pad_token_id = tok.pad_token_id

    # ★ 扩大 LoRA 覆盖范围：1.5B 小模型需要更多可训练参数才能有效适配
    lora = LoraConfig(task_type=TaskType.SEQ_CLS, r=CONFIG["qwen_lora_r"],
                      lora_alpha=CONFIG["qwen_lora_alpha"],
                      lora_dropout=0.1,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    model = get_peft_model(model, lora)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  可训练参数: {n_train:,} / {n_total:,} ({100*n_train/n_total:.2f}%)")
    return tok, model


def load_qwen_from_ckpt(ckpt_path):
    """从检查点加载 Qwen（量化基座 + LoRA 权重）"""
    tok = AutoTokenizer.from_pretrained(CONFIG["qwen_model"], trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"  # ★ 加载时也必须左填充

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    base = AutoModelForSequenceClassification.from_pretrained(
        CONFIG["qwen_model"], num_labels=2,
        quantization_config=bnb, device_map="auto", trust_remote_code=True)
    base.config.pad_token_id = tok.pad_token_id
    model = PeftModel.from_pretrained(base, ckpt_path)
    return tok, model


class TfidfDetector:
    def __init__(self):
        self.vec = TfidfVectorizer(
            max_features=CONFIG["tfidf_max_features"],
            ngram_range=CONFIG["tfidf_ngram_range"],
            sublinear_tf=True, strip_accents="unicode")
        self.clf = lgb.LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbose=-1, n_jobs=-1)

    def fit(self, texts, labels):
        self.clf.fit(self.vec.fit_transform(texts), labels)

    def predict_proba(self, texts):
        return self.clf.predict_proba(self.vec.transform(texts))[:, 1]


# ============================================================
# 训练 & 推理
# ============================================================
def train_model(model, tok, tr_texts, tr_labels, va_texts, va_labels, max_len, lr, epochs):
    """训练一个 Transformer 模型，有验证集则每 epoch 打印指标"""
    device = CONFIG["device"]
    tr_ids, tr_mask = tokenize(tr_texts, tok, max_len)
    tr_loader = DataLoader(TextDataset(tr_ids, tr_mask, tr_labels),
                           batch_size=CONFIG["batch_size"], shuffle=True)

    has_val = va_texts is not None and len(va_texts) > 0
    if has_val:
        va_ids, va_mask = tokenize(va_texts, tok, max_len)
        va_loader = DataLoader(TextDataset(va_ids, va_mask, va_labels),
                               batch_size=CONFIG["batch_size"])

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    total_steps = len(tr_loader) * epochs
    sched = get_linear_schedule_with_warmup(
        opt, int(total_steps * CONFIG["warmup_ratio"]), total_steps)

    for ep in range(epochs):
        model.train()
        total_loss = 0
        for batch in tr_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            total_loss += loss.item()

        avg_loss = total_loss / len(tr_loader)
        if has_val:
            va_probs = predict(model, va_loader, device)
            va_f1 = f1_score(va_labels, (va_probs >= 0.5).astype(int))
            print(f"    Epoch {ep+1}/{epochs}  loss={avg_loss:.4f}  val_f1={va_f1:.4f}")
        else:
            print(f"    Epoch {ep+1}/{epochs}  loss={avg_loss:.4f}")

    return model


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs = []
    for batch in loader:
        batch.pop("labels", None)
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits.float()
        probs.extend(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
    return np.array(probs)


def predict_texts(model, tok, texts, max_len):
    """对原始文本做推理，返回 AI 概率"""
    ids, mask = tokenize(texts, tok, max_len)
    loader = DataLoader(TextDataset(ids, mask), batch_size=CONFIG["batch_size"])
    return predict(model, loader, CONFIG["device"])


# ============================================================
# 训练或加载（自动缓存）
# ============================================================
def train_or_load_qwen(tr_texts, tr_labels, va_texts=None, va_labels=None, tag=""):
    ckpt = os.path.join(CONFIG["ckpt_dir"], f"qwen_lora{tag}")
    if os.path.exists(ckpt):
        print(f"  ★ 加载缓存: {ckpt}")
        return load_qwen_from_ckpt(ckpt)
    else:
        tok, model = build_qwen()
        model = train_model(model, tok, tr_texts, tr_labels, va_texts, va_labels,
                            CONFIG["qwen_max_len"], CONFIG["qwen_lr"], CONFIG["qwen_epochs"])
        model.save_pretrained(ckpt)
        print(f"  ★ 已保存: {ckpt}")
        return tok, model


def train_or_load_bert(tr_texts, tr_labels, va_texts=None, va_labels=None, tag=""):
    ckpt = os.path.join(CONFIG["ckpt_dir"], f"bert{tag}.pt")
    tok, model = build_bert()
    if os.path.exists(ckpt):
        print(f"  ★ 加载缓存: {ckpt}")
        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
        model.to(CONFIG["device"])
    else:
        model.to(CONFIG["device"])
        model = train_model(model, tok, tr_texts, tr_labels, va_texts, va_labels,
                            CONFIG["bert_max_len"], CONFIG["bert_lr"], CONFIG["bert_epochs"])
        torch.save(model.state_dict(), ckpt)
        print(f"  ★ 已保存: {ckpt}")
    return tok, model


def train_or_load_tfidf(tr_texts, tr_labels, tag=""):
    ckpt = os.path.join(CONFIG["ckpt_dir"], f"tfidf{tag}.joblib")
    if os.path.exists(ckpt):
        print(f"  ★ 加载缓存: {ckpt}")
        return joblib.load(ckpt)
    else:
        det = TfidfDetector()
        det.fit(tr_texts, tr_labels)
        joblib.dump(det, ckpt)
        print(f"  ★ 已保存: {ckpt}")
        return det


# ============================================================
# 集成推理 & 评估
# ============================================================
def ensemble_predict(tok_q: object, mod_q: object, tok_b: object, mod_b: object, tfidf: object, meta: object, texts: object) -> tuple[Any, Any, dict[str, ndarray[tuple[Any, ...], dtype[_ScalarT]] | Any]]:
    """三模型 + 元学习器集成推理"""
    probs_q = predict_texts(mod_q, tok_q, texts, CONFIG["qwen_max_len"])
    probs_b = predict_texts(mod_b, tok_b, texts, CONFIG["bert_max_len"])
    probs_t = tfidf.predict_proba(texts)
    meta_feat = np.column_stack([probs_q, probs_b, probs_t])
    final_probs = meta.predict_proba(meta_feat)[:, 1]
    final_preds = (final_probs >= CONFIG["threshold"]).astype(int)
    return final_preds, final_probs, {"qwen": probs_q, "bert": probs_b, "tfidf": probs_t}


def c_at_1(labels, preds, probs):
    labels, preds, probs = np.array(labels), np.array(preds), np.array(probs)
    conf = np.abs(probs - 0.5) >= 0.1
    if conf.sum() == 0: return 0.0
    acc = accuracy_score(labels[conf], preds[conf])
    return ((labels[conf] == preds[conf]).sum() + (~conf).sum() * acc) / len(labels)


def evaluate(labels, preds, probs, individual=None):
    """打印完整评估报告"""
    if individual:
        print("\n--- 各模型独立表现 ---")
        for name, p in individual.items():
            f1_i = f1_score(labels, (p >= 0.5).astype(int))
            auc_i = roc_auc_score(labels, p)
            print(f"  {name:10s}  F1={f1_i:.4f}  AUC={auc_i:.4f}")

    print("\n--- 集成表现 ---")
    print(classification_report(labels, preds, target_names=["人类(0)", "AI(1)"], digits=4))

    auc = roc_auc_score(labels, probs)
    brier = 1.0 - brier_score_loss(labels, probs)
    c1 = c_at_1(labels, preds, probs)
    f1 = f1_score(labels, preds)
    print(f"ROC-AUC:   {auc:.4f}")
    print(f"1-Brier:   {brier:.4f}")
    print(f"C@1:       {c1:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"PAN Mean:  {(auc + brier + c1 + f1) / 4:.4f}")


def write_predictions(ids, probs, path):
    with open(path, "w", encoding="utf-8") as f:
        for sid, prob in zip(ids, probs):
            f.write(json.dumps({"id": sid, "label": round(float(prob), 4)}) + "\n")
    print(f"\n预测文件已保存: {path}")
