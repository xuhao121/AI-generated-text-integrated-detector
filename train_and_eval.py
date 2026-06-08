"""
课程论文版：训练集训练 → 验证集评估
===================================
每个模型只训练一次，权重缓存在 checkpoints/ 下。
重新运行自动跳过已完成的训练。

用法：python train_and_eval.py
"""

from shared import *
import argparse
def main():
    init()
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file", default=None,
                        help="测试集 JSONL 文件路径（无标签）")
    args = parser.parse_args()
    # ---- 加载数据 ----
    print("加载数据集...\n")
    train_dir = os.path.join(CONFIG["data_dir"], "train")
    val_dir = os.path.join(CONFIG["data_dir"], "validation")

    print("训练集：")
    tr_ids, tr_texts, tr_labels = load_dir(train_dir)
    print(f"  共 {len(tr_texts)} 条 (人类:{tr_labels.count(0)} AI:{tr_labels.count(1)})\n")

    if os.path.isdir(val_dir):
        print("验证集：")
        va_ids, va_texts, va_labels = load_dir(val_dir)
        print(f"  共 {len(va_texts)} 条\n")
    else:
        from sklearn.model_selection import train_test_split
        tr_ids, va_ids, tr_texts, va_texts, tr_labels, va_labels = train_test_split(
            tr_ids, tr_texts, tr_labels, test_size=0.2,
            stratify=tr_labels, random_state=CONFIG["seed"])
        print(f"  无验证集目录，从训练集切分 20%: {len(va_texts)} 条\n")

    # ---- 训练三个模型（用训练集训练，验证集前200条监控过程）----
    print("=" * 50)
    print("[1/3] Qwen2.5 + QLoRA")
    print("=" * 50)
    tok_q, mod_q = train_or_load_qwen(
        tr_texts, tr_labels, va_texts[:200], va_labels[:200], tag="_eval")

    print(f"\n{'=' * 50}")
    print("[2/3] ModernBERT")
    print("=" * 50)
    tok_b, mod_b = train_or_load_bert(
        tr_texts, tr_labels, va_texts[:200], va_labels[:200], tag="_eval")

    print(f"\n{'=' * 50}")
    print("[3/3] TF-IDF + LightGBM")
    print("=" * 50)
    tfidf = train_or_load_tfidf(tr_texts, tr_labels, tag="_eval")

    # ---- 验证集预测 → 训练元学习器 ----
    print(f"\n{'=' * 50}")
    print("验证集预测 + 训练元学习器")
    print("=" * 50)
    meta_ckpt = os.path.join(CONFIG["ckpt_dir"], "meta_eval.joblib")
    print(meta_ckpt)
    if os.path.exists(meta_ckpt):
        print("  ★ 加载缓存的元学习器")
        meta = joblib.load(meta_ckpt)
    else:
        probs_q = predict_texts(mod_q, tok_q, va_texts, CONFIG["qwen_max_len"])
        probs_b = predict_texts(mod_b, tok_b, va_texts, CONFIG["bert_max_len"])
        probs_t = tfidf.predict_proba(va_texts)

        meta_feat = np.column_stack([probs_q, probs_b, probs_t])
        meta = lgb.LGBMClassifier(n_estimators=100, max_depth=3,
                                  learning_rate=0.1, random_state=42, verbose=-1)
        meta.fit(meta_feat, va_labels)
        joblib.dump(meta, meta_ckpt)

    for i, name in enumerate(["Qwen2.5", "ModernBERT", "TF-IDF"]):
        print(f"  {name} 重要性: {meta.feature_importances_[i]:.1f}")

    if args.test_file:
        print(f"\n{'=' * 50}")
        print(f"测试集推理: {args.test_file}")
        print("=" * 50)

        test_ids, test_texts, label = load_pan_jsonl(args.test_file)
        print(f"  {len(test_texts)} 条测试样本")

        preds, probs, _ = ensemble_predict(tok_q, mod_q, tok_b, mod_b, tfidf, meta, test_texts)
        evaluate(label, preds, probs)
        output_path = "predictions_test.jsonl"
        write_predictions(test_ids, probs, output_path)
        print(f"\n已为 {len(test_texts)} 条测试样本生成预测结果")
    else:
        print("\n未指定测试集。用法：python train_and_eval.py --test_file path/to/test.jsonl")
        print("模型已全量训练完毕并保存，等拿到测试集后直接运行上面的命令即可。")

if __name__ == "__main__":
    main()
