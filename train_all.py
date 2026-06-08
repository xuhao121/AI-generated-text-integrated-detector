"""
全量训练版：训练集+验证集合并训练 → 对测试集推理
================================================
用于比赛提交或追求最大性能。
每个模型只训练一次，权重缓存在 checkpoints/ 下。

用法：python train_all.py
      python train_all.py --test_file path/to/test.jsonl
"""

import argparse
from shared import *

def main():
    init()

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file", default=None,
                        help="测试集 JSONL 文件路径（无标签）")
    args = parser.parse_args()

    # ---- 加载并合并训练+验证数据 ----
    print("加载并合并全部标注数据...\n")
    train_dir = os.path.join(CONFIG["data_dir"], "train")
    val_dir = os.path.join(CONFIG["data_dir"], "validation")

    all_ids, all_texts, all_labels = [], [], []

    print("训练集：")
    tr_ids, tr_texts, tr_labels = load_dir(train_dir)
    all_ids.extend(tr_ids); all_texts.extend(tr_texts); all_labels.extend(tr_labels)

    if os.path.isdir(val_dir):
        print("验证集：")
        va_ids, va_texts, va_labels = load_dir(val_dir)
        all_ids.extend(va_ids); all_texts.extend(va_texts); all_labels.extend(va_labels)

    print(f"\n合并后: {len(all_texts)} 条 (人类:{all_labels.count(0)} AI:{all_labels.count(1)})\n")

    # ---- 训练三个模型（全量数据，无验证集监控）----
    print("=" * 50)
    print("[1/3] Qwen2.5 + QLoRA（全量）")
    print("=" * 50)
    tok_q, mod_q = train_or_load_qwen(all_texts, all_labels, tag="_full")

    print(f"\n{'=' * 50}")
    print("[2/3] ModernBERT（全量）")
    print("=" * 50)
    tok_b, mod_b = train_or_load_bert(all_texts, all_labels, tag="_full")

    print(f"\n{'=' * 50}")
    print("[3/3] TF-IDF + LightGBM（全量）")
    print("=" * 50)
    tfidf = train_or_load_tfidf(all_texts, all_labels, tag="_full")

    # ---- 元学习器：用训练数据的预测来拟合 ----
    # 注意：这里模型预测自己的训练集，元学习器有过拟合风险，
    # 但对于比赛提交（不需要评估，只需要预测）这没有问题。
    print(f"\n{'=' * 50}")
    print("训练元学习器")
    print("=" * 50)

    meta_ckpt = os.path.join(CONFIG["ckpt_dir"], "meta_full.joblib")
    if os.path.exists(meta_ckpt):
        print("  ★ 加载缓存的元学习器")
        meta = joblib.load(meta_ckpt)
    else:
        probs_q = predict_texts(mod_q, tok_q, all_texts, CONFIG["qwen_max_len"])
        probs_b = predict_texts(mod_b, tok_b, all_texts, CONFIG["bert_max_len"])
        probs_t = tfidf.predict_proba(all_texts)
        meta_feat = np.column_stack([probs_q, probs_b, probs_t])

        meta = lgb.LGBMClassifier(n_estimators=100, max_depth=3,
                                  learning_rate=0.1, random_state=42, verbose=-1)
        meta.fit(meta_feat, all_labels)
        joblib.dump(meta, meta_ckpt)

    for i, name in enumerate(["Qwen2.5", "ModernBERT", "TF-IDF"]):
        print(f"  {name} 重要性: {meta.feature_importances_[i]:.1f}")

    # ---- 对测试集推理 ----
    if args.test_file:
        print(f"\n{'=' * 50}")
        print(f"测试集推理: {args.test_file}")
        print("=" * 50)

        test_ids, test_texts, _ = load_pan_jsonl(args.test_file)
        print(f"  {len(test_texts)} 条测试样本")

        preds, probs, _ = ensemble_predict(tok_q, mod_q, tok_b, mod_b, tfidf, meta, test_texts)

        output_path = os.path.join(SCRIPT_DIR, "predictions_test.jsonl")
        write_predictions(test_ids, probs, output_path)
        print(f"\n已为 {len(test_texts)} 条测试样本生成预测结果")
    else:
        print("\n未指定测试集。用法：python train_all.py --test_file path/to/test.jsonl")
        print("模型已全量训练完毕并保存，等拿到测试集后直接运行上面的命令即可。")


if __name__ == "__main__":
    main()
