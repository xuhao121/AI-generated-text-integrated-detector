"""
AI 文本检测工具（网页界面）
==========================
启动后会打开一个网页，粘贴文本即可检测。

前置条件：先运行 train_and_eval.py 完成训练。
额外依赖：pip install gradio

用法：python detect.py
      python detect.py --tag _full    # 使用全量训练的模型
      python detect.py --share        # 生成公网链接，可分享给他人
"""

import argparse, os, sys

def check_models(tag):
    ckpt = "./checkpoints"
    needed = [f"qwen_lora{tag}", f"bert{tag}.pt", f"tfidf{tag}.joblib", f"meta{tag}.joblib"]
    missing = [n for n in needed if not os.path.exists(os.path.join(ckpt, n))]
    if missing:
        print("错误：模型文件缺失，请先完成训练：")
        for m in missing:
            print(f"  checkpoints/{m}")
        print("\n运行 python train_and_eval.py 训练后再使用。")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="_eval", help="权重标签（_eval 或 _full）")
    parser.add_argument("--share", action="store_true", help="生成公网分享链接")
    args = parser.parse_args()

    check_models(args.tag)

    # ---- 加载模型 ----
    from shared import (CONFIG, init, load_qwen_from_ckpt, build_bert,
                        predict_texts, joblib, torch, np)
    import gradio as gr

    init()
    ckpt = CONFIG["ckpt_dir"]
    tag = args.tag

    print("加载模型...")
    print("  [1/4] Qwen2.5")
    tok_q, mod_q = load_qwen_from_ckpt(os.path.join(ckpt, f"qwen_lora{tag}"))
    print("  [2/4] ModernBERT")
    tok_b, mod_b = build_bert()
    mod_b.load_state_dict(torch.load(os.path.join(ckpt, f"bert{tag}.pt"),
                                     map_location="cpu", weights_only=True))
    mod_b.to(CONFIG["device"])
    print("  [3/4] TF-IDF")
    tfidf = joblib.load(os.path.join(ckpt, f"tfidf{tag}.joblib"))
    print("  [4/4] 元学习器")
    meta = joblib.load(os.path.join(ckpt, f"meta{tag}.joblib"))
    print("  加载完成\n")

    # ---- 检测函数 ----
    def detect(text):
        if not text or not text.strip():
            return "", "", ""

        text = text.strip()

        prob_q = predict_texts(mod_q, tok_q, [text], CONFIG["qwen_max_len"])[0]
        prob_b = predict_texts(mod_b, tok_b, [text], CONFIG["bert_max_len"])[0]
        prob_t = tfidf.predict_proba([text])[0]

        meta_feat = np.array([[prob_q, prob_b, prob_t]])
        prob = meta.predict_proba(meta_feat)[0, 1]
        is_ai = prob >= 0.5

        # 判定结果
        if prob >= 0.85:
            label = "🤖 AI 生成（高置信度）"
        elif prob >= 0.5:
            label = "🤖 AI 生成（中置信度）"
        elif prob >= 0.15:
            label = "✍️ 人类写作（中置信度）"
        else:
            label = "✍️ 人类写作（高置信度）"

        confidence = prob if is_ai else (1 - prob)

        # 各模型详情
        details = (
            f"综合判定概率: {prob:.1%}（>50% 判为 AI）\n"
            f"判定置信度:   {confidence:.1%}\n"
            f"\n"
            f"--- 各模型独立判断 ---\n"
            f"Qwen2.5:    {prob_q:.1%} {'← AI' if prob_q >= 0.5 else '← 人类'}\n"
            f"ModernBERT: {prob_b:.1%} {'← AI' if prob_b >= 0.5 else '← 人类'}\n"
            f"TF-IDF:     {prob_t:.1%} {'← AI' if prob_t >= 0.5 else '← 人类'}\n"
            f"\n"
            f"字数: {len(text)} | Token 数（估）: ~{len(text.split())}"
        )

        # 进度条 HTML
        bar_color = "#ef4444" if is_ai else "#22c55e"
        bar_html = f"""
        <div style="margin: 10px 0;">
          <div style="display:flex; justify-content:space-between; font-size:14px; margin-bottom:4px;">
            <span>✍️ 人类</span>
            <span>🤖 AI</span>
          </div>
          <div style="background:#e5e7eb; border-radius:8px; height:28px; overflow:hidden;">
            <div style="background:{bar_color}; width:{prob*100:.1f}%;
                        height:100%; border-radius:8px;
                        transition: width 0.5s ease;"></div>
          </div>
          <div style="text-align:center; margin-top:4px; font-size:13px; color:#666;">
            AI 概率: {prob:.1%}
          </div>
        </div>
        """

        return label, bar_html, details

    # ---- Gradio 界面 ----
    with gr.Blocks(title="AI 文本检测", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🔍 AI 文本检测工具")
        gr.Markdown("粘贴一段文本，检测它是人类写的还是 AI 生成的。")

        with gr.Row():
            with gr.Column(scale=3):
                text_input = gr.Textbox(
                    label="输入文本",
                    placeholder="在这里粘贴要检测的文本...",
                    lines=12,
                    max_lines=30,
                )
                with gr.Row():
                    submit_btn = gr.Button("🔍 检测", variant="primary", scale=2)
                    clear_btn = gr.ClearButton([text_input], value="清空", scale=1)

            with gr.Column(scale=2):
                result_label = gr.Textbox(label="检测结果", interactive=False)
                result_bar = gr.HTML(label="概率")
                result_detail = gr.Textbox(label="详细信息", lines=10, interactive=False)

        submit_btn.click(fn=detect, inputs=text_input,
                         outputs=[result_label, result_bar, result_detail])
        text_input.submit(fn=detect, inputs=text_input,
                          outputs=[result_label, result_bar, result_detail])

        gr.Markdown("""
        ---
        **使用说明**
        - 建议输入 50 字以上的文本，文本越长检测越准确
        - 综合概率 > 50% 判为 AI 生成，< 50% 判为人类写作
        - 三个模型从不同角度检测：Qwen（语义）、ModernBERT（上下文）、TF-IDF（词汇统计）
        """)

    print("启动检测界面...")
    app.launch(share=args.share, inbrowser=True)


if __name__ == "__main__":
    main()
