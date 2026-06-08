"""
EchoHeart Gradio chatbot interface.
Supports multi-turn text conversation and optional image upload.
"""

import os
import sys

import torch
import gradio as gr
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from model.model import EchoHeart

CHECKPOINT = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "best_model.pt")
MODEL_NAME = "microsoft/DialoGPT-medium"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading EchoHeart model...")
model = EchoHeart(dialogpt_name=MODEL_NAME)
if os.path.exists(CHECKPOINT):
    model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu"))
    print(f"Loaded checkpoint: {CHECKPOINT}")
else:
    print("No checkpoint found — using base weights.")
model.to(device)
model.eval()


def chat(user_message: str, image, history: list):
    """
    history: list of [user_msg, bot_msg] pairs (Gradio format)
    """
    context = []
    for user_turn, bot_turn in history:
        context.append(user_turn)
        if bot_turn:
            context.append(bot_turn)
    context.append(user_message)

    img_tensor = None
    if image is not None:
        pil_image = Image.fromarray(image).convert("RGB")
        img_tensor = model.clip_preprocess(pil_image).unsqueeze(0).to(device)

    response = model.generate_response(context, image=img_tensor)

    history = history + [[user_message, response]]
    return "", None, history


with gr.Blocks(title="EchoHeart 🤗", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# EchoHeart 🤗\n### Your emotionally aware AI companion")

    chatbot = gr.Chatbot(height=480, label="Conversation")

    with gr.Row():
        with gr.Column(scale=4):
            msg_box = gr.Textbox(
                placeholder="Share what's on your mind...",
                label="Your message",
                lines=2,
                show_label=False,
            )
        with gr.Column(scale=1):
            image_box = gr.Image(label="Image (optional)", type="numpy", height=100)

    with gr.Row():
        send_btn = gr.Button("Send", variant="primary")
        clear_btn = gr.Button("Clear")

    state = gr.State([])

    send_btn.click(
        chat,
        inputs=[msg_box, image_box, state],
        outputs=[msg_box, image_box, state],
    ).then(lambda h: h, inputs=[state], outputs=[chatbot])
    msg_box.submit(
        chat,
        inputs=[msg_box, image_box, state],
        outputs=[msg_box, image_box, state],
    ).then(lambda h: h, inputs=[state], outputs=[chatbot])
    clear_btn.click(lambda: ([], []), outputs=[state, chatbot])


if __name__ == "__main__":
    demo.launch(share=False)
