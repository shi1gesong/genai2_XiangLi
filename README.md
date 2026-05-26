# EmoCompanion 🤗
### A Multimodal Emotional Support Chatbot

EmoCompanion is a fine-tuned conversational AI that provides empathetic, emotionally aware responses. It combines **DialoGPT** for dialogue generation with **CLIP** for multimodal image understanding, enabling users to share both text and images while receiving emotionally appropriate support.

---

## Features

- 💬 **Empathetic dialogue** — fine-tuned on 38,000+ emotional support conversations
- 🖼️ **Multimodal input** — upload images and receive emotionally relevant responses
- 🎨 **Chatbot GUI** — real-time conversational interface built with Gradio
- 📊 **Experiment tracking** — training metrics logged with Weights & Biases

---

## Dataset

| Dataset | Size | Source |
|---|---|---|
| EmpatheticDialogues (Facebook Research) | 25,000+ conversations, 32 emotion categories | HuggingFace |
| DailyDialog | 13,000+ daily conversations with emotion annotations | HuggingFace |

---

## Model Architecture

```
User Text  ──────────────────────────────► DialoGPT-medium
                                                  ▲
User Image ──► CLIP Image Encoder ──► Projection Layer
```

- **Base model:** `microsoft/DialoGPT-medium` (GPT-2 based, pre-trained on Reddit)
- **Fine-tuning:** Combined EmpatheticDialogues + DailyDialog corpus
- **Multimodal fusion:** CLIP image embeddings projected and prepended as soft prompt tokens

---

## Extra Criteria

| Criterion | Implementation |
|---|---|
| Chatbot GUI | Gradio interface with multi-turn dialogue and warm companion-style personality |
| MLOps | Weights & Biases (wandb) for loss/perplexity tracking across fine-tuning runs |
| Multimodal (Text + Image) | CLIP-encoded image embeddings fused with text input for emotionally conditioned generation |

---

## Project Structure

```
emocompanion/
├── data/
│   └── preprocess.py          # Dataset loading and preprocessing
├── model/
│   ├── train.py               # Fine-tuning script
│   ├── model.py               # EmoCompanion model class (DialoGPT + CLIP fusion)
│   └── evaluate.py            # Perplexity and qualitative evaluation
├── app/
│   └── gradio_app.py          # Chatbot GUI
├── notebooks/
│   └── exploration.ipynb      # Data exploration and ablations
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/emocompanion.git
cd emocompanion
pip install -r requirements.txt
```

**Run training:**
```bash
python model/train.py --model_name microsoft/DialoGPT-medium --epochs 3 --use_wandb
```

**Launch chatbot:**
```bash
python app/gradio_app.py
```

---

## Results

> Training metrics and evaluation results will be added here after fine-tuning.

---

## References

- [DialoGPT (Microsoft)](https://huggingface.co/microsoft/DialoGPT-medium)
- [CLIP (OpenAI)](https://huggingface.co/openai/clip-vit-base-patch32)
- [EmpatheticDialogues](https://huggingface.co/datasets/empathetic_dialogues)
- [DailyDialog](https://huggingface.co/datasets/daily_dialog)
