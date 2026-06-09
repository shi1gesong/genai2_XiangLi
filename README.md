# EchoHeart 🤗
### A Multimodal Emotional Support Chatbot

EchoHeart is a fine-tuned conversational AI that provides empathetic, emotionally aware responses. It combines **DialoGPT** for dialogue generation with **CLIP** for multimodal image understanding, enabling users to share both text and images while receiving emotionally appropriate support.

---

## Features

- 💬 **Empathetic dialogue** — fine-tuned on 145,000+ emotional support conversations
- 🖼️ **Multimodal input** — upload images and receive emotionally relevant responses
- 🎨 **Chatbot GUI** — real-time conversational interface built with Gradio
- 📊 **Experiment tracking** — training metrics logged with Weights & Biases

---

## Dataset

| Dataset | Size | Source |
|---|---|---|
| BlendedSkillTalk (Facebook Research) | ~76,000 empathetic multi-turn conversations | HuggingFace |
| PersonaChat | ~139,000 persona-based dialogues | HuggingFace |

> Note: Original datasets (EmpatheticDialogues, DailyDialog) were replaced due to incompatibility with HuggingFace datasets >= 5.0 (legacy loading scripts no longer supported).

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
echoheart/
├── data/
│   └── preprocess.py          # Dataset loading and preprocessing
├── model/
│   ├── train.py               # Fine-tuning script
│   ├── model.py               # EchoHeart model class (DialoGPT + CLIP fusion)
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
git clone https://github.com/shi1gesong/genai2_XiangLi.git
cd genai2_XiangLi
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

| Metric | Value |
|---|---|
| Training samples | 145,067 |
| Epochs completed | 1 |
| Validation Perplexity | 20.36 |
| Training precision | bfloat16 (autocast) |
| Hardware | NVIDIA GeForce RTX 5070 Ti |

Sample response (text-only input: *"I've been feeling really lonely lately."*):
> "I'm sorry you are feeling lonely. hugs"

---

## References

- [DialoGPT (Microsoft)](https://huggingface.co/microsoft/DialoGPT-medium)
- [CLIP (OpenAI)](https://huggingface.co/openai/clip-vit-base-patch32)
- [EmpatheticDialogues](https://huggingface.co/datasets/empathetic_dialogues)
- [DailyDialog](https://huggingface.co/datasets/daily_dialog)
