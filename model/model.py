"""
EchoHeart model: DialoGPT-medium fused with CLIP image embeddings.
Image features are projected into the text embedding space and
prepended as soft prompt tokens before the dialogue context.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import open_clip


class EchoHeart(nn.Module):
    def __init__(
        self,
        dialogpt_name: str = "microsoft/DialoGPT-medium",
        clip_model_name: str = "ViT-B-32",
        clip_pretrained: str = "openai",
        freeze_clip: bool = True,
    ):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(dialogpt_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.dialogpt = AutoModelForCausalLM.from_pretrained(dialogpt_name)
        hidden_size = self.dialogpt.config.hidden_size

        self.clip, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            clip_model_name, pretrained=clip_pretrained
        )
        clip_dim = self.clip.visual.output_dim

        if freeze_clip:
            for p in self.clip.parameters():
                p.requires_grad = False

        self.image_proj = nn.Sequential(
            nn.Linear(clip_dim, hidden_size),
            nn.Tanh(),
        )

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Returns (batch, hidden_size) image soft-prompt tokens."""
        with torch.no_grad() if not self.clip.visual.training else torch.enable_grad():
            image_features = self.clip.encode_image(images)
        return self.image_proj(image_features.float()).unsqueeze(1)  # (B, 1, H)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor = None,
        images: torch.Tensor = None,
    ):
        inputs_embeds = self.dialogpt.transformer.wte(input_ids)  # (B, T, H)

        if images is not None:
            img_embeds = self.encode_image(images)  # (B, 1, H)
            inputs_embeds = torch.cat([img_embeds, inputs_embeds], dim=1)
            # Extend masks and labels for the prepended image token
            img_mask = attention_mask.new_ones(attention_mask.size(0), 1)
            attention_mask = torch.cat([img_mask, attention_mask], dim=1)
            if labels is not None:
                img_label = labels.new_full((labels.size(0), 1), -100)
                labels = torch.cat([img_label, labels], dim=1)

        return self.dialogpt(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )

    @torch.no_grad()
    def generate_response(
        self,
        context: list[str],
        image: torch.Tensor = None,
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        top_p: float = 0.9,
    ) -> str:
        eos = self.tokenizer.eos_token
        text = eos.join(context) + eos
        enc = self.tokenizer(text, return_tensors="pt")
        input_ids = enc["input_ids"].to(next(self.parameters()).device)
        attention_mask = enc["attention_mask"].to(input_ids.device)

        inputs_embeds = self.dialogpt.transformer.wte(input_ids)

        if image is not None:
            image = image.to(input_ids.device)
            img_embed = self.encode_image(image)
            inputs_embeds = torch.cat([img_embed, inputs_embeds], dim=1)
            img_mask = attention_mask.new_ones(1, 1)
            attention_mask = torch.cat([img_mask, attention_mask], dim=1)

        output = self.dialogpt.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        # Decode only newly generated tokens
        n_input = inputs_embeds.shape[1]
        new_tokens = output[0][n_input:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
