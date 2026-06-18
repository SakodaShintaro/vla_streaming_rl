from typing import Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from vla_streaming_rl.simlingo.models.encoder.internvl2_vendored.modeling_internvl_chat import (
    InternVLChatModel,
)


class LLM(nn.Module):
    def __init__(
        self,
        **cfg,
    ):
        super().__init__()
        for key, value in cfg.items():
            setattr(self, key, value)

        assert "internvl" in self.variant.lower(), (
            f"Variant {self.variant} not supported (only InternVL2 variants are tested)."
        )
        self.model = InternVLChatModel.from_pretrained(self.variant)
        self.model = self.model.language_model
        try:
            self.model.embed_tokens = self.model.base_model.embed_tokens
        except:
            self.model.embed_tokens = self.model.model.tok_embeddings

        if self.lora:
            from peft import LoraConfig, get_peft_model

            peft_config = LoraConfig(
                inference_mode=False,
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                target_modules="all-linear",
            )
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()

        self.vocab_size = self.model.config.vocab_size
        self.hidden_size = self.model.config.hidden_size
        self.max_position_embeddings = self.model.config.max_position_embeddings

    def forward(
        self,
        embeddings: Tensor,
        attention_mask: Tensor = None,
        return_dict: bool = True,
        position_ids: Optional[Tensor] = None,
    ) -> Tensor:

        outputs = self.model(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            output_hidden_states=True,
            position_ids=position_ids,
            return_dict=return_dict,
        )  # .last_hidden_state
        features = outputs.hidden_states[-1]
        logits = outputs[0]

        return features, logits

    def sample_categorical(
        self,
        logits: Tensor,
        temperature: float = 0.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        restrict_tokens: Optional[Tuple[int, int]] = None,
    ):
        if restrict_tokens is not None:
            logits[..., : restrict_tokens[0]] = -float("inf")
            logits[..., restrict_tokens[0] + restrict_tokens[1] :] = -float("inf")

        if temperature <= 0.0:
            return logits.argmax(dim=-1, keepdim=False)

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            pivot = v.select(-1, -1).unsqueeze(-1)
            logits = torch.where(logits < pivot, -float("inf"), logits)

        temperature = max(temperature, 1e-9)
        logits = logits / temperature

        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
            # Shift the indices to the right to keep also the first token above the threshold
            mask = (cumulative_probs > top_p).roll(shifts=1, dims=-1)
            mask[..., 0] = False
            logits[mask.gather(-1, sorted_indices.argsort(-1))] = -float("inf")

        return torch.multinomial(logits.softmax(dim=-1), 1).squeeze(-1)

    def greedy_sample(
        self,
        input_embeds: Tensor,
        inputs_mask: Optional[Tensor] = None,
        max_new_tokens: int = 100,
        temperature: float = 0.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        eos_token_id: Optional[int] = None,
        cache_offset: int = 0,
        input_embed_matrix: Optional[Tensor] = None,
        logit_matrix: Optional[Tensor] = None,
        restrict_tokens: Optional[Tuple[int, int]] = None,
        attention_mask=None,
        position_ids=None,
    ) -> Tuple[Tensor, int]:

        if input_embed_matrix is None:
            if self.embed_tokens is None:
                raise ValueError(
                    "No input embeddings available because the model doesn't define a vocab. "
                    "Please provide input_embed_matrix. "
                )
            input_embed_matrix = self.embed_tokens.weight
        if logit_matrix is None:
            if self.lm_head is None:
                raise ValueError(
                    "No logit matrix available because the model doesn't define a vocab. "
                    "Please provide logit_matrix. "
                )
            logit_matrix = self.lm_head.weight
        # We generate tokens up to the eos token id if provided. If not provided, we generate until the end.
        # If no eos token id is provided, use -1 instead, so we will never stop generating until 'new_token'.
        sampled_tokens = torch.empty(
            (input_embeds.size(0), max_new_tokens), device=input_embeds.device, dtype=torch.long
        )
        if eos_token_id is not None:
            sampled_tokens.fill_(eos_token_id)

        # we start with all sequences left to complete
        incomplete_seq_mask = torch.ones(
            input_embeds.size(0), dtype=torch.bool, device=input_embeds.device
        )
        for i in range(max_new_tokens):
            features, logits = self.forward(
                embeddings=input_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )

            last_hidden_state = features[:, -1]

            # sample the next token
            logits = F.linear(last_hidden_state, logit_matrix)

            next_token = self.sample_categorical(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                restrict_tokens=restrict_tokens,
            )
            x = F.embedding(next_token.unsqueeze(1), input_embed_matrix)

            input_embeds = torch.cat([input_embeds, x], dim=1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones((input_embeds.size(0), 1), device=input_embeds.device)],
                dim=1,
            )

            # only update sequences where we haven't predicted the eos token before
            sampled_tokens[incomplete_seq_mask, i] = next_token[incomplete_seq_mask]
            # For completed sequences, we mask them out for future sampling
            # self.cache_mask[:, cache_offset - 1] &= incomplete_seq_mask

            if eos_token_id is not None:
                # only update the mask of incomplete sequences and stop early if an eos token id is provided.
                incomplete_seq_mask = sampled_tokens[:, i] != eos_token_id
                if not incomplete_seq_mask.any():
                    # finished all sequences, early exit
                    sampled_tokens = sampled_tokens[:, : i + 1]
                    break

        return sampled_tokens, input_embeds


if __name__ == "__main__":
    model = LLM("x-small", False)
    print(model)
