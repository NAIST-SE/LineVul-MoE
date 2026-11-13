from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from transformers import AutoConfig, RobertaModel


class ClassificationHead(nn.Module):
    def __init__(self, hidden_size: int, dropout_probability: float, outputs: int):
        super().__init__()
        self.dropout = nn.Dropout(dropout_probability)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, outputs)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        value = hidden_states[:, 0, :]
        value = self.dropout(value)
        value = torch.tanh(self.dense(value))
        value = self.dropout(value)
        return self.output(value)


class GeneralistModel(nn.Module):
    def __init__(self, model_path: Path, outputs: int = 1):
        super().__init__()
        config = AutoConfig.from_pretrained(
            str(model_path),
            local_files_only=True,
        )
        self.encoder = RobertaModel.from_pretrained(
            str(model_path),
            config=config,
            local_files_only=True,
        )
        self.classifier = ClassificationHead(
            config.hidden_size,
            config.hidden_dropout_prob,
            outputs,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        hidden_states = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state
        return {"logits": self.classifier(hidden_states)}


class BottleneckExpert(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        bottleneck_size: int,
        dropout_probability: float,
    ):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.down = nn.Linear(hidden_size, bottleneck_size)
        self.activation = nn.GELU()
        self.up = nn.Linear(bottleneck_size, hidden_size)
        self.dropout = nn.Dropout(dropout_probability)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        residual = self.up(self.activation(self.down(self.layer_norm(pooled))))
        adapted = pooled + self.dropout(residual)
        return self.output(adapted).squeeze(-1)


class SharedMoEModel(nn.Module):
    def __init__(
        self,
        model_path: Path,
        experts: int = 5,
        router_classes: int = 6,
        adapter_size: int = 64,
    ):
        super().__init__()
        config = AutoConfig.from_pretrained(
            str(model_path),
            local_files_only=True,
        )
        self.encoder = RobertaModel.from_pretrained(
            str(model_path),
            config=config,
            local_files_only=True,
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.generalist = ClassificationHead(
            config.hidden_size,
            config.hidden_dropout_prob,
            1,
        )
        self.experts = nn.ModuleList(
            [
                BottleneckExpert(
                    config.hidden_size,
                    adapter_size,
                    config.hidden_dropout_prob,
                )
                for _ in range(experts)
            ]
        )
        self.router = nn.Linear(config.hidden_size, router_classes)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        hidden_states = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state
        pooled = self.dropout(hidden_states[:, 0, :])
        return {
            "generalist_logits": self.generalist(hidden_states).squeeze(-1),
            "expert_logits": torch.stack(
                [expert(pooled) for expert in self.experts],
                dim=1,
            ),
            "router_logits": self.router(pooled),
        }


def build_model(
    model_kind: str,
    model_path: Path,
    adapter_size: int,
) -> nn.Module:
    if model_kind in {"generalist", "specialist"}:
        return GeneralistModel(model_path, outputs=1)
    if model_kind == "multiclass":
        return GeneralistModel(model_path, outputs=7)
    if model_kind == "sme":
        return SharedMoEModel(
            model_path,
            experts=5,
            router_classes=6,
            adapter_size=adapter_size,
        )
    raise ValueError(f"Unsupported model kind: {model_kind}")


def load_warm_start(model: nn.Module, checkpoint_path: Path) -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    original_state = checkpoint.get("model_state", checkpoint)
    state = dict(original_state)
    mapped_generalist_tensors = 0
    if hasattr(model, "generalist"):
        for key, value in original_state.items():
            if key.startswith("classifier."):
                mapped_key = "generalist." + key.removeprefix("classifier.")
                state[mapped_key] = value
                mapped_generalist_tensors += 1
    missing, unexpected = model.load_state_dict(state, strict=False)
    loaded_encoder = sum(
        1 for key in state if key.startswith("encoder.") and key not in unexpected
    )
    return {
        "checkpoint": str(checkpoint_path),
        "loaded_encoder_tensors": loaded_encoder,
        "mapped_generalist_tensors": mapped_generalist_tensors,
        "missing": list(missing),
        "unexpected": list(unexpected),
    }


def parameter_counts(model: nn.Module) -> dict[str, int]:
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
    }
