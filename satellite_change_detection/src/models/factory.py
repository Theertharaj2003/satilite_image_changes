from __future__ import annotations

from typing import Any, Mapping

from .change_detection import SiameseCNNChangeDetector, TemporalTransformerChangeDetector, ProposedTemporalTransformer


def build_model(model_cfg: Mapping[str, Any]):
    model_type = str(model_cfg.get("type", "proposed")).lower()
    channels = tuple(model_cfg.get("channels", [32, 64, 128, 256]))
    heads = int(model_cfg.get("heads", 8))
    layers = int(model_cfg.get("layers", 2))
    dropout = float(model_cfg.get("dropout", 0.1))
    if model_type in {"cnn", "siamese_cnn"}:
        return SiameseCNNChangeDetector(channels=channels)
    if model_type in {"transformer", "temporal_transformer"}:
        return TemporalTransformerChangeDetector(channels=channels, heads=heads, layers=layers, dropout=dropout)
    if model_type in {"proposed", "temporal_cross_attention"}:
        return ProposedTemporalTransformer(
            channels=channels,
            heads=heads,
            layers=layers,
            dropout=dropout,
            use_temporal_self_attention=bool(model_cfg.get("use_temporal_self_attention", True)),
            use_cross_attention=bool(model_cfg.get("use_cross_attention", True)),
            use_multiscale=bool(model_cfg.get("use_multiscale", True)),
        )
    raise ValueError(f"Unknown model type: {model_type}")
