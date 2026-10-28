"""Evaluate replacing ONLY the direct expert features before the projector.

All ordinary mask_fusion.infer arguments are forwarded unchanged. The mask,
its ROI selection, and the mask-guided attention still use the real input.
Example: python -m mask_fusion.infer_expert_ablation --expert-input zero \
    --adapter outputs/siglip_jet_v1/sft --data test.jsonl --output zero.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from unittest.mock import patch


def validate_direct_path(config):
    if not config.get("direct_to_projector", True):
        raise ValueError("This adapter has no direct expert input to the projector (attention-only architecture)")
    if config.get("mode", "both") not in {"both", "direct"}:
        raise ValueError("The saved adapter must have an enabled direct expert path: mode='both' or 'direct'")


class ExpertInputAblation:
    """Process-local forward hook at the output of the fusion module.

    The output is [updated image features; adapted expert features]. Replacing
    its tail here affects every base/ROI view before projector and pooling,
    without changing the attention inputs, image prefix, or saved weights.
    """

    def __init__(self, model, mode, *, seed=0, mean=0.0, std=1.0):
        if mode not in {"original", "zero", "random"}:
            raise ValueError("expert input must be original, zero, or random")
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
            raise ValueError("noise seed must be an integer in [0, 2**63)")
        if not math.isfinite(mean) or not math.isfinite(std) or std <= 0:
            raise ValueError("noise mean must be finite and noise std must be finite and positive")
        core = model.get_model()
        fusion = core.mask_fusion
        validate_direct_path(fusion.config.to_dict())
        if getattr(model.config, "mask_fusion_pretrain", False):
            raise ValueError("Use a normal SFT/inference model, not the mask-only pretraining runtime")
        self.visual_dim = fusion.vision_dim
        self.expert_dim = fusion.expert_feature_dim
        self.patch_size = fusion.patch_size
        self.mode, self.seed, self.mean, self.std = mode, seed, float(mean), float(std)
        self.calls = self.views = self.valid_patches = 0
        self.generators = {}
        self.handle = fusion.register_forward_hook(self._replace, with_kwargs=True)

    def _replace(self, module, args, kwargs, output):
        import torch
        import torch.nn.functional as F

        if output.ndim != 3 or output.shape[-1] != self.visual_dim + self.expert_dim:
            raise ValueError("Fusion output does not contain the expected visual/expert channel split")
        valid = kwargs.get("valid_tokens")
        if valid is None:
            pixels = kwargs.get("mask_valid", args[2] if len(args) > 2 else None)
            if pixels is None:
                valid = torch.ones(output.shape[:2], device=output.device, dtype=torch.bool)
            else:
                if pixels.ndim == 3:
                    pixels = pixels.unsqueeze(1)
                valid = F.max_pool2d(pixels.float(), self.patch_size, self.patch_size).flatten(1).bool()
        if valid.shape != output.shape[:2]:
            raise ValueError("Expert validity grid differs from the fusion output")
        valid = valid.to(device=output.device, dtype=torch.bool)
        self.calls += 1
        self.views += output.shape[0]
        self.valid_patches += int(valid.sum().item())
        if self.mode == "original":
            return output
        expert = output[..., self.visual_dim:]
        if self.mode == "zero":
            replacement = torch.zeros_like(expert)
        else:
            device_key = str(expert.device)
            if device_key not in self.generators:
                generator = torch.Generator(device=expert.device)
                generator.manual_seed(self.seed)
                self.generators[device_key] = generator
            # An independent generator leaves inference/ROI RNG state unchanged.
            replacement = torch.randn(
                expert.shape, dtype=torch.float32, device=expert.device,
                generator=self.generators[device_key],
            ).mul_(self.std).add_(self.mean).to(dtype=expert.dtype)
            replacement = replacement.masked_fill(~valid.unsqueeze(-1), 0.0)
        return torch.cat((output[..., :self.visual_dim], replacement), dim=-1)

    def close(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def report(self):
        return {
            "target": "fusion output expert channels, after direct MLP and before projector",
            "expert_input": self.mode,
            "visual_channels": self.visual_dim, "expert_channels": self.expert_dim,
            "fusion_forward_calls": self.calls, "encoded_views": self.views,
            "valid_patch_rows": self.valid_patches,
            "raw_mask_unchanged": True, "roi_selection_unchanged": True,
            "attention_unchanged": True, "invalid_expert_patches_zero": True,
            "noise": ({"distribution": "normal", "mean": self.mean, "std": self.std,
                       "seed": self.seed, "independent_rng": True}
                      if self.mode == "random" else None),
        }


def summarize_metrics(paths, output):
    """Write a comparison without merging checkpoints or averaging seeds."""
    if not paths:
        raise ValueError("At least one metrics file is required")
    rows = []
    for filename in paths:
        path = Path(filename).resolve()
        metrics = json.loads(path.read_text(encoding="utf-8"))
        ablation = metrics.get("expert_ablation")
        if not ablation:
            raise ValueError(f"Not an expert-ablation metrics file: {path}")
        rows.append({
            "model": ablation["label"], "expert_input": ablation["expert_input"],
            "noise_seed": (ablation["noise"] or {}).get("seed"),
            **{key: metrics.get(key) for key in (
                "qa_records", "accuracy", "positive_recall", "negative_recall",
                "balanced_accuracy", "tp", "tn", "fp", "fn", "unparseable_anomaly",
                "dtype", "attention", "seed", "anomaly_prompt", "adapter",
            )},
            "metrics_file": str(path),
        })
    output = Path(output)
    destinations = (output.with_suffix(".json"), output.with_suffix(".csv"))
    if any(path.exists() for path in destinations):
        raise FileExistsError("Comparison already exists; use another output directory or summary name")
    output.parent.mkdir(parents=True, exist_ok=True)
    destinations[0].write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    with destinations[1].open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("\nmodel\texpert_input\taccuracy\tabnormal_recall\tnormal_recall\tFP\tFN")
    for row in rows:
        percentages = [f"{100 * row[key]:.2f}%" if row[key] is not None else "N/A"
                       for key in ("accuracy", "positive_recall", "negative_recall")]
        print("\t".join(map(str, [row["model"], row["expert_input"], *percentages, row["fp"], row["fn"]])))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
        epilog="Other arguments, including --adapter/--data/--output/--dtype, are forwarded to mask_fusion.infer.")
    parser.add_argument("--expert-input", choices=("original", "zero", "random"))
    parser.add_argument("--ablation-label", default="model")
    parser.add_argument("--noise-seed", type=int, default=0)
    parser.add_argument("--noise-mean", type=float, default=0.0)
    parser.add_argument("--noise-std", type=float, default=1.0)
    parser.add_argument("--summarize", nargs="+", metavar="METRICS_JSON")
    parser.add_argument("--summary-output", default="results/expert_ablation_comparison")
    arguments, infer_arguments = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    if arguments.summarize:
        if infer_arguments or arguments.expert_input:
            parser.error("--summarize cannot be combined with inference arguments")
        return summarize_metrics(arguments.summarize, arguments.summary_output)
    if not arguments.expert_input:
        parser.error("--expert-input is required for inference")
    if not 0 <= arguments.noise_seed < 2**63:
        parser.error("--noise-seed must be in [0, 2**63)")
    if not math.isfinite(arguments.noise_mean) or not math.isfinite(arguments.noise_std) or arguments.noise_std <= 0:
        parser.error("--noise-mean must be finite and --noise-std must be finite and positive")
    probe = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    for flag in ("--adapter", "--data", "--output"):
        probe.add_argument(flag, required=True)
    paths, _ = probe.parse_known_args(infer_arguments)

    from . import infer
    from .model import read_adapter_config

    metadata = read_adapter_config(paths.adapter)
    validate_direct_path(metadata["fusion_config"])
    if metadata.get("training_stage") == "mask_pretrain":
        raise ValueError("Select the SFT adapter for this real-image ablation, not the mask-pretrain adapter")
    original_loader = infer.load_model
    hooks = []

    def load_with_hook(*args, **kwargs):
        loaded = original_loader(*args, **kwargs)
        hooks.append(ExpertInputAblation(
            loaded[0], arguments.expert_input, seed=arguments.noise_seed,
            mean=arguments.noise_mean, std=arguments.noise_std,
        ))
        return loaded

    try:
        # Patch only this process while reusing the exact ordinary evaluation,
        # prompts, dataset ordering, scoring and output handling.
        with patch.object(infer, "load_model", new=load_with_hook):
            infer.main(infer_arguments)
    finally:
        for hook in hooks:
            hook.close()
    if len(hooks) != 1:
        raise RuntimeError("Expected exactly one model load for this ablation")
    metrics_path = Path(paths.output).with_suffix(".metrics.json")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if hooks[0].calls != metrics["qa_records"]:
        raise RuntimeError("Fusion was not called exactly once per evaluated QA record; verify the model path")
    metrics["expert_ablation"] = {
        **hooks[0].report(), "label": arguments.ablation_label,
        "data": str(Path(paths.data).resolve()),
    }
    metrics["fusion_config"] = metadata["fusion_config"]
    temporary = metrics_path.with_name(metrics_path.name + ".ablation.tmp")
    temporary.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(metrics_path)
    print(json.dumps({"event": "expert_ablation_complete", "metrics": str(metrics_path),
                      "expert_ablation": metrics["expert_ablation"]}, ensure_ascii=False))
    return metrics


if __name__ == "__main__":
    main()
