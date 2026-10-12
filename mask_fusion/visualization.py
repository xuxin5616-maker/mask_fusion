"""Render the already prepared inference views without resampling any crops."""

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from .preprocessing import load_mask, _resize_scores


_RESAMPLE = getattr(Image, "Resampling", Image)
_INVALID_RGB = (96, 96, 96)


def restore_rgb_views(images, processor):
    """Undo processor normalization/rescaling on a COPY of the actual input."""
    pixels = images.detach().float().cpu().numpy().copy()
    if pixels.ndim != 4 or pixels.shape[1] != 3:
        raise ValueError("Visualization requires image tensors shaped [V,3,H,W]")
    normalized = getattr(processor, "do_normalize", hasattr(processor, "image_std"))
    if normalized:
        mean = np.asarray(processor.image_mean, dtype=np.float32).reshape(1, 3, 1, 1)
        std = np.asarray(processor.image_std, dtype=np.float32).reshape(1, 3, 1, 1)
        pixels = pixels * std + mean
    if getattr(processor, "do_rescale", hasattr(processor, "rescale_factor")):
        factor = float(processor.rescale_factor)
        if not math.isfinite(factor) or factor <= 0:
            raise ValueError("Cannot invert the processor's rescale_factor")
        pixels = pixels / factor
    pixels = np.rint(np.clip(pixels, 0, 255)).astype(np.uint8)
    return [Image.fromarray(view.transpose(1, 2, 0)) for view in pixels]


def score_heatmap(scores, valid, score_max):
    """Fixed blue/cyan/yellow/red scale, with invalid padding shown in gray."""
    if not math.isfinite(score_max) or score_max <= 0:
        raise ValueError("Visualization score maximum must be finite and positive")
    scaled = np.clip(np.asarray(scores, dtype=np.float32) / score_max, 0, 1)
    anchors = np.array([[0, 0, 128], [0, 160, 255], [255, 220, 0], [200, 0, 0]], dtype=np.float32)
    channels = [np.interp(scaled, [0, 1 / 3, 2 / 3, 1], anchors[:, channel]) for channel in range(3)]
    colors = np.rint(np.stack(channels, axis=-1)).astype(np.uint8)
    colors[~np.asarray(valid, dtype=bool)] = _INVALID_RGB
    return Image.fromarray(colors)


def _draw_boxes(image, metadata):
    """Map actual half-open crop boxes to the original image for annotation."""
    result = image.copy()
    draw = ImageDraw.Draw(result)
    width, height = result.size
    palette = ("red", "lime", "cyan", "magenta", "orange")
    for index, view in enumerate(metadata["views"]):
        if view["kind"] == "base":
            continue
        x0, y0, x1, y1 = view["crop_box"]
        if view["coordinate_space"] == "padded_canvas":
            left, top, _, _ = metadata["padding_ltrb"]
            resized_w, resized_h = metadata["content_resized_size"]
            x0, x1 = (x0 - left) * width / resized_w, (x1 - left) * width / resized_w
            y0, y1 = (y0 - top) * height / resized_h, (y1 - top) * height / resized_h
        elif view["coordinate_space"] != "original_image":
            raise ValueError("Unknown visualization crop coordinate space")
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(width, x1), min(height, y1)
        if x1 <= x0 or y1 <= y0:
            continue  # Entirely padded tiles have no corresponding source area.
        color = palette[(index - 1) % len(palette)]
        draw.rectangle((x0, y0, max(x0, x1 - 1), max(y0, y1 - 1)),
                       outline=color, width=max(1, round(min(width, height) / 180)))
        draw.text((x0 + 3, y0 + 3), f"{view['kind']} {index:02d}", fill=color,
                  stroke_width=1, stroke_fill="black")
    return result


def prediction_display(result):
    """Use the evaluator's labels; never infer abnormality from an A/B letter."""
    display = {key: result.get(key) for key in
               ("gt", "anomaly_prediction", "prediction", "choice", "answer", "correct")}
    status = "CORRECT" if result.get("correct") is True else (
        "INCORRECT" if result.get("correct") is False else "UNSCORED")
    if "anomaly_prediction" in result:
        labels = {0: "NORMAL", 1: "ABNORMAL"}
        truth = labels.get(result.get("gt"), "UNKNOWN")
        predicted = labels.get(result.get("anomaly_prediction"), "UNPARSED")
        if predicted == "UNPARSED":
            status = "UNPARSED"
        display.update({"ground_truth_text": truth, "model_judgment_text": predicted,
                        "headline": f"GT: {truth} | Model: {predicted} | {status}"})
    else:
        # Ordinary QA choices need not mean normal/abnormal.
        display["headline"] = f"Reference: {result.get('answer') or 'UNKNOWN'} | {status}"
    display["status"] = status
    return display


def _banner_font(size):
    for name in ("NotoSansCJK-Regular.ttc", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                 "C:/Windows/Fonts/msyh.ttc", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def _fit_text(draw, text, font, max_width):
    text = " ".join(str(text).split())
    if draw.textlength(text, font=font) <= max_width:
        return text
    while text and draw.textlength(text + "...", font=font) > max_width:
        text = text[:-1]
    return text + "..."


def _overview(rows, destination, score_max, decision=None):
    side, gap, label_height = 256, 12, 30
    row_height = side + label_height + gap
    banner_height = 80 if decision is not None else 0
    sheet = Image.new("RGB", (3 * side + 4 * gap, banner_height + 54 + len(rows) * row_height + 50), "white")
    draw = ImageDraw.Draw(sheet)
    if decision is not None:
        palette = {"CORRECT": ("#e5f5e9", "#146b32"), "INCORRECT": ("#fde8e8", "#aa2020"),
                   "UNPARSED": ("#fff2d5", "#885700"), "UNSCORED": ("#eeeeee", "#444444")}
        background, foreground = palette[decision["status"]]
        draw.rectangle((0, 0, sheet.width, banner_height - 8), fill=background)
        title_font, response_font = _banner_font(20), _banner_font(14)
        draw.text((gap, 10), _fit_text(draw, decision["headline"], title_font, sheet.width - 2 * gap),
                  font=title_font, fill=foreground)
        response = "Model response: " + (decision.get("prediction") or "(empty)")
        draw.text((gap, 43), _fit_text(draw, response, response_font, sheet.width - 2 * gap),
                  font=response_font, fill="black")
    draw.text((gap, banner_height + 8), "Image / selected crops", fill="black")
    draw.text((2 * gap + side, banner_height + 8), "Continuous expert mask", fill="black")
    draw.text((3 * gap + 2 * side, banner_height + 8), "Image + mask overlay", fill="black")
    draw.text((gap, banner_height + 28), "Input rows follow the actual vision-tower order (base first).", fill="black")
    for row_index, (label, panels) in enumerate(rows):
        top = banner_height + 54 + row_index * row_height
        draw.text((gap, top), label, fill="black")
        for column, panel in enumerate(panels):
            preview = ImageOps.contain(panel, (side, side), method=_RESAMPLE.BILINEAR)
            left = gap + column * (side + gap) + (side - preview.width) // 2
            sheet.paste(preview, (left, top + label_height + (side - preview.height) // 2))
    legend_top = sheet.height - 38
    ramp = np.linspace(0, score_max, side, dtype=np.float32)[None, :].repeat(12, axis=0)
    sheet.paste(score_heatmap(ramp, np.ones_like(ramp, dtype=bool), score_max), (gap, legend_top))
    draw.text((gap, legend_top + 16), "0", fill="black")
    maximum_label = f"{score_max:g}"
    draw.text((gap + side - draw.textlength(maximum_label), legend_top + 16), maximum_label, fill="black")
    draw.text((side + 2 * gap, legend_top), "Gray = padding; fixed scale; no per-image min-max.", fill="black")
    sheet.save(destination, quality=92)


def save_input_visualization(sample, record, source, processor, output_dir, index, *, score_max=255.0,
                             prediction_result=None):
    """Save original context and exact model views; never rerun preprocessing.

    PNG/JPEG files are displays. ``mask_values.npz`` retains the unmodified raw
    continuous score tensors (before fusion score_scale) and padding validity.
    A numeric dataset index prevents collisions from duplicate IDs and paths.
    Prediction annotations are added only to the overview, after evaluation;
    the individual input images and score arrays remain unannotated.
    """
    metadata = sample.get("preprocessing_metadata")
    if not metadata or "views" not in metadata:
        raise ValueError("Input visualization requires preprocessing metadata from QADataset")
    rgb_views = restore_rgb_views(sample["images"], processor)
    masks = sample["masks"].detach().float().cpu().numpy()
    valid = sample["mask_valid"].detach().cpu().numpy().astype(bool, copy=False)
    if masks.shape != valid.shape or masks.shape != (len(rgb_views), 1, rgb_views[0].height, rgb_views[0].width):
        raise ValueError("Visualization image/mask shapes do not match")
    if len(metadata["views"]) != len(rgb_views):
        raise ValueError("Visualization metadata/view counts do not match")
    with Image.open(record["image"]) as image_file:
        original = image_file.convert("RGB")
    source_mask = load_mask(record["mask"])
    aligned_mask = _resize_scores(source_mask, original.size)
    source_heatmap = score_heatmap(aligned_mask, np.ones(aligned_mask.shape, dtype=bool), score_max)
    selected_image = _draw_boxes(original, metadata)
    source_overlay = _draw_boxes(Image.blend(original, source_heatmap, 0.4), metadata)

    folder = Path(output_dir).resolve() / f"{index:08d}"
    folder.mkdir(parents=True, exist_ok=True)
    original.save(folder / "original.png")
    selected_image.save(folder / "selected_regions.png")
    source_heatmap.save(folder / "original_mask.png")
    source_overlay.save(folder / "original_overlay.png")
    rows = [("Source image with actual ROI/tile boxes", (selected_image, source_heatmap, source_overlay))]
    saved_views = []
    for view_index, (rgb, view) in enumerate(zip(rgb_views, metadata["views"])):
        heatmap = score_heatmap(masks[view_index, 0], valid[view_index, 0], score_max)
        overlay = Image.blend(rgb, heatmap, 0.4)
        stem = f"view_{view_index:02d}_{view['kind']}"
        filenames = {}
        for name, panel in (("image", rgb), ("mask", heatmap), ("overlay", overlay)):
            filename = f"{stem}_{name}.png"
            panel.save(folder / filename)
            filenames[name] = filename
        row_label = f"Input {view_index:02d}: {view['kind']} ({rgb.width} x {rgb.height})"
        rows.append((row_label, (rgb, heatmap, overlay)))
        selected_scores = masks[view_index, 0][valid[view_index, 0]]
        saved_views.append({
            "index": view_index, **view, "files": filenames,
            "valid_score_min": float(selected_scores.min()) if selected_scores.size else None,
            "valid_score_max": float(selected_scores.max()) if selected_scores.size else None,
            "valid_pixels": int(selected_scores.size),
        })
    np.savez_compressed(folder / "mask_values.npz", masks=masks, mask_valid=valid, source_mask=source_mask)
    decision = prediction_display(prediction_result) if prediction_result is not None else None
    _overview(rows, folder / "overview.jpg", score_max, decision=decision)
    details = {
        "dataset_index": index, "id": record["id"], "category": record.get("category"),
        "image": record["image"], "mask": record["mask"], **source,
        "preprocessing": metadata, "views": saved_views,
        "display": {"score_min": 0, "score_max": score_max, "invalid_rgb": list(_INVALID_RGB),
                    "overlay_alpha": 0.4, "per_image_min_max": False,
                    "rgb": "inverse processor normalization/rescaling of actual input tensor",
                    "scores": "Only displays are clipped; mask_values.npz preserves input scores."},
    }
    if decision is not None:
        details["prediction_result"] = decision
    (folder / "metadata.json").write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"directory": str(folder), "overview": str(folder / "overview.jpg"),
            "metadata": str(folder / "metadata.json"), "mask_values": str(folder / "mask_values.npz")}
