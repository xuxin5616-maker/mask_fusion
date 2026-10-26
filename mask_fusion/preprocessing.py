"""Keep continuous expert scores in the exact coordinate frame of each image view.

RGB pixels are processed by the bundled baseline. Scalar masks remain float32
in their original units. The optional SigLIP mask path colorizes the aligned
scores with fixed-range Jet before applying the same RGB processor once.
"""

from contextlib import contextmanager
from copy import copy
import importlib
import math
from pathlib import Path
import threading

import numpy as np
from PIL import Image


_RNG_LOCK = threading.Lock()
_BILINEAR = getattr(Image, "Resampling", Image).BILINEAR
_NEAREST = getattr(Image, "Resampling", Image).NEAREST
_MODES = {"square", "anyres_max_9", "randomroi", "randompatch"}


def _make_jet_rgb_lut():
    # Matplotlib's standard 256-entry jet, gamma=1, RGB byte output. These are
    # its channel breakpoints, not OpenCV's BGR map or a generic rainbow:
    # https://github.com/matplotlib/matplotlib/blob/v3.10.8/lib/matplotlib/_cm.py#L225-L240
    # Sample/interpolate on the 0..255 grid as LinearSegmentedColormap does;
    # byte conversion truncates (rather than rounds) channel * 255.
    channels = (
        ((0.00, 0.0), (0.35, 0.0), (0.66, 1.0), (0.89, 1.0), (1.00, 0.5)),
        ((0.000, 0.0), (0.125, 0.0), (0.375, 1.0), (0.640, 1.0), (0.910, 0.0), (1.000, 0.0)),
        ((0.00, 0.5), (0.11, 1.0), (0.34, 1.0), (0.65, 0.0), (1.00, 0.0)),
    )
    samples = np.linspace(0.0, 1.0, 256) * 255
    colors = []
    for channel in channels:
        points = np.asarray(channel, dtype=np.float64)
        positions, values = points[:, 0] * 255, points[:, 1]
        right = np.searchsorted(positions, samples).clip(1, len(points) - 1)
        fraction = (samples - positions[right - 1]) / (positions[right] - positions[right - 1])
        colors.append(fraction * (values[right] - values[right - 1]) + values[right - 1])
    table = (np.clip(np.stack(colors, axis=-1), 0.0, 1.0) * 255).astype(np.uint8)
    table.setflags(write=False)
    return table


_JET_RGB_LUT = _make_jet_rgb_lut()


def scores_to_jet_rgb(scores, valid=None, score_scale=255.0):
    """Return an HxWx3 uint8 RGB array using Matplotlib's standard 256-bin Jet.

    Scores use the fixed range [0, score_scale], never per-image min/max. Float
    inputs index the LUT with floor(256 * score / score_scale), with the upper
    endpoint assigned to entry 255, matching Matplotlib colormap semantics.
    ``valid`` is an optional same-shaped boolean array. Only valid pixels must
    be finite and within range; all invalid pixels become neutral (128,128,128),
    including when their scores are NaN or infinity. Input arrays are unchanged.
    """
    if (isinstance(score_scale, (bool, np.bool_))
            or not isinstance(score_scale, (int, float, np.integer, np.floating))
            or not math.isfinite(float(score_scale)) or score_scale <= 0):
        raise ValueError("score_scale must be a finite positive number")
    scores = np.asarray(scores)
    if scores.ndim != 2 or min(scores.shape, default=0) <= 0:
        raise ValueError("Jet scores must be a nonempty 2D scalar score map")
    if (not (np.issubdtype(scores.dtype, np.number) or scores.dtype == np.bool_)
            or np.iscomplexobj(scores)):
        raise ValueError("Jet scores must contain real numeric values")
    if valid is None:
        valid = np.ones(scores.shape, dtype=bool)
    else:
        valid = np.asarray(valid)
        if valid.shape != scores.shape or valid.dtype != np.bool_:
            raise ValueError("Jet valid mask must be boolean and have the same HW shape as scores")
    values = scores[valid].astype(np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Valid Jet scores must be finite")
    if ((values < 0) | (values > float(score_scale))).any():
        raise ValueError(f"Valid Jet scores must lie in [0, {float(score_scale):g}]")
    rgb = np.full((*scores.shape, 3), 128, dtype=np.uint8)
    indices = np.minimum((values / float(score_scale) * 256).astype(np.int64), 255)
    rgb[valid] = _JET_RGB_LUT[indices]
    return rgb


def _validate_mask(mask):
    array = np.asarray(mask)
    if array.ndim != 2 or min(array.shape, default=0) <= 0:
        raise ValueError("Expert mask must be a nonempty 2D scalar score map, not an RGB heatmap")
    if not (np.issubdtype(array.dtype, np.number) or array.dtype == np.bool_):
        raise ValueError("Expert mask must contain numeric scalar scores")
    if np.iscomplexobj(array) or not np.isfinite(array).all():
        raise ValueError("Expert mask must contain finite real scores (no NaN or infinity)")
    if (array < 0).any():
        raise ValueError("Expert mask scores must be nonnegative")
    with np.errstate(over="ignore", invalid="ignore"):
        result = array.astype(np.float32, copy=True)
    if not np.isfinite(result).all():
        raise ValueError("Expert mask scores exceed the finite float32 range")
    return result


def load_mask(path):
    """Read a 2D PNG/TIFF/NPY/NPZ score map without quantization or rescaling.

    NPZ files must contain ``anomaly_map``. RGB/RGBA and palette images are
    rejected even if their channels happen to be equal: color heatmaps are not
    interchangeable with the original scalar anomaly scores. Scores above one
    are accepted; the fusion configuration owns ``score_scale``.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        array = np.load(path, allow_pickle=False)
    elif suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if "anomaly_map" not in archive:
                raise ValueError("NPZ expert mask must contain an 'anomaly_map' array")
            array = archive["anomaly_map"]
    elif suffix in {".png", ".tif", ".tiff"}:
        with Image.open(path) as source:
            if source.mode not in {"1", "L", "I", "F", "I;16", "I;16L", "I;16B", "I;16N"}:
                raise ValueError("Expert mask image must be grayscale, not an RGB/palette heatmap")
            if getattr(source, "n_frames", 1) != 1:
                raise ValueError("Expert mask must be a single image, not a multipage TIFF")
            array = np.array(source)
    else:
        raise ValueError("Supported expert mask formats: PNG, TIF/TIFF, NPY, NPZ")
    return _validate_mask(array)


def _load_baseline():
    # Do not import transformers, OpenCV, or the language model at module import.
    from .loader import activate_vendor

    activate_vendor()
    return importlib.import_module("llava.mm_utils")


@contextmanager
def _baseline_seed(seed):
    """Make the baseline's legacy NumPy RNG reproducible without retaining state."""
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer in [0, 2**32)")
    if not 0 <= int(seed) < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    with _RNG_LOCK:
        state = np.random.get_state()
        try:
            np.random.seed(int(seed))
            yield
        finally:
            np.random.set_state(state)


def _resize_scores(array, size):
    """PIL mode F preserves continuous values and uses the RGB resize geometry."""
    if (array.shape[1], array.shape[0]) == tuple(size):
        return array.copy()
    return np.array(Image.fromarray(array).resize(tuple(size), resample=_BILINEAR), dtype=np.float32)


def _resize_valid(array, size):
    # Validity follows source-pixel centers; it is never interpolated into scores.
    return np.array(
        Image.fromarray(array.astype(np.uint8)).resize(tuple(size), resample=_NEAREST),
        dtype=bool,
    )


def _validate_boxes(boxes):
    if boxes is None:
        return None
    result = []
    for box in boxes:
        if len(box) != 4:
            raise ValueError("Each box must contain four inclusive xyxy coordinates")
        values = np.asarray(box)
        if not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values):
            raise ValueError("Box coordinates must be finite integers")
        if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
            raise ValueError("Box coordinates must be finite integers")
        x0, y0, x1, y1 = [int(value) for value in values]
        if x1 < x0 or y1 < y0:
            raise ValueError("Inclusive box endpoints must satisfy xmax >= xmin, ymax >= ymin")
        result.append([x0, y0, x1, y1])
    if len(result) > 4:
        raise ValueError("The bundled randomroi baseline supports at most four ROI boxes")
    return result


def _unpack_pixels(output):
    import torch

    if isinstance(output, tuple):
        pixels, all_metadata = output
        metadata = all_metadata[0] if all_metadata else {}
    else:
        # The baseline's square branch returns early without metadata.
        pixels, metadata = output, {}
    if isinstance(pixels, list):
        if len(pixels) != 1:
            raise ValueError("Expected baseline views for exactly one source image")
        pixels = pixels[0]
    if not isinstance(pixels, torch.Tensor):
        raise ValueError("Baseline processor did not return a tensor")
    if pixels.ndim == 5 and pixels.shape[0] == 1:
        pixels = pixels[0]
    if pixels.ndim == 3:
        pixels = pixels.unsqueeze(0)
    if pixels.ndim != 4 or pixels.shape[1] != 3 or pixels.shape[-2] != pixels.shape[-1]:
        raise ValueError("Expected square image views with shape [V, 3, S, S]")
    if pixels.shape[0] == 0 or pixels.shape[-1] == 0:
        raise ValueError("Baseline processor returned empty image views")
    return pixels, metadata


def prepare_image_mask(image, mask, processor, model_config, *, mode="anyres_max_9", boxes=None, seed=0):
    """Create matched RGB/score views using the actual baseline crop selection.

    Returns ``images[V,3,S,S]``, float32 ``masks[V,1,S,S]``, boolean
    ``mask_valid[V,1,S,S]``, original ``image_size=(width,height)``, and JSON-safe
    ``metadata``. Invalid padded score pixels are zero. Mask values are NOT
    divided by 255, normalized, clipped, or binarized.

    When ``model_config.mask_fusion['mask_encoder_type'] == 'siglip_jet'``,
    also returns floating ``mask_images[V,3,S,S]``. Each already-aligned score
    view is colorized with fixed-range Jet and processed once by ``processor``;
    no crops are selected again. Invalid pixels are neutral RGB 128 before the
    processor's ordinary rescale/normalization. The scalar masks stay unchanged.

    ``boxes`` use the baseline's INCLUSIVE original-image ``[x0,y0,x1,y1]``
    convention, with at most four boxes. Metadata records the actual HALF-OPEN
    PIL crop boxes, including ``x1+1,y1+1``. Out-of-image crop pixels are invalid.
    ``randompatch`` calls the baseline ``randomroi`` with no selection mask;
    ``randomroi`` passes a separate score copy for baseline ROI selection. The
    selected boxes are reused, never sampled a second time for the score path.

    A differently sized mask is bilinearly resized onto the original image
    frame, recorded explicitly in metadata. This assumes the same full-image
    field of view; shape checks cannot establish registration of unrelated maps.
    The processor must use resize-only geometry, as bundled SigLIP does.
    """
    import torch

    if not isinstance(image, Image.Image) or min(image.size) <= 0:
        raise ValueError("image must be a nonempty PIL image")
    if mode not in _MODES:
        raise ValueError(f"Unsupported mode {mode!r}; expected one of {sorted(_MODES)}")
    if getattr(processor, "do_center_crop", False):
        raise ValueError("Mask alignment requires a resize-only processor, not center cropping")
    if boxes is not None and mode not in {"randomroi", "randompatch"}:
        raise ValueError("Explicit boxes are only supported for randomroi/randompatch")
    boxes = _validate_boxes(boxes)
    scores = _validate_mask(mask)
    original_mask_size = [int(scores.shape[1]), int(scores.shape[0])]
    image_size = tuple(int(value) for value in image.size)
    scores = _resize_scores(scores, image_size)
    valid = np.ones(scores.shape, dtype=bool)
    image = image.convert("RGB")

    baseline_mode = "randomroi" if mode == "randompatch" else mode
    config = copy(model_config)
    config.image_aspect_ratio = baseline_mode
    baseline = _load_baseline()
    selection_mask = scores.copy() if mode == "randomroi" else None
    with _baseline_seed(seed):
        output = baseline.process_images(
            [image], processor, config,
            masks=[selection_mask],
            boxes_list=[boxes] if boxes is not None else None,
            return_pro_data=True,
        )
    images, baseline_metadata = _unpack_pixels(output)
    side = int(images.shape[-1])
    output_size = (side, side)
    score_views = [_resize_scores(scores, output_size)]
    valid_views = [np.ones((side, side), dtype=bool)]
    views = [{
        "kind": "base", "coordinate_space": "original_image",
        "crop_box": [0, 0, *image_size], "resized_size": [side, side],
    }]
    metadata = {
        "mode": mode, "baseline_mode": baseline_mode, "seed": int(seed),
        "original_size": list(image_size), "mask_original_size": original_mask_size,
        "mask_resized_to_image": original_mask_size != list(image_size),
        "score_interpolation": "bilinear", "score_units": "raw",
        "validity_interpolation": "nearest", "processed_size": [side, side],
        "box_input_convention": "inclusive_xyxy", "crop_box_convention": "half_open_xyxy",
        "image_token_order": getattr(config, "image_token_order", "base_first"),
        "views": views,
    }

    if mode == "anyres_max_9":
        grid = baseline_metadata.get("grid_patches")
        if grid is None or len(grid) != 2 or min(grid) <= 0:
            raise ValueError("Baseline AnyRes metadata did not provide a valid tile grid")
        grid_w, grid_h = [int(value) for value in grid]
        tile_w = int(processor.crop_size["width"])
        tile_h = int(processor.crop_size["height"])
        if tile_w != tile_h:
            raise ValueError("Bundled AnyRes requires square source tiles")
        canvas_w, canvas_h = grid_w * tile_w, grid_h * tile_h
        width, height = image_size
        scale_w, scale_h = canvas_w / width, canvas_h / height
        if scale_w < scale_h:
            resized_w, resized_h = canvas_w, min(math.ceil(height * scale_w), canvas_h)
        else:
            resized_h, resized_w = canvas_h, min(math.ceil(width * scale_h), canvas_w)
        left, top = (canvas_w - resized_w) // 2, (canvas_h - resized_h) // 2
        canvas = np.zeros((canvas_h, canvas_w), dtype=np.float32)
        canvas_valid = np.zeros((canvas_h, canvas_w), dtype=bool)
        canvas[top:top + resized_h, left:left + resized_w] = _resize_scores(scores, (resized_w, resized_h))
        canvas_valid[top:top + resized_h, left:left + resized_w] = True
        padding = [left, top, canvas_w - resized_w - left, canvas_h - resized_h - top]
        metadata.update({
            "grid_patches": [grid_w, grid_h], "canvas_size": [canvas_w, canvas_h],
            "content_resized_size": [resized_w, resized_h], "padding_ltrb": padding,
            "view_order": "base_then_row_major_tiles",
        })
        for row in range(grid_h):
            for column in range(grid_w):
                x, y = column * tile_w, row * tile_h
                score_views.append(_resize_scores(canvas[y:y + tile_h, x:x + tile_w], output_size))
                valid_views.append(_resize_valid(canvas_valid[y:y + tile_h, x:x + tile_w], output_size))
                views.append({
                    "kind": "tile", "coordinate_space": "padded_canvas", "grid_rc": [row, column],
                    "crop_box": [x, y, x + tile_w, y + tile_h], "resized_size": [side, side],
                })
    elif mode in {"randomroi", "randompatch"}:
        crop_boxes = baseline_metadata.get("roi_boxes")
        if crop_boxes is None:
            raise ValueError("Baseline ROI metadata did not provide the actual crop boxes")
        # ROI pixels use crop -> resize(384) -> processor resize. Match both stages.
        roi_intermediate_size = (384, 384)
        score_image = Image.fromarray(scores)
        valid_image = Image.fromarray(valid.astype(np.uint8))
        clean_boxes = []
        for box in crop_boxes:
            crop_box = [int(value) for value in box]
            clean_boxes.append(crop_box)
            cropped = np.array(score_image.crop(tuple(crop_box)), dtype=np.float32)
            cropped_valid = np.array(valid_image.crop(tuple(crop_box)), dtype=bool)
            score_views.append(_resize_scores(_resize_scores(cropped, roi_intermediate_size), output_size))
            valid_views.append(_resize_valid(_resize_valid(cropped_valid, roi_intermediate_size), output_size))
            views.append({
                "kind": "roi", "coordinate_space": "original_image", "crop_box": crop_box,
                "intermediate_size": list(roi_intermediate_size), "resized_size": [side, side],
            })
        metadata.update({"roi_boxes": clean_boxes, "view_order": "base_then_selected_rois"})
    else:
        metadata["view_order"] = "base_only"

    if len(score_views) != images.shape[0]:
        raise ValueError(f"Image/mask crop count mismatch: {images.shape[0]} versus {len(score_views)}")
    mask_array = np.stack(score_views).astype(np.float32, copy=False)
    valid_array = np.stack(valid_views).astype(bool, copy=False)
    mask_array[~valid_array] = 0
    if not np.isfinite(mask_array).all() or (mask_array < 0).any():
        raise ValueError("Mask preprocessing produced invalid scores")
    result = {
        "images": images,
        "masks": torch.from_numpy(mask_array[:, None]),
        "mask_valid": torch.from_numpy(valid_array[:, None]),
        "image_size": image_size,
        "metadata": metadata,
    }
    fusion_config = getattr(model_config, "mask_fusion", None)
    if isinstance(fusion_config, dict) and fusion_config.get("mask_encoder_type") == "siglip_jet":
        score_scale = fusion_config.get("score_scale", 255.0)
        jet_views = [Image.fromarray(scores_to_jet_rgb(scores, valid, score_scale))
                     for scores, valid in zip(mask_array, valid_array)]
        mask_images = processor.preprocess(jet_views, return_tensors="pt")["pixel_values"]
        if (not isinstance(mask_images, torch.Tensor) or mask_images.shape != images.shape
                or not mask_images.is_floating_point() or not torch.isfinite(mask_images).all()):
            raise ValueError("Jet processor must return finite floating mask_images with the same [V,3,S,S] shape as images")
        result["mask_images"] = mask_images
        metadata["mask_image_encoding"] = {
            "colormap": "matplotlib_jet_256", "color_order": "RGB",
            "score_range": [0.0, float(score_scale)], "padding_rgb": [128, 128, 128],
            "colorize_after_geometry": True, "processor_passes_per_view": 1,
        }
    return result
