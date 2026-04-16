import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from simple_lama_inpainting import SimpleLama


MODEL = SimpleLama()


def read_image(path: Path, flags: int) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, flags)
    if image is None:
        raise RuntimeError(f"无法读取文件: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> str:
    extension = path.suffix.lower()
    params: list[int] = []

    if extension in {".jpg", ".jpeg"}:
        mime_type = "image/jpeg"
        params = [int(cv2.IMWRITE_JPEG_QUALITY), 100]
    elif extension == ".png":
        mime_type = "image/png"
        params = [int(cv2.IMWRITE_PNG_COMPRESSION), 1]
    elif extension == ".webp":
        mime_type = "image/webp"
        params = [int(cv2.IMWRITE_WEBP_QUALITY), 100]
    elif extension == ".bmp":
        mime_type = "image/bmp"
    else:
        path = path.with_suffix(".png")
        extension = ".png"
        mime_type = "image/png"
        params = [int(cv2.IMWRITE_PNG_COMPRESSION), 1]

    success, encoded = cv2.imencode(extension, image, params)
    if not success:
        raise RuntimeError("图像编码失败")

    encoded.tofile(str(path))
    return mime_type


def parse_mode(raw: str | None) -> str:
    return "brush" if raw == "brush" else "rect"


def parse_strength(raw: str | None) -> int:
    if raw is None or raw == "":
        return 55

    try:
        value = int(raw)
    except ValueError as error:
        raise RuntimeError("强度参数错误") from error

    return max(1, min(100, value))


def split_source(source: np.ndarray) -> tuple[np.ndarray, str, np.ndarray | None]:
    if len(source.shape) == 2:
        return cv2.cvtColor(source, cv2.COLOR_GRAY2BGR), "gray", None

    if source.shape[2] == 4:
        return source[:, :, :3], "bgra", source[:, :, 3]

    return source.copy(), "bgr", None


def merge_source(result_bgr: np.ndarray, kind: str, alpha_channel: np.ndarray | None) -> np.ndarray:
    if kind == "gray":
        return cv2.cvtColor(result_bgr, cv2.COLOR_BGR2GRAY)

    if kind == "bgra" and alpha_channel is not None:
        return np.dstack((result_bgr, alpha_channel))

    return result_bgr


def ellipse_kernel(radius: int) -> np.ndarray:
    size = max(3, radius * 2 + 1)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def expand_mask(binary_mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return binary_mask.copy()
    return cv2.dilate(binary_mask, ellipse_kernel(radius), iterations=1)


def erode_mask(binary_mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return binary_mask.copy()
    return cv2.erode(binary_mask, ellipse_kernel(radius), iterations=1)


def build_masks(mask_image: np.ndarray, target_shape: tuple[int, int], strength: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(mask_image.shape) == 2:
        grayscale = mask_image
    elif mask_image.shape[2] == 4:
        grayscale = cv2.cvtColor(mask_image, cv2.COLOR_BGRA2GRAY)
    else:
        grayscale = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)

    if grayscale.shape[:2] != target_shape:
        grayscale = cv2.resize(grayscale, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)

    _, binary = cv2.threshold(grayscale, 1, 255, cv2.THRESH_BINARY)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, ellipse_kernel(2), iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, ellipse_kernel(1), iterations=1)

    clear_expand = 7 + int(round(strength / 13.0))

    clear_mask = expand_mask(binary, clear_expand)
    blend_mask = expand_mask(clear_mask, 6)
    return binary, clear_mask, blend_mask


def mask_bbox(binary_mask: np.ndarray) -> tuple[int, int, int, int]:
    points = cv2.findNonZero(binary_mask)
    if points is None:
        raise RuntimeError("蒙版为空")
    x, y, w, h = cv2.boundingRect(points)
    return x, y, x + w, y + h


def crop_to_bbox(image: np.ndarray, bbox: tuple[int, int, int, int], margin: int) -> tuple[np.ndarray, tuple[slice, slice]]:
    x0, y0, x1, y1 = bbox
    height, width = image.shape[:2]
    left = max(0, x0 - margin)
    top = max(0, y0 - margin)
    right = min(width, x1 + margin)
    bottom = min(height, y1 + margin)
    y_slice = slice(top, bottom)
    x_slice = slice(left, right)
    return image[y_slice, x_slice].copy(), (y_slice, x_slice)


def resolve_roi_margin(strength: int) -> int:
    return 56 + int(round(strength * 0.55))


def blend_alpha(clear_mask: np.ndarray, blend_mask: np.ndarray) -> np.ndarray:
    alpha = cv2.GaussianBlur((blend_mask > 0).astype(np.float32), (0, 0), 1.8)
    alpha = np.clip(alpha, 0.0, 1.0)
    alpha[clear_mask > 0] = 1.0
    return alpha


def tone_match_boundary(result_bgr: np.ndarray, source_bgr: np.ndarray, clear_mask: np.ndarray, blend_mask: np.ndarray) -> np.ndarray:
    outer_ring = cv2.subtract(blend_mask, clear_mask)
    inner_ring = cv2.subtract(clear_mask, erode_mask(clear_mask, 2))
    if np.count_nonzero(outer_ring) == 0 or np.count_nonzero(inner_ring) == 0:
        return result_bgr

    result_lab = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    source_lab = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    outer_values = source_lab[outer_ring > 0]
    inner_values = result_lab[inner_ring > 0]
    if outer_values.size == 0 or inner_values.size == 0:
        return result_bgr

    mean_shift = outer_values.mean(axis=0) - inner_values.mean(axis=0)
    alpha = cv2.GaussianBlur((outer_ring > 0).astype(np.float32), (0, 0), 2.0)
    for channel in range(3):
        result_lab[:, :, channel] += mean_shift[channel] * alpha * 0.45

    return cv2.cvtColor(np.clip(result_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def sharpen_detail(result_bgr: np.ndarray, clear_mask: np.ndarray) -> np.ndarray:
    result_float = result_bgr.astype(np.float32)
    blurred = cv2.GaussianBlur(result_float, (0, 0), 0.9)
    sharpened = np.clip(result_float * 1.12 - blurred * 0.12, 0, 255)
    alpha = cv2.GaussianBlur((clear_mask > 0).astype(np.float32), (0, 0), 1.0)[..., None] * 0.26
    return np.clip(result_float * (1.0 - alpha) + sharpened * alpha, 0, 255).astype(np.uint8)


def model_inpaint(roi_source_bgr: np.ndarray, roi_clear_mask: np.ndarray) -> np.ndarray:
    roi_rgb = cv2.cvtColor(roi_source_bgr, cv2.COLOR_BGR2RGB)
    source_image = Image.fromarray(roi_rgb)
    mask_image = Image.fromarray(roi_clear_mask)
    result = MODEL(source_image, mask_image)
    result_array = np.array(result)
    result_array = result_array[: roi_source_bgr.shape[0], : roi_source_bgr.shape[1]]
    return cv2.cvtColor(result_array, cv2.COLOR_RGB2BGR)


def paste_roi(target: np.ndarray, source: np.ndarray, region: tuple[slice, slice]) -> np.ndarray:
    result = target.copy()
    y_slice, x_slice = region
    result[y_slice, x_slice] = source
    return result


def repair_image_bgr(source_bgr: np.ndarray, core_mask: np.ndarray, clear_mask: np.ndarray, blend_mask: np.ndarray, strength: int) -> np.ndarray:
    bbox = mask_bbox(core_mask)
    roi_source, region = crop_to_bbox(source_bgr, bbox, resolve_roi_margin(strength))
    roi_core = core_mask[region]
    roi_clear = clear_mask[region]
    roi_blend = blend_mask[region]

    model_result = model_inpaint(roi_source, roi_clear)
    model_result = tone_match_boundary(model_result, roi_source, roi_clear, roi_blend)
    model_result = sharpen_detail(model_result, roi_clear)
    alpha = blend_alpha(roi_clear, roi_blend)[..., None]
    blended = roi_source.astype(np.float32) * (1.0 - alpha) + model_result.astype(np.float32) * alpha
    blended_uint8 = np.clip(blended, 0, 255).astype(np.uint8)
    blended_uint8[roi_core > 0] = model_result[roi_core > 0]
    return paste_roi(source_bgr, blended_uint8, region)


def main() -> None:
    if len(sys.argv) != 6:
        raise RuntimeError("参数错误")

    input_path = Path(sys.argv[1])
    mask_path = Path(sys.argv[2])
    output_path = Path(sys.argv[3])
    mode = parse_mode(sys.argv[4])
    strength = parse_strength(sys.argv[5])

    source = read_image(input_path, cv2.IMREAD_UNCHANGED)
    mask_image = read_image(mask_path, cv2.IMREAD_UNCHANGED)
    source_bgr, source_kind, alpha_channel = split_source(source)
    core_mask, clear_mask, blend_mask = build_masks(mask_image, source_bgr.shape[:2], strength)
    repaired_bgr = repair_image_bgr(source_bgr, core_mask, clear_mask, blend_mask, strength)
    result = merge_source(repaired_bgr, source_kind, alpha_channel)
    mime_type = write_image(output_path, result)

    print(
        json.dumps(
            {
                "output_path": str(output_path if output_path.exists() else output_path.with_suffix(".png")),
                "mime_type": mime_type,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
