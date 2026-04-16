import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


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


def parse_quality(raw: str | None) -> str:
    return "hq" if raw == "hq" else "fast"


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


def build_masks(mask_image: np.ndarray, target_shape: tuple[int, int], strength: int, quality: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    clear_expand = 5 + int(round(strength / 15.0))

    if quality == "hq":
        clear_expand += 1

    clear_mask = expand_mask(binary, clear_expand)
    blend_mask = expand_mask(clear_mask, 4 if quality == "fast" else 6)
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


def sobel_magnitude(gray_image: np.ndarray) -> np.ndarray:
    dx = cv2.Sobel(gray_image, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(gray_image, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(dx, dy)


@dataclass
class RegionStats:
    bbox: tuple[int, int, int, int]
    core_area: int
    area_ratio: float
    bbox_area_ratio: float
    center_overlap: float
    texture_score: float
    edge_score: float
    complex_region: bool


def analyze_region(source_bgr: np.ndarray, core_mask: np.ndarray) -> RegionStats:
    height, width = core_mask.shape[:2]
    bbox = mask_bbox(core_mask)
    x0, y0, x1, y1 = bbox
    core_area = max(1, int(np.count_nonzero(core_mask)))
    area_ratio = float(core_area) / float(height * width)
    bbox_area_ratio = float((x1 - x0) * (y1 - y0)) / float(height * width)

    center_mask = np.zeros_like(core_mask)
    center_mask[int(height * 0.22):int(height * 0.78), int(width * 0.22):int(width * 0.78)] = 255
    center_overlap = float(np.count_nonzero(cv2.bitwise_and(core_mask, center_mask))) / float(core_area)

    local_source, region = crop_to_bbox(source_bgr, bbox, 30)
    local_core = core_mask[region]
    gray = cv2.cvtColor(local_source, cv2.COLOR_BGR2GRAY)
    texture = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    edges = sobel_magnitude(gray)

    outer_ring = cv2.subtract(expand_mask(local_core, 8), expand_mask(local_core, 2))
    ring_selector = outer_ring > 0
    texture_score = float(texture[ring_selector].mean()) if np.any(ring_selector) else 0.0
    edge_score = float(edges[ring_selector].mean()) if np.any(ring_selector) else 0.0

    complex_region = (
        (center_overlap >= 0.42 and core_area >= 10000)
        or bbox_area_ratio >= 0.11
        or area_ratio >= 0.05
        or (texture_score >= 24.0 and edge_score >= 32.0 and core_area >= 14000)
    )

    return RegionStats(
        bbox=bbox,
        core_area=core_area,
        area_ratio=area_ratio,
        bbox_area_ratio=bbox_area_ratio,
        center_overlap=center_overlap,
        texture_score=texture_score,
        edge_score=edge_score,
        complex_region=complex_region,
    )


@dataclass
class RepairStrategy:
    mode: str
    strength: int
    requested_quality: str
    effective_quality: str
    roi_margin: int
    inpaint_radius: float
    harmonic_iterations: int
    stage1_scales: list[float]
    use_exemplar: bool
    patch_radius: int
    search_radius: int
    patch_stride: int
    max_candidates: int
    max_frontier_points: int
    reuse_penalty: float
    run_retry: bool


def build_strategy(stats: RegionStats, strength: int, requested_quality: str) -> RepairStrategy:
    effective_quality = "hq" if requested_quality == "hq" or stats.complex_region else "fast"
    high_quality = effective_quality == "hq"

    patch_radius = 4 if strength <= 40 else 5 if strength <= 78 else 6
    if high_quality:
        patch_radius += 1

    roi_margin = 24 + int(round(strength * 0.42)) + patch_radius * 4
    if high_quality:
        roi_margin += 18

    use_exemplar = (
        high_quality
        and requested_quality == "hq"
        and stats.core_area <= 22000
        and stats.bbox_area_ratio <= 0.14
    )

    return RepairStrategy(
        mode="shared",
        strength=strength,
        requested_quality=requested_quality,
        effective_quality=effective_quality,
        roi_margin=roi_margin,
        inpaint_radius=3.4 + ((strength - 1) / 99.0) * 3.6 + (0.4 if high_quality else 0.0),
        harmonic_iterations=(24 + strength // 3) if not high_quality else (52 + strength // 2),
        stage1_scales=[0.72, 1.0] if not high_quality else [0.52, 0.74, 1.0],
        use_exemplar=use_exemplar,
        patch_radius=patch_radius,
        search_radius=60 + int(round(strength * 0.7)) + (34 if high_quality else 0),
        patch_stride=2 if not high_quality else 1,
        max_candidates=88 if not high_quality else 140,
        max_frontier_points=12 if not high_quality else 18,
        reuse_penalty=2.4 if not high_quality else 1.5,
        run_retry=(requested_quality == "hq") and stats.core_area <= 52000,
    )


def aggressive_clear(source_bgr: np.ndarray, clear_mask: np.ndarray, strategy: RepairStrategy, edge_score: float) -> np.ndarray:
    previous: np.ndarray | None = None
    telea_weight = 0.76 if edge_score < 24.0 else 0.62

    for scale in strategy.stage1_scales:
        target_width = max(48, int(round(source_bgr.shape[1] * scale)))
        target_height = max(48, int(round(source_bgr.shape[0] * scale)))
        scaled_source = cv2.resize(source_bgr, (target_width, target_height), interpolation=cv2.INTER_AREA)
        scaled_mask = cv2.resize(clear_mask, (target_width, target_height), interpolation=cv2.INTER_NEAREST)

        if previous is not None:
            previous_up = cv2.resize(previous, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
            scaled_source[scaled_mask > 0] = previous_up[scaled_mask > 0]

        radius = max(2.0, strategy.inpaint_radius * max(0.72, scale))
        telea = cv2.inpaint(scaled_source, scaled_mask, radius, cv2.INPAINT_TELEA)
        ns = cv2.inpaint(scaled_source, scaled_mask, radius + 0.9, cv2.INPAINT_NS)
        merged = cv2.addWeighted(telea, telea_weight, ns, 1.0 - telea_weight, 0.0)

        if scale >= 0.99:
            softened = cv2.GaussianBlur(merged, (3, 3), 0)
            merged = cv2.addWeighted(merged, 0.9, softened, 0.1, 0.0)

        previous = merged

    return previous if previous is not None else source_bgr.copy()


def harmonic_fill(source_bgr: np.ndarray, clear_mask: np.ndarray, initial_bgr: np.ndarray, iterations: int) -> np.ndarray:
    source_lab = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    working_lab = cv2.cvtColor(initial_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    selector = clear_mask > 0
    kernel = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float32) * 0.25

    for _ in range(iterations):
        averaged = cv2.filter2D(working_lab, -1, kernel, borderType=cv2.BORDER_REFLECT)
        working_lab[selector] = averaged[selector]
        working_lab[~selector] = source_lab[~selector]

    return cv2.cvtColor(np.clip(working_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def texture_weight_map(source_bgr: np.ndarray, clear_mask: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    detail = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    known = (clear_mask == 0).astype(np.float32)
    sigma = max(3.0, min(source_bgr.shape[:2]) / 28.0)
    propagated = cv2.GaussianBlur(detail * known, (0, 0), sigma) / (cv2.GaussianBlur(known, (0, 0), sigma) + 1e-4)

    values = propagated[clear_mask > 0]
    if values.size == 0:
        return np.zeros_like(propagated, dtype=np.float32)

    low = float(np.percentile(values, 25))
    high = float(np.percentile(values, 85))
    if high - low < 1e-3:
        return np.zeros_like(propagated, dtype=np.float32)

    normalized = (propagated - low) / (high - low)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


def extract_detail_layers(image_bgr: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    image_float = image_bgr.astype(np.float32)
    low = cv2.GaussianBlur(image_float, (0, 0), sigma)
    high = image_float - low
    return low, high


def reconstruct_detail_layer(source_bgr: np.ndarray, clear_mask: np.ndarray, strategy: RepairStrategy) -> np.ndarray:
    _, source_high = extract_detail_layers(source_bgr, 0.95)
    shifted = np.clip(source_high + 128.0, 0, 255).astype(np.uint8)
    radius = 2.1 if strategy.effective_quality == "fast" else 2.8
    telea = cv2.inpaint(shifted, clear_mask, radius, cv2.INPAINT_TELEA)
    ns = cv2.inpaint(shifted, clear_mask, radius + 0.6, cv2.INPAINT_NS)
    merged = cv2.addWeighted(telea, 0.7, ns, 0.3, 0.0)
    return merged.astype(np.float32) - 128.0


def build_detail_weight(texture_weight: np.ndarray, clear_mask: np.ndarray) -> np.ndarray:
    distance_inside = cv2.distanceTransform((clear_mask > 0).astype(np.uint8) * 255, cv2.DIST_L2, 3).astype(np.float32)
    if float(distance_inside.max()) > 0.0:
        distance_inside /= float(distance_inside.max())

    weight = (0.2 + texture_weight * 0.8) * (0.3 + distance_inside * 0.7)
    return np.clip(weight, 0.0, 1.0)


def rebuild_texture_details(
    source_bgr: np.ndarray,
    base_bgr: np.ndarray,
    detail_seed_bgr: np.ndarray,
    clear_mask: np.ndarray,
    texture_weight: np.ndarray,
    strategy: RepairStrategy,
) -> np.ndarray:
    # Keep low-frequency lighting from the structure pass, but inject high-frequency detail back
    # from the sharper synthesis seed so the repaired area does not collapse into a blurry patch.
    base_low, base_high = extract_detail_layers(base_bgr, 1.18)
    _, seed_high = extract_detail_layers(detail_seed_bgr, 1.0)
    _, source_high = extract_detail_layers(source_bgr, 1.0)
    reconstructed_high = reconstruct_detail_layer(source_bgr, clear_mask, strategy)

    outer_ring = cv2.subtract(expand_mask(clear_mask, 5), clear_mask)
    detail_gain = 1.0

    if np.count_nonzero(outer_ring) > 0 and np.count_nonzero(clear_mask) > 0:
        outer_energy = float(np.mean(np.abs(source_high[outer_ring > 0])))
        inner_energy = float(np.mean(np.abs(seed_high[clear_mask > 0]))) + 1e-4
        max_gain = 1.42 if strategy.effective_quality == "hq" else 1.24
        detail_gain = float(np.clip(outer_energy / inner_energy, 0.9, max_gain))

    detail_weight = build_detail_weight(texture_weight, clear_mask)[..., None]
    if strategy.effective_quality == "hq":
        hybrid_high = seed_high * 0.5 + reconstructed_high * 0.34 + base_high * 0.16
    else:
        hybrid_high = seed_high * 0.54 + reconstructed_high * 0.18 + base_high * 0.28
    hybrid_high *= detail_gain

    rebuilt = np.clip(base_low + hybrid_high, 0, 255)
    return np.clip(base_bgr.astype(np.float32) * (1.0 - detail_weight) + rebuilt * detail_weight, 0, 255).astype(np.uint8)


def sharpen_boundary_detail(
    result_bgr: np.ndarray,
    clear_mask: np.ndarray,
    texture_weight: np.ndarray,
    strategy: RepairStrategy,
) -> np.ndarray:
    inner_ring = cv2.subtract(clear_mask, erode_mask(clear_mask, 2))
    if np.count_nonzero(inner_ring) == 0:
        return result_bgr

    result_float = result_bgr.astype(np.float32)
    blurred = cv2.GaussianBlur(result_float, (0, 0), 0.95)
    sharpened = np.clip(result_float * (1.16 if strategy.effective_quality == "hq" else 1.1) - blurred * (0.16 if strategy.effective_quality == "hq" else 0.1), 0, 255)
    alpha = cv2.GaussianBlur((inner_ring > 0).astype(np.float32), (0, 0), 0.85)
    alpha *= 0.2 + texture_weight * 0.35
    alpha = np.clip(alpha, 0.0, 0.42)[..., None]
    return np.clip(result_float * (1.0 - alpha) + sharpened * alpha, 0, 255).astype(np.uint8)


def match_clear_region_contrast(result_bgr: np.ndarray, source_bgr: np.ndarray, clear_mask: np.ndarray, strategy: RepairStrategy) -> np.ndarray:
    outer_ring = cv2.subtract(expand_mask(clear_mask, 5), clear_mask)
    if np.count_nonzero(outer_ring) == 0 or np.count_nonzero(clear_mask) == 0:
        return result_bgr

    result_lab = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    source_lab = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    outer_l = source_lab[:, :, 0][outer_ring > 0]
    inner_l = result_lab[:, :, 0][clear_mask > 0]
    if outer_l.size == 0 or inner_l.size == 0:
        return result_bgr

    outer_mean = float(outer_l.mean())
    outer_std = float(outer_l.std()) + 1e-4
    inner_mean = float(inner_l.mean())
    inner_std = float(inner_l.std()) + 1e-4

    gain_limit = 1.18 if strategy.effective_quality == "hq" else 1.1
    contrast_gain = float(np.clip(outer_std / inner_std, 0.92, gain_limit))
    adjusted_l = (result_lab[:, :, 0] - inner_mean) * contrast_gain + inner_mean * 0.35 + outer_mean * 0.65
    alpha = cv2.GaussianBlur((clear_mask > 0).astype(np.float32), (0, 0), 1.1)
    result_lab[:, :, 0] = result_lab[:, :, 0] * (1.0 - alpha) + adjusted_l * alpha
    return cv2.cvtColor(np.clip(result_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def patch_bounds(center_y: int, center_x: int, radius: int, height: int, width: int) -> tuple[int, int, int, int] | None:
    top = center_y - radius
    left = center_x - radius
    bottom = center_y + radius + 1
    right = center_x + radius + 1
    if top < 0 or left < 0 or bottom > height or right > width:
        return None
    return top, bottom, left, right


def build_candidate_centers(valid_mask: np.ndarray, stride: int, radius: int) -> np.ndarray:
    height, width = valid_mask.shape[:2]
    ys: list[int] = []
    xs: list[int] = []

    for y in range(radius, height - radius, stride):
        for x in range(radius, width - radius, stride):
            if valid_mask[y, x]:
                ys.append(y)
                xs.append(x)

    if not ys:
        return np.empty((0, 2), dtype=np.int32)

    return np.column_stack((np.array(ys, dtype=np.int32), np.array(xs, dtype=np.int32)))


def choose_candidates(candidate_points: np.ndarray, center_y: int, center_x: int, search_radius: int, limit: int) -> np.ndarray:
    if candidate_points.size == 0:
        return candidate_points

    delta_y = np.abs(candidate_points[:, 0] - center_y)
    delta_x = np.abs(candidate_points[:, 1] - center_x)
    local_selector = (delta_y <= search_radius) & (delta_x <= search_radius)
    points = candidate_points[local_selector]

    if points.size == 0:
        points = candidate_points
        delta_y = np.abs(points[:, 0] - center_y)
        delta_x = np.abs(points[:, 1] - center_x)
    else:
        delta_y = np.abs(points[:, 0] - center_y)
        delta_x = np.abs(points[:, 1] - center_x)

    if len(points) > limit:
        order = np.argsort(delta_y + delta_x)
        points = points[order[:limit]]

    return points


def patch_score(
    target_patch_lab: np.ndarray,
    candidate_patch_lab: np.ndarray,
    known_selector: np.ndarray,
    target_luma: np.ndarray,
    candidate_luma: np.ndarray,
    usage_penalty: float,
    distance_penalty: float,
) -> float:
    if not np.any(known_selector):
        return float("inf")

    target_values = target_patch_lab[known_selector].astype(np.float32)
    candidate_values = candidate_patch_lab[known_selector].astype(np.float32)
    weights = np.array([1.0, 0.65, 0.65], dtype=np.float32)
    color_error = np.mean(((target_values - candidate_values) ** 2) * weights)
    gradient_error = np.mean((target_luma[known_selector].astype(np.float32) - candidate_luma[known_selector].astype(np.float32)) ** 2)
    return float(color_error + gradient_error * 0.34 + usage_penalty + distance_penalty)


def exemplar_fill(
    source_bgr: np.ndarray,
    seed_bgr: np.ndarray,
    clear_mask: np.ndarray,
    texture_weight: np.ndarray,
    strategy: RepairStrategy,
) -> np.ndarray:
    radius = strategy.patch_radius
    height, width = clear_mask.shape[:2]
    source_lab = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    working_lab = cv2.cvtColor(seed_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    source_luma = source_lab[:, :, 0]
    fill_mask = clear_mask > 0

    valid_source = erode_mask((clear_mask == 0).astype(np.uint8) * 255, radius)
    candidate_points = build_candidate_centers(valid_source > 0, strategy.patch_stride, radius)
    if candidate_points.size == 0:
        return seed_bgr

    usage_counts: dict[tuple[int, int], int] = {}
    frontier_kernel = np.ones((3, 3), dtype=np.uint8)
    distance_map = cv2.distanceTransform((fill_mask.astype(np.uint8) * 255), cv2.DIST_L2, 3)
    max_rounds = 48 if strategy.effective_quality == "fast" else 90

    for _ in range(max_rounds):
        if not np.any(fill_mask):
            break

        frontier = fill_mask & (cv2.dilate((~fill_mask).astype(np.uint8), frontier_kernel, iterations=1) > 0)
        frontier_points = np.argwhere(frontier)
        if frontier_points.size == 0:
            break

        weights = texture_weight[frontier]
        order = np.argsort(-weights)
        frontier_points = frontier_points[order]

        selected: list[tuple[int, int]] = []
        occupancy = np.zeros_like(fill_mask, dtype=np.uint8)
        min_gap = max(2, radius - 1)

        for point in frontier_points:
            center_y = int(point[0])
            center_x = int(point[1])
            top = max(0, center_y - min_gap)
            left = max(0, center_x - min_gap)
            bottom = min(height, center_y + min_gap + 1)
            right = min(width, center_x + min_gap + 1)
            if np.any(occupancy[top:bottom, left:right]):
                continue
            occupancy[top:bottom, left:right] = 1
            selected.append((center_y, center_x))
            if len(selected) >= strategy.max_frontier_points:
                break

        progressed = False

        for center_y, center_x in selected:
            bounds = patch_bounds(center_y, center_x, radius, height, width)
            if bounds is None:
                continue

            top, bottom, left, right = bounds
            unknown_patch = fill_mask[top:bottom, left:right]
            known_selector = ~unknown_patch
            if int(np.count_nonzero(known_selector)) < max(16, radius * 6):
                continue

            layer_mask = unknown_patch & (distance_map[top:bottom, left:right] <= distance_map[center_y, center_x] + 0.8)
            if not np.any(layer_mask):
                layer_mask = unknown_patch

            target_patch_lab = working_lab[top:bottom, left:right]
            target_patch_luma = target_patch_lab[:, :, 0]
            candidates = choose_candidates(candidate_points, center_y, center_x, strategy.search_radius, strategy.max_candidates)
            if candidates.size == 0:
                continue

            best_score = float("inf")
            best_candidate: tuple[int, int] | None = None

            for candidate_y, candidate_x in candidates:
                candidate_bounds = patch_bounds(int(candidate_y), int(candidate_x), radius, height, width)
                if candidate_bounds is None:
                    continue

                candidate_top, candidate_bottom, candidate_left, candidate_right = candidate_bounds
                candidate_patch_lab = source_lab[candidate_top:candidate_bottom, candidate_left:candidate_right]
                candidate_patch_luma = source_luma[candidate_top:candidate_bottom, candidate_left:candidate_right]

                usage_penalty = usage_counts.get((int(candidate_y), int(candidate_x)), 0) * strategy.reuse_penalty
                distance_penalty = ((abs(int(candidate_y) - center_y) + abs(int(candidate_x) - center_x)) / max(1.0, strategy.search_radius)) * 0.45
                score = patch_score(
                    target_patch_lab,
                    candidate_patch_lab,
                    known_selector,
                    target_patch_luma,
                    candidate_patch_luma,
                    usage_penalty,
                    distance_penalty,
                )

                if score < best_score:
                    best_score = score
                    best_candidate = (int(candidate_y), int(candidate_x))

            if best_candidate is None:
                continue

            candidate_bounds = patch_bounds(best_candidate[0], best_candidate[1], radius, height, width)
            if candidate_bounds is None:
                continue

            candidate_top, candidate_bottom, candidate_left, candidate_right = candidate_bounds
            candidate_patch_lab = source_lab[candidate_top:candidate_bottom, candidate_left:candidate_right]
            patch_view = working_lab[top:bottom, left:right]
            patch_view[layer_mask] = candidate_patch_lab[layer_mask]
            fill_mask[top:bottom, left:right][layer_mask] = False
            usage_counts[best_candidate] = usage_counts.get(best_candidate, 0) + 1
            progressed = True

        if not progressed:
            break

    if np.any(fill_mask):
        fallback = aggressive_clear(
            cv2.cvtColor(np.clip(working_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR),
            fill_mask.astype(np.uint8) * 255,
            strategy,
            edge_score=22.0,
        )
        fallback_lab = cv2.cvtColor(fallback, cv2.COLOR_BGR2LAB).astype(np.float32)
        working_lab[fill_mask] = fallback_lab[fill_mask]

    return cv2.cvtColor(np.clip(working_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def blend_alpha(clear_mask: np.ndarray, blend_mask: np.ndarray) -> np.ndarray:
    alpha = cv2.GaussianBlur((blend_mask > 0).astype(np.float32), (0, 0), 1.9)
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
    alpha = cv2.GaussianBlur((outer_ring > 0).astype(np.float32), (0, 0), 1.8)
    for channel in range(3):
        result_lab[:, :, channel] += mean_shift[channel] * alpha * 0.58

    return cv2.cvtColor(np.clip(result_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


@dataclass
class ConsistencyMetrics:
    brightness_jump: float
    chroma_jump: float
    edge_jump: float

    def score(self) -> float:
        return self.brightness_jump + self.chroma_jump + self.edge_jump

    def needs_retry(self) -> bool:
        return self.brightness_jump >= 1.05 or self.chroma_jump >= 1.0 or self.edge_jump >= 1.5


def evaluate_consistency(result_bgr: np.ndarray, clear_mask: np.ndarray) -> ConsistencyMetrics:
    outer_ring = cv2.subtract(expand_mask(clear_mask, 2), clear_mask)
    inner_ring = cv2.subtract(clear_mask, erode_mask(clear_mask, 2))
    if np.count_nonzero(outer_ring) == 0 or np.count_nonzero(inner_ring) == 0:
        return ConsistencyMetrics(0.0, 0.0, 0.0)

    result_lab = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    outer_values = result_lab[outer_ring > 0]
    inner_values = result_lab[inner_ring > 0]
    if outer_values.size == 0 or inner_values.size == 0:
        return ConsistencyMetrics(0.0, 0.0, 0.0)

    outer_std = np.std(outer_values, axis=0) + np.array([8.0, 6.0, 6.0], dtype=np.float32)
    brightness_jump = float(abs(inner_values[:, 0].mean() - outer_values[:, 0].mean()) / outer_std[0])

    inner_chroma = np.sqrt((inner_values[:, 1] - 128.0) ** 2 + (inner_values[:, 2] - 128.0) ** 2)
    outer_chroma = np.sqrt((outer_values[:, 1] - 128.0) ** 2 + (outer_values[:, 2] - 128.0) ** 2)
    chroma_jump = float(abs(inner_chroma.mean() - outer_chroma.mean()) / (outer_chroma.std() + 6.0))

    gradient = sobel_magnitude(cv2.cvtColor(result_bgr, cv2.COLOR_BGR2GRAY))
    edge_jump = float((gradient[inner_ring > 0].mean() + 1e-4) / (gradient[outer_ring > 0].mean() + 1e-4))

    return ConsistencyMetrics(
        brightness_jump=brightness_jump,
        chroma_jump=chroma_jump,
        edge_jump=edge_jump,
    )


def mix_results(structure_bgr: np.ndarray, texture_bgr: np.ndarray, texture_weight: np.ndarray) -> np.ndarray:
    alpha = np.clip(texture_weight, 0.0, 1.0)[..., None]
    blended = structure_bgr.astype(np.float32) * (1.0 - alpha) + texture_bgr.astype(np.float32) * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)


def paste_roi(target: np.ndarray, source: np.ndarray, region: tuple[slice, slice]) -> np.ndarray:
    result = target.copy()
    y_slice, x_slice = region
    result[y_slice, x_slice] = source
    return result


def repair_pass(
    source_bgr: np.ndarray,
    core_mask: np.ndarray,
    clear_mask: np.ndarray,
    blend_mask: np.ndarray,
    strategy: RepairStrategy,
    stats: RegionStats,
) -> tuple[np.ndarray, ConsistencyMetrics]:
    roi_source, region = crop_to_bbox(source_bgr, stats.bbox, strategy.roi_margin)
    roi_core = core_mask[region]
    roi_clear = clear_mask[region]
    roi_blend = blend_mask[region]

    stage1 = aggressive_clear(roi_source, roi_clear, strategy, stats.edge_score)
    structure = harmonic_fill(roi_source, roi_clear, stage1, strategy.harmonic_iterations)
    texture_weight = texture_weight_map(roi_source, roi_clear)

    if strategy.use_exemplar:
        texture = exemplar_fill(roi_source, stage1, roi_clear, texture_weight, strategy)
        base = mix_results(structure, stage1, np.clip(0.18 + texture_weight * 0.26, 0.18, 0.46))
        repaired = mix_results(base, texture, texture_weight * 0.9)
        repaired = rebuild_texture_details(roi_source, repaired, texture, roi_clear, texture_weight, strategy)
    else:
        base = mix_results(structure, stage1, np.clip(0.28 + texture_weight * 0.5, 0.28, 0.78))
        repaired = rebuild_texture_details(roi_source, base, stage1, roi_clear, texture_weight, strategy)

    repaired = match_clear_region_contrast(repaired, roi_source, roi_clear, strategy)
    matched = tone_match_boundary(repaired, roi_source, roi_clear, roi_blend)
    matched = sharpen_boundary_detail(matched, roi_clear, texture_weight, strategy)
    alpha = blend_alpha(roi_clear, roi_blend)[..., None]
    blended = roi_source.astype(np.float32) * (1.0 - alpha) + matched.astype(np.float32) * alpha
    blended_uint8 = np.clip(blended, 0, 255).astype(np.uint8)
    blended_uint8[roi_core > 0] = matched[roi_core > 0]
    metrics = evaluate_consistency(blended_uint8, roi_clear)
    return paste_roi(source_bgr, blended_uint8, region), metrics


def repair_image_bgr(source_bgr: np.ndarray, core_mask: np.ndarray, clear_mask: np.ndarray, blend_mask: np.ndarray, strength: int, quality: str) -> np.ndarray:
    stats = analyze_region(source_bgr, core_mask)
    strategy = build_strategy(stats, strength, quality)
    first_result, first_metrics = repair_pass(source_bgr, core_mask, clear_mask, blend_mask, strategy, stats)

    if not strategy.run_retry or not first_metrics.needs_retry():
        return first_result

    retry_clear = expand_mask(clear_mask, 2)
    retry_blend = expand_mask(blend_mask, 2)
    retry_strategy = build_strategy(stats, min(100, strength + 14), "hq")
    retry_result, retry_metrics = repair_pass(first_result, core_mask, retry_clear, retry_blend, retry_strategy, stats)

    return retry_result if retry_metrics.score() <= first_metrics.score() else first_result


def main() -> None:
    if len(sys.argv) != 7:
        raise RuntimeError("参数错误")

    input_path = Path(sys.argv[1])
    mask_path = Path(sys.argv[2])
    output_path = Path(sys.argv[3])
    mode = parse_mode(sys.argv[4])
    strength = parse_strength(sys.argv[5])
    quality = parse_quality(sys.argv[6])

    source = read_image(input_path, cv2.IMREAD_UNCHANGED)
    mask_image = read_image(mask_path, cv2.IMREAD_UNCHANGED)
    source_bgr, source_kind, alpha_channel = split_source(source)
    core_mask, clear_mask, blend_mask = build_masks(mask_image, source_bgr.shape[:2], strength, quality)
    repaired_bgr = repair_image_bgr(source_bgr, core_mask, clear_mask, blend_mask, strength, quality)
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
