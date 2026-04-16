import { loadOpenCv } from '../hooks/useOpenCv';
import type { ImageRepairQuality } from '../types';

const FORCE_LIGHTWEIGHT_BROWSER_PIPELINE = true;
const ENABLE_OPENCV_BROWSER_REFINEMENT = false;
const ENABLE_RECT_SELECTION_FOREGROUND_REFINEMENT = true;

interface ProcessImageLocallyOptions {
  sourceCanvas: HTMLCanvasElement;
  maskCanvas: HTMLCanvasElement;
  quality: ImageRepairQuality;
  strength: number;
  onProgress?: (progress: number, message: string) => void;
}

interface MaskBounds {
  left: number;
  top: number;
  right: number;
  bottom: number;
}

interface RoiRect {
  x: number;
  y: number;
  width: number;
  height: number;
}

interface WorkingRoi {
  source: ImageData;
  mask: ImageData;
  scale: number;
}

type Rgb = {
  r: number;
  g: number;
  b: number;
};

type PatchMatch = {
  x: number;
  y: number;
  score: number;
  adjustment: Rgb;
};

type RegionStats = {
  count: number;
  mean: Rgb;
  std: Rgb;
};

type BoundarySample = Rgb & {
  distance: number;
  diagonal: boolean;
};

export async function processImageLocally({
  sourceCanvas,
  maskCanvas,
  quality,
  strength,
  onProgress,
}: ProcessImageLocallyOptions) {
  const maskContext = maskCanvas.getContext('2d');
  const sourceContext = sourceCanvas.getContext('2d');

  if (!maskContext || !sourceContext) {
    throw new Error('无法读取当前画布数据。');
  }

  const maskImageData = maskContext.getImageData(0, 0, maskCanvas.width, maskCanvas.height);
  const bounds = getMaskBounds(maskImageData);
  if (!bounds) {
    throw new Error('没有检测到有效选区，请先标注水印区域。');
  }

  const roi = expandBounds(bounds, sourceCanvas.width, sourceCanvas.height, resolveMargin(quality, strength));

  onProgress?.(18, '正在读取局部处理区域…');
  await waitForFrame();

  const sourceRoiData = sourceContext.getImageData(roi.x, roi.y, roi.width, roi.height);
  const maskRoiData = maskContext.getImageData(roi.x, roi.y, roi.width, roi.height);
  const working = resizeForProcessing(sourceRoiData, maskRoiData, quality);

  if (working.scale < 1) {
    onProgress?.(26, '局部区域较大，正在缩放后修复以保持流畅…');
    await waitForFrame();
  }

  const originalPixels = new Uint8ClampedArray(working.source.data);
  const selectionMask = imageDataToMask(working.mask);
  const baseMask = await refineSelectionMask({
    originalPixels,
    selectionMask,
    width: working.source.width,
    height: working.source.height,
    quality,
    onProgress,
  });
  const transparentOverlayPixels = await removeTransparentOverlayCandidate({
    originalPixels,
    selectionMask,
    width: working.source.width,
    height: working.source.height,
    quality,
    onProgress,
  });
  if (transparentOverlayPixels) {
    const processedWorkingImage = new ImageData(
      new Uint8ClampedArray(transparentOverlayPixels),
      working.source.width,
      working.source.height,
    );
    const finalImageData = working.scale < 1
      ? resizeImageData(processedWorkingImage, roi.width, roi.height, true)
      : processedWorkingImage;

    return {
      roi,
      imageData: finalImageData,
    };
  }

  const expandedMask = dilateMask(
    baseMask,
    working.source.width,
    working.source.height,
    resolveDilateIterations(quality, strength, working.scale),
  );
  const preservedMask = new Uint8Array(expandedMask);
  const protectedMask = resolveProtectedCoreMask(baseMask, working.source.width, working.source.height, quality);
  const maskedPixelCount = countMasked(preservedMask);
  const distanceToKnown = computeDistanceMap(preservedMask, working.source.width, working.source.height, 0);
  const distanceInsideSelection = computeDistanceMap(baseMask, working.source.width, working.source.height, 0);
  const distanceToSelection = computeDistanceMap(baseMask, working.source.width, working.source.height, 1);

  onProgress?.(34, '正在优先彻底清除水印内容…');
  const useLightweightPipeline = FORCE_LIGHTWEIGHT_BROWSER_PIPELINE || shouldUseLightweightRemovalPipeline({
    selectionMask,
    baseMask,
    maskedPixelCount,
    width: working.source.width,
    height: working.source.height,
    quality,
  });

  if (useLightweightPipeline) {
    onProgress?.(34, '正在使用轻量对象移除流程，避免处理时卡住…');
    await yieldToBrowser();

    const openCvCandidate = await generateOpenCvInpaintCandidate({
      originalPixels,
      fillMask: preservedMask,
      maskedPixelCount,
      width: working.source.width,
      height: working.source.height,
      quality,
      strength,
      onProgress,
    });

    const diffusionCandidate = await generateDiffusionCandidate({
      originalPixels,
      fillMask: preservedMask,
      maskedPixelCount,
      width: working.source.width,
      height: working.source.height,
      quality,
      onProgress,
    });

    const repairConfidence = computeRepairConfidenceMap({
      selectionMask: baseMask,
      width: working.source.width,
      height: working.source.height,
      quality,
    });

    const mergedCandidate = await mergeRepairCandidates({
      patchPixels: diffusionCandidate,
      openCvPixels: openCvCandidate,
      originalPixels,
      fillMask: preservedMask,
      distanceToKnown,
      maskedPixelCount,
      width: working.source.width,
      height: working.source.height,
      quality,
      onProgress,
    });

    const smoothPriorityPixels = await mergeSmoothGradientCandidate({
      basePixels: mergedCandidate,
      smoothPixels: diffusionCandidate,
      originalPixels,
      fillMask: preservedMask,
      distanceToKnown,
      maskedPixelCount,
      width: working.source.width,
      height: working.source.height,
      quality,
      preserveBaseDetail: Boolean(openCvCandidate),
      onProgress,
    });

    const boundaryHarmonized = await harmonizeBoundaryTransition({
      pixels: smoothPriorityPixels,
      originalPixels,
      fillMask: preservedMask,
      protectedMask,
      distanceToKnown,
      maskedPixelCount,
      width: working.source.width,
      height: working.source.height,
      quality,
      onProgress,
    });

    const smoothedPixels = await smoothFilledRegion({
      pixels: boundaryHarmonized,
      fillMask: preservedMask,
      distanceToKnown,
      maskedPixelCount,
      width: working.source.width,
      height: working.source.height,
      quality,
      onProgress,
    });

    const texturedPixels = await reinjectDirectionalTexture({
      pixels: smoothedPixels,
      originalPixels,
      fillMask: preservedMask,
      distanceToKnown,
      maskedPixelCount,
      width: working.source.width,
      height: working.source.height,
      quality,
      onProgress,
    });

    const seamSuppressedPixels = await suppressMaskBoundaryEcho({
      pixels: texturedPixels,
      originalPixels,
      fillMask: preservedMask,
      baseMask,
      distanceInsideSelection,
      distanceToSelection,
      maskedPixelCount,
      width: working.source.width,
      height: working.source.height,
      quality,
      onProgress,
    });

    const toneMatchedPixels = await matchFilledTone(
      seamSuppressedPixels,
      originalPixels,
      preservedMask,
      working.source.width,
      working.source.height,
      quality,
      onProgress,
    );

    const boundaryAlignedPixels = await alignFilledRegionToBoundaryField({
      pixels: toneMatchedPixels,
      originalPixels,
      fillMask: preservedMask,
      distanceToKnown,
      maskedPixelCount,
      width: working.source.width,
      height: working.source.height,
      quality,
      onProgress,
    });

    const structureRestoredPixels = await restoreLinearBoundaryFeatures({
      pixels: boundaryAlignedPixels,
      originalPixels,
      fillMask: preservedMask,
      distanceToKnown,
      maskedPixelCount,
      width: working.source.width,
      height: working.source.height,
      quality,
      onProgress,
    });

    const textureMatchedPixels = await regrainFilledRegion({
      pixels: structureRestoredPixels,
      originalPixels,
      fillMask: preservedMask,
      distanceToKnown,
      maskedPixelCount,
      width: working.source.width,
      height: working.source.height,
      quality,
      onProgress,
    });

    const blendedPixels = await blendWithOriginal({
      originalPixels,
      rebuiltPixels: textureMatchedPixels,
      baseMask,
      fillMask: preservedMask,
      protectedMask,
      repairConfidence,
      distanceToKnown,
      distanceInsideSelection,
      distanceToSelection,
      maskedPixelCount,
      width: working.source.width,
      height: working.source.height,
      quality,
      onProgress,
    });

    const residualCleanedPixels = await cleanupResidualSelectionForeground({
      pixels: blendedPixels,
      originalPixels,
      selectionMask,
      baseMask,
      maskedPixelCount,
      width: working.source.width,
      height: working.source.height,
      quality,
      onProgress,
    });

    const processedWorkingImage = new ImageData(new Uint8ClampedArray(residualCleanedPixels), working.source.width, working.source.height);
    const finalImageData = working.scale < 1
      ? resizeImageData(processedWorkingImage, roi.width, roi.height, true)
      : processedWorkingImage;

    return {
      roi,
      imageData: finalImageData,
    };
  }

  const workingPixels = new Uint8ClampedArray(working.source.data);

  await fillMaskedPixels({
    pixels: workingPixels,
    mask: expandedMask,
    width: working.source.width,
    height: working.source.height,
    quality,
    strength,
    onProgress,
  });

  onProgress?.(72, '正在重建局部纹理细节…');
  await waitForFrame();

  const textureRebuilt = await rebuildTexture({
    pixels: workingPixels,
    originalPixels,
    fillMask: preservedMask,
    width: working.source.width,
    height: working.source.height,
    quality,
    strength,
    onProgress,
  });

  onProgress?.(86, '正在修正补丁边缘并压掉局部锯齿…');
  await waitForFrame();

  const openCvCandidate = await generateOpenCvInpaintCandidate({
    originalPixels,
    fillMask: preservedMask,
    maskedPixelCount,
    width: working.source.width,
    height: working.source.height,
    quality,
    strength,
    onProgress,
  });

  const diffusionCandidate = await generateDiffusionCandidate({
    originalPixels,
    fillMask: preservedMask,
    maskedPixelCount,
    width: working.source.width,
    height: working.source.height,
    quality,
    onProgress,
  });
  const repairConfidence = computeRepairConfidenceMap({
    selectionMask: baseMask,
    width: working.source.width,
    height: working.source.height,
    quality,
  });

  const mergedPixels = await mergeRepairCandidates({
    patchPixels: textureRebuilt,
    openCvPixels: openCvCandidate,
    originalPixels,
    fillMask: preservedMask,
    distanceToKnown,
    maskedPixelCount,
    width: working.source.width,
    height: working.source.height,
    quality,
    onProgress,
  });

  const smoothPriorityPixels = await mergeSmoothGradientCandidate({
    basePixels: mergedPixels,
    smoothPixels: diffusionCandidate,
    originalPixels,
    fillMask: preservedMask,
    distanceToKnown,
    maskedPixelCount,
    width: working.source.width,
    height: working.source.height,
    quality,
    preserveBaseDetail: Boolean(openCvCandidate),
    onProgress,
  });

  const boundaryHarmonized = await harmonizeBoundaryTransition({
    pixels: smoothPriorityPixels,
    originalPixels,
    fillMask: preservedMask,
    protectedMask,
    distanceToKnown,
    maskedPixelCount,
    width: working.source.width,
    height: working.source.height,
    quality,
    onProgress,
  });

  const smoothedPixels = await smoothFilledRegion({
    pixels: boundaryHarmonized,
    fillMask: preservedMask,
    distanceToKnown,
    maskedPixelCount,
    width: working.source.width,
    height: working.source.height,
    quality,
    onProgress,
  });

  const texturedPixels = await reinjectDirectionalTexture({
    pixels: smoothedPixels,
    originalPixels,
    fillMask: preservedMask,
    distanceToKnown,
    maskedPixelCount,
    width: working.source.width,
    height: working.source.height,
    quality,
    onProgress,
  });

  const seamSuppressedPixels = await suppressMaskBoundaryEcho({
    pixels: texturedPixels,
    originalPixels,
    fillMask: preservedMask,
    baseMask,
    distanceInsideSelection,
    distanceToSelection,
    maskedPixelCount,
    width: working.source.width,
    height: working.source.height,
    quality,
    onProgress,
  });

  const toneMatchedPixels = await matchFilledTone(
    seamSuppressedPixels,
    originalPixels,
    preservedMask,
    working.source.width,
    working.source.height,
    quality,
    onProgress,
  );

  const boundaryAlignedPixels = await alignFilledRegionToBoundaryField({
    pixels: toneMatchedPixels,
    originalPixels,
    fillMask: preservedMask,
    distanceToKnown,
    maskedPixelCount,
    width: working.source.width,
    height: working.source.height,
    quality,
    onProgress,
  });

  const structureRestoredPixels = await restoreLinearBoundaryFeatures({
    pixels: boundaryAlignedPixels,
    originalPixels,
    fillMask: preservedMask,
    distanceToKnown,
    maskedPixelCount,
    width: working.source.width,
    height: working.source.height,
    quality,
    onProgress,
  });

  const textureMatchedPixels = await regrainFilledRegion({
    pixels: structureRestoredPixels,
    originalPixels,
    fillMask: preservedMask,
    distanceToKnown,
    maskedPixelCount,
    width: working.source.width,
    height: working.source.height,
    quality,
    onProgress,
  });

  const blendedPixels = await blendWithOriginal({
    originalPixels,
    rebuiltPixels: textureMatchedPixels,
    baseMask,
    fillMask: preservedMask,
    protectedMask,
    repairConfidence,
    distanceToKnown,
    distanceInsideSelection,
    distanceToSelection,
    maskedPixelCount,
    width: working.source.width,
    height: working.source.height,
    quality,
    onProgress,
  });

  const residualCleanedPixels = await cleanupResidualSelectionForeground({
    pixels: blendedPixels,
    originalPixels,
    selectionMask,
    baseMask,
    maskedPixelCount,
    width: working.source.width,
    height: working.source.height,
    quality,
    onProgress,
  });

  const processedWorkingImage = new ImageData(new Uint8ClampedArray(residualCleanedPixels), working.source.width, working.source.height);
  const finalImageData = working.scale < 1
    ? resizeImageData(processedWorkingImage, roi.width, roi.height, true)
    : processedWorkingImage;

  return {
    roi,
    imageData: finalImageData,
  };
}

async function fillMaskedPixels({
  pixels,
  mask,
  width,
  height,
  quality,
  strength,
  onProgress,
}: {
  pixels: Uint8ClampedArray;
  mask: Uint8Array;
  width: number;
  height: number;
  quality: ImageRepairQuality;
  strength: number;
  onProgress?: (progress: number, message: string) => void;
}) {
  const totalMasked = countMasked(mask);
  if (totalMasked === 0) {
    return;
  }

  const patchRadius = resolvePatchRadius(quality, strength);
  const searchRadius = resolveSearchRadius(quality, strength, width, height);
  const candidateStep = quality === 'hq' ? 1 : 2;
  const minKnownSamples = quality === 'hq' ? 12 : 6;
  const yieldEvery = quality === 'hq' ? 8 : 14;
  let remaining = totalMasked;

  while (remaining > 0) {
    const frontier: number[] = [];

    for (let index = 0; index < mask.length; index += 1) {
      if (mask[index] !== 0 && hasKnownNeighbor(mask, width, height, index)) {
        frontier.push(index);
      }
    }

    if (frontier.length === 0) {
      fillRemainingPixels(pixels, mask, width, height, searchRadius);
      return;
    }

    frontier.sort((left, right) => frontierPriority(mask, width, height, right) - frontierPriority(mask, width, height, left));

    let waveWrites = 0;

    for (let index = 0; index < frontier.length; index += 1) {
      const pixelIndex = frontier[index];
      if (mask[pixelIndex] === 0) {
        continue;
      }

      const x = pixelIndex % width;
      const y = Math.floor(pixelIndex / width);
      const match = findBestDonorPatch({
        pixels,
        donorPixels: pixels,
        mask,
        width,
        height,
        targetX: x,
        targetY: y,
        searchRadius,
        patchRadius,
        candidateStep,
        minKnownSamples,
      });

      let writes = 0;
      if (match) {
        writes = copyDonorPatchIntoMask({
          pixels,
          mask,
          donorPixels: pixels,
          width,
          height,
          targetX: x,
          targetY: y,
          donorX: match.x,
          donorY: match.y,
          patchRadius,
          adjustment: match.adjustment,
        });
      }

      if (writes === 0) {
        const fallback = averageKnownNeighbors(pixels, mask, width, height, x, y, patchRadius + 1);
        if (fallback) {
          const rgbaIndex = pixelIndex * 4;
          pixels[rgbaIndex] = clampByte(fallback.r);
          pixels[rgbaIndex + 1] = clampByte(fallback.g);
          pixels[rgbaIndex + 2] = clampByte(fallback.b);
          pixels[rgbaIndex + 3] = 255;
          mask[pixelIndex] = 0;
          writes = 1;
        }
      }

      if (writes > 0) {
        waveWrites += writes;
        remaining = Math.max(0, remaining - writes);
      }

      if ((index + 1) % yieldEvery === 0) {
        const completedRatio = 1 - remaining / totalMasked;
        onProgress?.(
          34 + Math.round(completedRatio * 30),
          completedRatio > 0.45 ? '正在按周围纹理逐步重建局部内容…' : '正在优先彻底清除水印内容…',
        );
        await waitForFrame();
      }
    }

    if (waveWrites === 0) {
      fillRemainingPixels(pixels, mask, width, height, searchRadius);
      return;
    }

    onProgress?.(34 + Math.round((1 - remaining / totalMasked) * 30), '正在按周围纹理逐步重建局部内容…');
    await waitForFrame();
  }
}

async function rebuildTexture({
  pixels,
  originalPixels,
  fillMask,
  width,
  height,
  quality,
  strength,
  onProgress,
}: {
  pixels: Uint8ClampedArray;
  originalPixels: Uint8ClampedArray;
  fillMask: Uint8Array;
  width: number;
  height: number;
  quality: ImageRepairQuality;
  strength: number;
  onProgress?: (progress: number, message: string) => void;
}) {
  const rebuilt = new Uint8ClampedArray(pixels);
  const targets: number[] = [];
  for (let index = 0; index < fillMask.length; index += 1) {
    if (fillMask[index] !== 0) {
      targets.push(index);
    }
  }

  const patchRadius = resolvePatchRadius(quality, strength);
  const searchRadius = resolveSearchRadius(quality, strength, width, height) + (quality === 'hq' ? 8 : 4);
  const candidateStep = quality === 'hq' ? 1 : 2;
  const minKnownSamples = quality === 'hq' ? 10 : 6;
  const yieldEvery = quality === 'hq' ? 18 : 28;

  for (let index = 0; index < targets.length; index += 1) {
    const pixelIndex = targets[index];
    const x = pixelIndex % width;
    const y = Math.floor(pixelIndex / width);
    const match = findBestDonorPatch({
      pixels: rebuilt,
      donorPixels: originalPixels,
      mask: fillMask,
      width,
      height,
      targetX: x,
      targetY: y,
      searchRadius,
      patchRadius,
      candidateStep,
      minKnownSamples,
    });

    if (match) {
      const rgbaIndex = pixelIndex * 4;
      const donorColor = sampleAdjustedDonorPixel(originalPixels, (match.y * width + match.x) * 4, match.adjustment);
      const alpha = quality === 'hq' ? 0.72 : 0.58;
      rebuilt[rgbaIndex] = blendChannel(rebuilt[rgbaIndex], donorColor.r, alpha);
      rebuilt[rgbaIndex + 1] = blendChannel(rebuilt[rgbaIndex + 1], donorColor.g, alpha);
      rebuilt[rgbaIndex + 2] = blendChannel(rebuilt[rgbaIndex + 2], donorColor.b, alpha);
      rebuilt[rgbaIndex + 3] = 255;
    }

    if ((index + 1) % yieldEvery === 0) {
      onProgress?.(72 + Math.round(((index + 1) / targets.length) * 14), '正在重建纹理细节并修正局部边缘…');
      await waitForFrame();
    }
  }

  const toneMatched = await matchFilledTone(rebuilt, originalPixels, fillMask, width, height, quality, onProgress);
  return repairOutlierArtifacts(toneMatched, fillMask, width, height, quality, onProgress);
}

async function generateOpenCvInpaintCandidate({
  originalPixels,
  fillMask,
  maskedPixelCount,
  width,
  height,
  quality,
  strength,
  onProgress,
}: {
  originalPixels: Uint8ClampedArray;
  fillMask: Uint8Array;
  maskedPixelCount: number;
  width: number;
  height: number;
  quality: ImageRepairQuality;
  strength: number;
  onProgress?: (progress: number, message: string) => void;
}) {
  if (!ENABLE_OPENCV_BROWSER_REFINEMENT) {
    return null;
  }

  try {
    onProgress?.(88, '正在补一层平滑重建，优化边界过渡…');
    if (!shouldUseOpenCvRefinement({ width, height, maskedPixelCount, quality })) {
      return null;
    }

    await yieldToBrowser();

    const cv = await promiseWithTimeout(loadOpenCv(), quality === 'hq' ? 18_000 : 8_000);
    if (!cv?.Mat || !cv?.inpaint || !cv?.cvtColor || typeof cv.matFromImageData !== 'function') {
      return null;
    }

    const sourceImage = new ImageData(new Uint8ClampedArray(originalPixels), width, height);
    const sourceRgba = cv.matFromImageData(sourceImage);
    const sourceRgb = new cv.Mat();
    const maskMat = new cv.Mat(height, width, cv.CV_8UC1);
    const resultRgb = new cv.Mat();
    const resultRgba = new cv.Mat();
    const navierRgb = new cv.Mat();
    const navierRgba = new cv.Mat();

    try {
      cv.cvtColor(sourceRgba, sourceRgb, cv.COLOR_RGBA2RGB);

      for (let index = 0; index < fillMask.length; index += 1) {
        maskMat.data[index] = fillMask[index] !== 0 ? 255 : 0;
      }

      const radius = quality === 'hq'
        ? clampNumber(2.4 + clampStrength(strength) / 45, 2.5, 4.8)
        : clampNumber(1.8 + clampStrength(strength) / 55, 1.8, 3.4);

      cv.inpaint(sourceRgb, maskMat, resultRgb, radius, cv.INPAINT_TELEA);
      cv.cvtColor(resultRgb, resultRgba, cv.COLOR_RGB2RGBA);

      const teleaPixels = new Uint8ClampedArray(resultRgba.data);
      const canRunNavierCandidate = quality === 'hq' && maskedPixelCount <= 48_000 && typeof cv.INPAINT_NS === 'number';
      if (!canRunNavierCandidate) {
        return teleaPixels;
      }

      cv.inpaint(sourceRgb, maskMat, navierRgb, Math.max(1.8, radius * 0.78), cv.INPAINT_NS);
      cv.cvtColor(navierRgb, navierRgba, cv.COLOR_RGB2RGBA);

      const blendedPixels = new Uint8ClampedArray(teleaPixels);
      const distanceToKnown = computeDistanceMap(fillMask, width, height, 0);
      for (let index = 0; index < fillMask.length; index += 1) {
        if (fillMask[index] === 0) {
          continue;
        }

        const rgbaIndex = index * 4;
        const alpha = 0.22 + 0.16 * smoothstep(1, 7, distanceToKnown[index]);
        blendedPixels[rgbaIndex] = blendChannel(teleaPixels[rgbaIndex], navierRgba.data[rgbaIndex], alpha);
        blendedPixels[rgbaIndex + 1] = blendChannel(teleaPixels[rgbaIndex + 1], navierRgba.data[rgbaIndex + 1], alpha);
        blendedPixels[rgbaIndex + 2] = blendChannel(teleaPixels[rgbaIndex + 2], navierRgba.data[rgbaIndex + 2], alpha);
        blendedPixels[rgbaIndex + 3] = 255;
      }

      return blendedPixels;
    } finally {
      sourceRgba.delete();
      sourceRgb.delete();
      maskMat.delete();
      resultRgb.delete();
      resultRgba.delete();
      navierRgb.delete();
      navierRgba.delete();
    }
  } catch {
    return null;
  }
}

async function refineSelectionMask({
  originalPixels,
  selectionMask,
  width,
  height,
  quality,
  onProgress,
}: {
  originalPixels: Uint8ClampedArray;
  selectionMask: Uint8Array;
  width: number;
  height: number;
  quality: ImageRepairQuality;
  onProgress?: (progress: number, message: string) => void;
}) {
  const bounds = getBinaryMaskBounds(selectionMask, width, height);
  if (!bounds) {
    return selectionMask;
  }

  const selectionArea = countMasked(selectionMask);
  const rectWidth = bounds.right - bounds.left + 1;
  const rectHeight = bounds.bottom - bounds.top + 1;
  const rectArea = rectWidth * rectHeight;
  const totalPixels = width * height;
  const fillRatio = selectionArea / Math.max(rectArea, 1);
  const isRectLike = fillRatio > 0.9;

  if (!isRectLike || selectionArea < 120 || rectWidth < 12 || rectHeight < 12) {
    return selectionMask;
  }

  if (!ENABLE_RECT_SELECTION_FOREGROUND_REFINEMENT) {
    return selectionMask;
  }

  const borderRefined = refineRectSelectionByBorderBackground({
    originalPixels,
    selectionMask,
    width,
    height,
    bounds,
    quality,
  });
  if (borderRefined) {
    return borderRefined;
  }

  if (!ENABLE_OPENCV_BROWSER_REFINEMENT) {
    return selectionMask;
  }

  const grabCutScale = resolveGrabCutScale(width, height, selectionArea, quality);
  if (grabCutScale <= 0) {
    return selectionMask;
  }

  try {
    onProgress?.(22, '正在细化框选前景，避免整块矩形一起被重绘…');
    await yieldToBrowser();

    const cv = await promiseWithTimeout(loadOpenCv(), 5000);
    if (!cv?.Mat || !cv?.grabCut || !cv?.Rect || typeof cv.matFromImageData !== 'function') {
      return selectionMask;
    }

    const sourceImage = new ImageData(new Uint8ClampedArray(originalPixels), width, height);
    const sourceForGrabCut = grabCutScale < 1
      ? resizeImageData(sourceImage, Math.max(1, Math.round(width * grabCutScale)), Math.max(1, Math.round(height * grabCutScale)), true)
      : sourceImage;
    const sourceRgba = cv.matFromImageData(sourceImage);
    const scaledSourceRgba = sourceForGrabCut === sourceImage ? sourceRgba : cv.matFromImageData(sourceForGrabCut);
    const sourceRgb = new cv.Mat();
    const scaledWidth = sourceForGrabCut.width;
    const scaledHeight = sourceForGrabCut.height;
    const grabCutMask = new cv.Mat(scaledHeight, scaledWidth, cv.CV_8UC1);
    const bgdModel = new cv.Mat();
    const fgdModel = new cv.Mat();

    try {
      cv.cvtColor(scaledSourceRgba, sourceRgb, cv.COLOR_RGBA2RGB);
      grabCutMask.setTo(new cv.Scalar(cv.GC_BGD));

      const scaledBounds = {
        left: Math.max(0, Math.round(bounds.left * grabCutScale)),
        top: Math.max(0, Math.round(bounds.top * grabCutScale)),
        right: Math.max(0, Math.round(bounds.right * grabCutScale)),
        bottom: Math.max(0, Math.round(bounds.bottom * grabCutScale)),
      };
      const rectX = Math.max(1, Math.min(scaledWidth - 2, scaledBounds.left));
      const rectY = Math.max(1, Math.min(scaledHeight - 2, scaledBounds.top));
      const safeRight = Math.max(rectX + 1, Math.min(scaledWidth - 2, scaledBounds.right));
      const safeBottom = Math.max(rectY + 1, Math.min(scaledHeight - 2, scaledBounds.bottom));
      const rect = new cv.Rect(rectX, rectY, safeRight - rectX + 1, safeBottom - rectY + 1);

      cv.grabCut(
        sourceRgb,
        grabCutMask,
        rect,
        bgdModel,
        fgdModel,
        resolveGrabCutIterations(totalPixels, selectionArea, quality),
        cv.GC_INIT_WITH_RECT,
      );

      const scaledForeground = new Uint8Array(scaledWidth * scaledHeight);
      for (let index = 0; index < scaledForeground.length; index += 1) {
        const value = grabCutMask.data[index];
        if (value === cv.GC_FGD || value === cv.GC_PR_FGD) {
          scaledForeground[index] = 1;
        }
      }

      const probableForeground = grabCutScale < 1
        ? resizeMaskArray(scaledForeground, scaledWidth, scaledHeight, width, height)
        : scaledForeground;
      for (let index = 0; index < probableForeground.length; index += 1) {
        if (selectionMask[index] === 0) {
          probableForeground[index] = 0;
        }
      }

      const refined = keepLargestConnectedComponent(probableForeground, width, height);
      const refinedArea = countMasked(refined);
      if (refinedArea < selectionArea * 0.04 || refinedArea > selectionArea * 0.92) {
        return selectionMask;
      }

      return dilateMask(refined, width, height, 1);
    } finally {
      sourceRgba.delete();
      if (scaledSourceRgba !== sourceRgba) {
        scaledSourceRgba.delete();
      }
      sourceRgb.delete();
      grabCutMask.delete();
      bgdModel.delete();
      fgdModel.delete();
    }
  } catch {
    return selectionMask;
  }
}

function keepLargestConnectedComponent(mask: Uint8Array, width: number, height: number) {
  const visited = new Uint8Array(mask.length);
  let largest: number[] = [];

  for (let index = 0; index < mask.length; index += 1) {
    if (mask[index] === 0 || visited[index] !== 0) {
      continue;
    }

    const queue = [index];
    const component: number[] = [];
    visited[index] = 1;

    while (queue.length > 0) {
      const currentIndex = queue.pop()!;
      component.push(currentIndex);
      const x = currentIndex % width;
      const y = Math.floor(currentIndex / width);

      for (let offsetY = -1; offsetY <= 1; offsetY += 1) {
        const sampleY = y + offsetY;
        if (sampleY < 0 || sampleY >= height) {
          continue;
        }

        for (let offsetX = -1; offsetX <= 1; offsetX += 1) {
          const sampleX = x + offsetX;
          if (sampleX < 0 || sampleX >= width || (offsetX === 0 && offsetY === 0)) {
            continue;
          }

          const sampleIndex = sampleY * width + sampleX;
          if (mask[sampleIndex] === 0 || visited[sampleIndex] !== 0) {
            continue;
          }

          visited[sampleIndex] = 1;
          queue.push(sampleIndex);
        }
      }
    }

    if (component.length > largest.length) {
      largest = component;
    }
  }

  const result = new Uint8Array(mask.length);
  for (const index of largest) {
    result[index] = 1;
  }

  return result;
}

function keepConnectedComponentsByArea(mask: Uint8Array, width: number, height: number, minArea: number) {
  const visited = new Uint8Array(mask.length);
  const result = new Uint8Array(mask.length);

  for (let index = 0; index < mask.length; index += 1) {
    if (mask[index] === 0 || visited[index] !== 0) {
      continue;
    }

    const queue = [index];
    const component: number[] = [];
    visited[index] = 1;

    while (queue.length > 0) {
      const currentIndex = queue.pop()!;
      component.push(currentIndex);
      const x = currentIndex % width;
      const y = Math.floor(currentIndex / width);

      for (let offsetY = -1; offsetY <= 1; offsetY += 1) {
        const sampleY = y + offsetY;
        if (sampleY < 0 || sampleY >= height) {
          continue;
        }

        for (let offsetX = -1; offsetX <= 1; offsetX += 1) {
          const sampleX = x + offsetX;
          if (sampleX < 0 || sampleX >= width || (offsetX === 0 && offsetY === 0)) {
            continue;
          }

          const sampleIndex = sampleY * width + sampleX;
          if (mask[sampleIndex] === 0 || visited[sampleIndex] !== 0) {
            continue;
          }

          visited[sampleIndex] = 1;
          queue.push(sampleIndex);
        }
      }
    }

    if (component.length < minArea) {
      continue;
    }

    for (const componentIndex of component) {
      result[componentIndex] = 1;
    }
  }

  return result;
}

function refineRectSelectionByBorderBackground({
  originalPixels,
  selectionMask,
  width,
  height,
  bounds,
  quality,
}: {
  originalPixels: Uint8ClampedArray;
  selectionMask: Uint8Array;
  width: number;
  height: number;
  bounds: MaskBounds;
  quality: ImageRepairQuality;
}) {
  const ringWidth = quality === 'hq' ? 4 : 3;
  const rectWidth = bounds.right - bounds.left + 1;
  const rectHeight = bounds.bottom - bounds.top + 1;
  const stripRatio = Math.max(rectWidth / Math.max(1, rectHeight), rectHeight / Math.max(1, rectWidth));
  const isLongTextStrip = stripRatio >= 3.2 && Math.max(rectWidth, rectHeight) >= 80;
  const backgroundMask = new Uint8Array(selectionMask.length);

  for (let y = bounds.top; y <= bounds.bottom; y += 1) {
    for (let x = bounds.left; x <= bounds.right; x += 1) {
      const index = y * width + x;
      if (selectionMask[index] === 0) {
        continue;
      }

      const borderDistance = Math.min(x - bounds.left, bounds.right - x, y - bounds.top, bounds.bottom - y);
      if (borderDistance < ringWidth) {
        backgroundMask[index] = 1;
      }
    }
  }

  const backgroundStats = computeStats(originalPixels, backgroundMask);
  if (backgroundStats.count < Math.max(24, ringWidth * 8)) {
    return null;
  }
  const selectionArea = countMasked(selectionMask);
  const outsideRing = subtractMasks(dilateMask(selectionMask, width, height, ringWidth), selectionMask);
  const outsideStats = computeStats(originalPixels, outsideRing);
  if (outsideStats.count >= Math.max(16, ringWidth * 8)) {
    const borderToOutsideDistance = colorDistance(backgroundStats.mean, outsideStats.mean);
    const borderStd = (backgroundStats.std.r + backgroundStats.std.g + backgroundStats.std.b) / 3;
    const borderLuma = luma(backgroundStats.mean.r, backgroundStats.mean.g, backgroundStats.mean.b);
    const outsideLuma = luma(outsideStats.mean.r, outsideStats.mean.g, outsideStats.mean.b);
    const borderMinimumChannel = Math.min(backgroundStats.mean.r, backgroundStats.mean.g, backgroundStats.mean.b);
    const tolerance = Math.max(
      quality === 'hq' ? 28 : 34,
      (backgroundStats.std.r + backgroundStats.std.g + backgroundStats.std.b + outsideStats.std.r + outsideStats.std.g + outsideStats.std.b) * 0.42,
    );
    const looksLikeSelectedOverlay =
      borderLuma - outsideLuma > 34 &&
      borderStd < 48 &&
      borderMinimumChannel > 145;
    if (borderToOutsideDistance > tolerance && looksLikeSelectedOverlay) {
      return null;
    }
  }

  const outsideReferenceAvailable = outsideStats.count >= Math.max(16, ringWidth * 8);
  const backgroundReference = isLongTextStrip && outsideReferenceAvailable ? outsideStats : backgroundStats;
  const backgroundReferenceStd = (backgroundReference.std.r + backgroundReference.std.g + backgroundReference.std.b) / 3;
  const backgroundReferenceLuma = luma(
    backgroundReference.mean.r,
    backgroundReference.mean.g,
    backgroundReference.mean.b,
  );

  const meanDistanceThreshold = Math.max(
    isLongTextStrip ? (quality === 'hq' ? 18 : 22) : (quality === 'hq' ? 24 : 30),
    (backgroundReference.std.r + backgroundReference.std.g + backgroundReference.std.b) * (isLongTextStrip ? 0.72 : 1.05),
  );
  const propagationThreshold = isLongTextStrip ? (quality === 'hq' ? 12 : 16) : (quality === 'hq' ? 18 : 22);
  const maxBackgroundDistance = meanDistanceThreshold * (isLongTextStrip ? 1.2 : 1.45);
  const visitedBackground = new Uint8Array(selectionMask.length);
  const queue: number[] = [];

  for (let y = bounds.top; y <= bounds.bottom; y += 1) {
    for (let x = bounds.left; x <= bounds.right; x += 1) {
      const index = y * width + x;
      if (backgroundMask[index] === 0) {
        continue;
      }

      if (colorDistanceToMean(originalPixels, index * 4, backgroundReference.mean) <= meanDistanceThreshold) {
        visitedBackground[index] = 1;
        queue.push(index);
      }
    }
  }

  if (queue.length === 0) {
    return null;
  }

  for (let queueIndex = 0; queueIndex < queue.length; queueIndex += 1) {
    const currentIndex = queue[queueIndex];
    const x = currentIndex % width;
    const y = Math.floor(currentIndex / width);

    for (let offsetY = -1; offsetY <= 1; offsetY += 1) {
      const sampleY = y + offsetY;
      if (sampleY < bounds.top || sampleY > bounds.bottom) {
        continue;
      }

      for (let offsetX = -1; offsetX <= 1; offsetX += 1) {
        const sampleX = x + offsetX;
        if (
          sampleX < bounds.left ||
          sampleX > bounds.right ||
          (offsetX === 0 && offsetY === 0)
        ) {
          continue;
        }

        const sampleIndex = sampleY * width + sampleX;
        if (selectionMask[sampleIndex] === 0 || visitedBackground[sampleIndex] !== 0) {
          continue;
        }

        const sampleDistance = colorDistanceToMean(originalPixels, sampleIndex * 4, backgroundReference.mean);
        if (sampleDistance > maxBackgroundDistance) {
          continue;
        }

        const continuityDistance = colorDistanceBetweenPixels(originalPixels, currentIndex * 4, sampleIndex * 4);
        if (continuityDistance > propagationThreshold && sampleDistance > meanDistanceThreshold) {
          continue;
        }

        visitedBackground[sampleIndex] = 1;
        queue.push(sampleIndex);
      }
    }
  }

  const foreground = new Uint8Array(selectionMask.length);
  const directForegroundThreshold = Math.max(
    isLongTextStrip ? (quality === 'hq' ? 8 : 11) : (quality === 'hq' ? 16 : 20),
    backgroundReferenceStd * (isLongTextStrip ? 0.42 : 0.72),
  );
  const lumaForegroundThreshold = isLongTextStrip ? (quality === 'hq' ? 5.5 : 7) : Number.POSITIVE_INFINITY;
  for (let index = 0; index < selectionMask.length; index += 1) {
    if (selectionMask[index] === 0) {
      continue;
    }

    const rgbaIndex = index * 4;
    const directDistance = colorDistanceToMean(originalPixels, rgbaIndex, backgroundReference.mean);
    const pixelLuma = luma(originalPixels[rgbaIndex], originalPixels[rgbaIndex + 1], originalPixels[rgbaIndex + 2]);
    const lumaDistance = Math.abs(pixelLuma - backgroundReferenceLuma);
    if (isLongTextStrip) {
      const x = index % width;
      const y = Math.floor(index / width);
      const localVariance = localLumaVariance(originalPixels, width, height, x, y, 1);
      const edgeLike = localVariance > (quality === 'hq' ? 9 : 13);
      const strongInk = directDistance > directForegroundThreshold * 2.2 || lumaDistance > lumaForegroundThreshold * 2.1;
      const softInk = edgeLike && (directDistance > directForegroundThreshold || lumaDistance > lumaForegroundThreshold);
      foreground[index] = strongInk || softInk || (visitedBackground[index] === 0 && edgeLike) ? 1 : 0;
      continue;
    }

    foreground[index] =
      visitedBackground[index] === 0 ||
      directDistance > directForegroundThreshold ||
      lumaDistance > lumaForegroundThreshold
        ? 1
        : 0;
  }

  const refined = keepConnectedComponentsByArea(
    foreground,
    width,
    height,
    Math.max(isLongTextStrip ? 1 : 3, Math.round(selectionArea * (isLongTextStrip ? 0.00025 : 0.0008))),
  );
  const refinedArea = countMasked(refined);

  if (refinedArea < selectionArea * (isLongTextStrip ? 0.002 : 0.015) || refinedArea > selectionArea * 0.88) {
    return null;
  }

  return dilateMask(refined, width, height, isLongTextStrip ? (quality === 'hq' ? 5 : 3) : (quality === 'hq' ? 3 : 2));
}

async function generateDiffusionCandidate({
  originalPixels,
  fillMask,
  maskedPixelCount,
  width,
  height,
  quality,
  onProgress,
}: {
  originalPixels: Uint8ClampedArray;
  fillMask: Uint8Array;
  maskedPixelCount: number;
  width: number;
  height: number;
  quality: ImageRepairQuality;
  onProgress?: (progress: number, message: string) => void;
}) {
  const current = new Uint8ClampedArray(originalPixels);
  const targets: number[] = [];
  const fallbackColor = averageBoundaryColor(originalPixels, fillMask, width, height);
  const initialRadius = quality === 'hq' ? 6 : 4;
  const directionalSearchRadius = quality === 'hq'
    ? Math.min(Math.max(width, height), 180)
    : Math.min(Math.max(width, height), 120);

  for (let index = 0; index < fillMask.length; index += 1) {
    if (fillMask[index] === 0) {
      continue;
    }

    targets.push(index);
    const x = index % width;
    const y = Math.floor(index / width);
    const average = averageKnownNeighbors(originalPixels, fillMask, width, height, x, y, initialRadius)
      ?? directionalKnownColor(originalPixels, fillMask, width, height, x, y, directionalSearchRadius)
      ?? fallbackColor;
    const rgbaIndex = index * 4;
    current[rgbaIndex] = clampByte(average.r);
    current[rgbaIndex + 1] = clampByte(average.g);
    current[rgbaIndex + 2] = clampByte(average.b);
    current[rgbaIndex + 3] = 255;
  }

  const iterations = quality === 'hq'
    ? (maskedPixelCount < 3_000 ? 18 : maskedPixelCount < 9_000 ? 12 : maskedPixelCount < 18_000 ? 8 : 4)
    : (maskedPixelCount < 2_500 ? 8 : maskedPixelCount < 7_000 ? 5 : 3);
  const yieldEvery = resolveLoopYieldEvery(maskedPixelCount, quality);
  const startedAt = performance.now();
  const maxDurationMs = quality === 'hq'
    ? (maskedPixelCount < 8_000 ? 3_200 : maskedPixelCount < 18_000 ? 2_400 : 1_600)
    : (maskedPixelCount < 6_000 ? 1_400 : 900);

  for (let iteration = 0; iteration < iterations; iteration += 1) {
    const next = new Uint8ClampedArray(current);

    for (let targetIndex = 0; targetIndex < targets.length; targetIndex += 1) {
      const index = targets[targetIndex];
      const x = index % width;
      const y = Math.floor(index / width);
      const diffused = diffusionNeighborColor(current, width, height, x, y, quality === 'hq' ? 2 : 1);
      if (!diffused) {
        continue;
      }

      const rgbaIndex = index * 4;
      const blendAlpha = iteration < Math.max(2, Math.round(iterations / 3))
        ? (quality === 'hq' ? 0.74 : 0.66)
        : (quality === 'hq' ? 0.58 : 0.48);
      next[rgbaIndex] = blendChannel(current[rgbaIndex], diffused.r, blendAlpha);
      next[rgbaIndex + 1] = blendChannel(current[rgbaIndex + 1], diffused.g, blendAlpha);
      next[rgbaIndex + 2] = blendChannel(current[rgbaIndex + 2], diffused.b, blendAlpha);
      next[rgbaIndex + 3] = 255;

      if ((targetIndex + 1) % yieldEvery === 0) {
        onProgress?.(89, '正在做平滑场重建，压掉块感和直线…');
        await yieldToBrowser();

        if (performance.now() - startedAt > maxDurationMs) {
          current.set(next);
          return current;
        }
      }
    }

    current.set(next);

    if (performance.now() - startedAt > maxDurationMs) {
      break;
    }
  }

  return current;
}

async function removeTransparentOverlayCandidate({
  originalPixels,
  selectionMask,
  width,
  height,
  quality,
  onProgress,
}: {
  originalPixels: Uint8ClampedArray;
  selectionMask: Uint8Array;
  width: number;
  height: number;
  quality: ImageRepairQuality;
  onProgress?: (progress: number, message: string) => void;
}) {
  const bounds = getBinaryMaskBounds(selectionMask, width, height);
  if (!bounds) {
    return null;
  }

  const selectionArea = countMasked(selectionMask);
  if (selectionArea < 120) {
    return null;
  }

  const rectWidth = bounds.right - bounds.left + 1;
  const rectHeight = bounds.bottom - bounds.top + 1;
  const rectArea = Math.max(1, rectWidth * rectHeight);
  const fillRatio = selectionArea / rectArea;
  const shapeLikeSelection = fillRatio < 0.78;
  const stripRatio = Math.max(rectWidth / Math.max(1, rectHeight), rectHeight / Math.max(1, rectWidth));
  if (stripRatio > 1.7) {
    return null;
  }

  const ringWidth = quality === 'hq' ? 5 : 3;
  const outsideRing = subtractMasks(dilateMask(selectionMask, width, height, ringWidth), selectionMask);
  const insideEdge = subtractMasks(selectionMask, erodeMask(selectionMask, width, height, Math.max(1, ringWidth - 1)));
  const outsideStats = computeStats(originalPixels, outsideRing);
  const selectedStats = computeStats(originalPixels, selectionMask);
  const insideEdgeStats = computeStats(originalPixels, insideEdge);
  if (outsideStats.count < 24 || selectedStats.count < 24) {
    return null;
  }

  const lowFrequencyBackgroundPixels = await generateDiffusionCandidate({
    originalPixels,
    fillMask: selectionMask,
    maskedPixelCount: selectionArea,
    width,
    height,
    quality,
    onProgress,
  });

  const selectedLuma = luma(selectedStats.mean.r, selectedStats.mean.g, selectedStats.mean.b);
  const outsideLuma = luma(outsideStats.mean.r, outsideStats.mean.g, outsideStats.mean.b);
  const selectedSaturation = computeMeanSaturation(originalPixels, selectionMask);
  const outsideSaturation = computeMeanSaturation(originalPixels, outsideRing);
  const selectedTexture = computeMeanLocalLumaVariance(originalPixels, selectionMask, width, height, quality === 'hq' ? 2 : 3);
  const outsideTexture = computeMeanLocalLumaVariance(originalPixels, outsideRing, width, height, quality === 'hq' ? 2 : 3);
  const lumaLift = selectedLuma - outsideLuma;
  const saturationDrop = outsideSaturation - selectedSaturation;
  const edgeLuma = luma(insideEdgeStats.mean.r, insideEdgeStats.mean.g, insideEdgeStats.mean.b);
  const edgeLift = edgeLuma - outsideLuma;
  const textureStillPresent = selectedTexture > 12 || selectedTexture > outsideTexture * 0.28;
  const liftLooksTranslucent = lumaLift < (quality === 'hq' ? 82 : 68) && edgeLift < (quality === 'hq' ? 92 : 76);
  const textureLooksPreserved = selectedTexture > outsideTexture * 0.36 && selectedTexture < outsideTexture * 2.2;
  const localOverlayStats = computeTransparentOverlayLocalStats(
    originalPixels,
    lowFrequencyBackgroundPixels,
    selectionMask,
  );
  if (
    !shapeLikeSelection &&
    stripRatio > 1.45 &&
    localOverlayStats.negativeRatio > (quality === 'hq' ? 0.012 : 0.018)
  ) {
    return null;
  }

  const localLooksLikeOverlay =
    localOverlayStats.positiveRatio > (quality === 'hq' ? 0.075 : 0.1) &&
    localOverlayStats.strongRatio > (quality === 'hq' ? 0.018 : 0.026) &&
    (shapeLikeSelection || localOverlayStats.negativeRatio < (quality === 'hq' ? 0.028 : 0.02)) &&
    localOverlayStats.meanPositiveLift > (quality === 'hq' ? 5.2 : 7) &&
    selectedLuma > 38;
  const looksLikeLightOverlay =
    (
      textureStillPresent &&
      textureLooksPreserved &&
      liftLooksTranslucent &&
      (lumaLift > (quality === 'hq' ? 3.2 : 4.5) || edgeLift > (quality === 'hq' ? 4.5 : 6) || saturationDrop > 2.4) &&
      selectedLuma > 38
    ) ||
    localLooksLikeOverlay;

  if (!looksLikeLightOverlay) {
    return null;
  }

  const signalRepairMask = shapeLikeSelection
    ? null
    : buildTransparentOverlayRepairMask({
        originalPixels,
        backgroundPixels: lowFrequencyBackgroundPixels,
        selectionMask,
        width,
        height,
        quality,
      });
  const repairMask = signalRepairMask ?? selectionMask;
  if (!repairMask) {
    return null;
  }

  const repairArea = countMasked(repairMask);
  const repairBounds = getBinaryMaskBounds(repairMask, width, height);
  if (!repairBounds || repairArea < 120) {
    return null;
  }

  const repairRingWidth = quality === 'hq' ? 5 : 3;
  const repairOutsideRing = subtractMasks(dilateMask(repairMask, width, height, repairRingWidth), repairMask);
  const repairInsideEdge = subtractMasks(
    repairMask,
    erodeMask(repairMask, width, height, Math.max(1, repairRingWidth - 1)),
  );
  const repairOutsideStats = computeStats(originalPixels, repairOutsideRing);
  const repairSelectedStats = computeStats(originalPixels, repairMask);
  const repairInsideEdgeStats = computeStats(originalPixels, repairInsideEdge);
  if (repairOutsideStats.count < 24 || repairSelectedStats.count < 24) {
    return null;
  }

  const repairBackgroundPixels = repairMask === selectionMask
    ? lowFrequencyBackgroundPixels
    : await generateDiffusionCandidate({
        originalPixels,
        fillMask: repairMask,
        maskedPixelCount: repairArea,
        width,
        height,
        quality,
        onProgress,
      });
  const repairRectWidth = repairBounds.right - repairBounds.left + 1;
  const repairRectHeight = repairBounds.bottom - repairBounds.top + 1;
  const repairFillRatio = repairArea / Math.max(1, repairRectWidth * repairRectHeight);
  const focusedShapeLikeSelection = shapeLikeSelection || repairFillRatio < 0.82;
  const repairSelectedLuma = luma(
    repairSelectedStats.mean.r,
    repairSelectedStats.mean.g,
    repairSelectedStats.mean.b,
  );
  const repairOutsideLuma = luma(
    repairOutsideStats.mean.r,
    repairOutsideStats.mean.g,
    repairOutsideStats.mean.b,
  );
  const repairSelectedSaturation = computeMeanSaturation(originalPixels, repairMask);
  const repairOutsideSaturation = computeMeanSaturation(originalPixels, repairOutsideRing);
  const repairLumaLift = repairSelectedLuma - repairOutsideLuma;
  const repairSaturationDrop = repairOutsideSaturation - repairSelectedSaturation;
  const repairEdgeLuma = luma(
    repairInsideEdgeStats.mean.r,
    repairInsideEdgeStats.mean.g,
    repairInsideEdgeStats.mean.b,
  );
  const repairEdgeLift = repairEdgeLuma - repairOutsideLuma;
  onProgress?.(34, '检测到透明水印，正在反向去除半透明叠层…');
  await yieldToBrowser();

  const output = new Uint8ClampedArray(originalPixels);
  const distanceInside = computeDistanceMap(repairMask, width, height, 0);
  const searchRadius = quality === 'hq'
    ? Math.min(Math.max(width, height), 220)
    : Math.min(Math.max(width, height), 140);
  const overlayColor = 252;
  const alphaFromMeanLift = Math.max(0, repairLumaLift) / Math.max(24, overlayColor - repairOutsideLuma);
  const alphaFromEdgeLift = Math.max(0, repairEdgeLift) / Math.max(24, overlayColor - repairOutsideLuma);
  const baseAlpha = clampNumber(
    alphaFromMeanLift * 1.02 +
      Math.max(0, alphaFromEdgeLift - alphaFromMeanLift) * 0.35 +
      Math.max(0, repairSaturationDrop) / 260,
    quality === 'hq' ? 0.08 : 0.06,
    quality === 'hq' ? 0.58 : 0.46,
  );
  const yieldEvery = resolveLoopYieldEvery(repairArea, quality);
  let processed = 0;

  for (let index = 0; index < repairMask.length; index += 1) {
    if (repairMask[index] === 0) {
      continue;
    }

    const x = index % width;
    const y = Math.floor(index / width);
    const rgbaIndex = index * 4;
    const observed = {
      r: originalPixels[rgbaIndex],
      g: originalPixels[rgbaIndex + 1],
      b: originalPixels[rgbaIndex + 2],
    };
    const boundaryBackground = interpolatedBoundaryColor(originalPixels, repairMask, width, height, x, y, searchRadius)
      ?? directionalKnownColor(originalPixels, repairMask, width, height, x, y, searchRadius)
      ?? repairOutsideStats.mean;
    const lowFrequencyIndex = index * 4;
    const diffusionBackground = {
      r: repairBackgroundPixels[lowFrequencyIndex],
      g: repairBackgroundPixels[lowFrequencyIndex + 1],
      b: repairBackgroundPixels[lowFrequencyIndex + 2],
    };
    const localBackground = mixRgbColors(
      boundaryBackground,
      diffusionBackground,
      focusedShapeLikeSelection ? 0.62 : 0.88,
    );
    const observedLuma = luma(observed.r, observed.g, observed.b);
    const localBackgroundLuma = luma(localBackground.r, localBackground.g, localBackground.b);
    const localLift = observedLuma - localBackgroundLuma;
    const localSaturationDrop =
      saturation(localBackground.r, localBackground.g, localBackground.b) -
      saturation(observed.r, observed.g, observed.b);
    const minimumChannelLift =
      Math.min(observed.r, observed.g, observed.b) -
      Math.min(localBackground.r, localBackground.g, localBackground.b);
    const overlaySignal =
      localLift +
      Math.max(0, localSaturationDrop) * 0.16 +
      Math.max(0, minimumChannelLift) * 0.1;
    const signalPresence = smoothstep(
      quality === 'hq' ? 0.8 : 1.4,
      quality === 'hq' ? 12 : 16,
      overlaySignal,
    );
    const overlayPresence = focusedShapeLikeSelection ? 1 : signalPresence;

    if (!focusedShapeLikeSelection && overlayPresence <= 0.025) {
      processed += 1;
      if (processed % yieldEvery === 0) {
        onProgress?.(72, 'Restoring transparent overlay texture...');
        await yieldToBrowser();
      }
      continue;
    }

    const localAlpha = clampNumber(
      focusedShapeLikeSelection
        ? (observedLuma - localBackgroundLuma) / Math.max(22, 252 - localBackgroundLuma)
        : Math.max(0, localLift) / Math.max(18, 252 - localBackgroundLuma) * 1.42 +
          Math.max(0, localSaturationDrop) / 340 +
          Math.max(0, minimumChannelLift) / 520,
      0,
      quality === 'hq' ? 0.66 : 0.54,
    );
    const edgeFactor = smoothstep(0.45, quality === 'hq' ? 5.5 : 3.8, distanceInside[index]);
    const baseAlphaWeight = 0.01 + edgeFactor * 0.98;
    const alphaBoost = shapeLikeSelection ? 0.84 : (focusedShapeLikeSelection ? 1.12 : 1.08);
    const alpha = clampNumber(
      (baseAlpha * baseAlphaWeight + localAlpha * (1 - baseAlphaWeight)) * alphaBoost,
      0.015,
      quality === 'hq' ? 0.68 : 0.54,
    );
    const inverse = 1 / Math.max(0.22, 1 - alpha);
    const restored = {
      r: clampByte((observed.r - overlayColor * alpha) * inverse),
      g: clampByte((observed.g - overlayColor * alpha) * inverse),
      b: clampByte((observed.b - overlayColor * alpha) * inverse),
    };
    const restoredLuma = luma(restored.r, restored.g, restored.b);
    const lowFrequencyCorrection = clampNumber((localBackgroundLuma - restoredLuma) * 0.12, -9, 9);
    const corrected = {
      r: clampByte(restored.r + lowFrequencyCorrection),
      g: clampByte(restored.g + lowFrequencyCorrection),
      b: clampByte(restored.b + lowFrequencyCorrection),
    };
    const observedSmooth = bilateralNeighborColor(originalPixels, width, height, x, y, quality === 'hq' ? 2 : 1, 72);
    const detailGain = clampNumber(0.82 + alpha * 0.78, 0.84, quality === 'hq' ? 1.36 : 1.24);
    const detailMatched = observedSmooth
      ? {
          r: clampByte(localBackground.r + (observed.r - observedSmooth.r) * detailGain),
          g: clampByte(localBackground.g + (observed.g - observedSmooth.g) * detailGain),
          b: clampByte(localBackground.b + (observed.b - observedSmooth.b) * detailGain),
        }
      : localBackground;
    const backgroundBlend = clampNumber(0.012 + edgeFactor * 0.022, 0.008, 0.04);
    const lowFrequencyMatched = mixRgbColors(corrected, detailMatched, backgroundBlend);
    const removalStrength = focusedShapeLikeSelection
      ? clampNumber(0.975 + edgeFactor * 0.015 + alpha * 0.02, 0.96, 0.998)
      : overlayPresence * clampNumber(0.985 + edgeFactor * 0.012 + alpha * 0.018, 0.96, 0.999);
    output[rgbaIndex] = blendChannel(observed.r, lowFrequencyMatched.r, removalStrength);
    output[rgbaIndex + 1] = blendChannel(observed.g, lowFrequencyMatched.g, removalStrength);
    output[rgbaIndex + 2] = blendChannel(observed.b, lowFrequencyMatched.b, removalStrength);
    output[rgbaIndex + 3] = 255;

    processed += 1;
    if (processed % yieldEvery === 0) {
      onProgress?.(72, '正在保留底图纹理并恢复透明水印下的对比度…');
      await yieldToBrowser();
    }
  }

  const harmonizedPixels = await harmonizeTransparentOverlayResult({
    pixels: output,
    originalPixels,
    selectionMask: repairMask,
    outsideStats: repairOutsideStats,
    width,
    height,
    quality,
    onProgress,
  });
  const harmonizedStats = computeStats(harmonizedPixels, repairMask);
  const harmonizedEdgeStats = computeStats(harmonizedPixels, repairInsideEdge);
  const harmonizedLuma = luma(harmonizedStats.mean.r, harmonizedStats.mean.g, harmonizedStats.mean.b);
  const harmonizedEdgeLuma = luma(harmonizedEdgeStats.mean.r, harmonizedEdgeStats.mean.g, harmonizedEdgeStats.mean.b);
  const removedLift = repairSelectedLuma - harmonizedLuma;
  const remainingByRemoval = repairLumaLift - removedLift;
  const remainingLift = Math.max(
    harmonizedLuma - repairOutsideLuma,
    harmonizedEdgeLuma - repairOutsideLuma,
    remainingByRemoval,
  );
  const remainingEdgeLift = harmonizedEdgeLuma - repairOutsideLuma;
  const residualSuppressedPixels =
    remainingLift < (quality === 'hq' ? 3.2 : 4.4) &&
    remainingEdgeLift < (quality === 'hq' ? 1.4 : 2)
      ? harmonizedPixels
      : await suppressTransparentOverlayResidual({
          pixels: harmonizedPixels,
          originalPixels,
          selectionMask: repairMask,
          lowFrequencyBackgroundPixels: repairBackgroundPixels,
          width,
          height,
          quality,
          onProgress,
        });

  return await suppressTransparentOverlayBoundaryHalo({
    pixels: residualSuppressedPixels,
    originalPixels,
    selectionMask: repairMask,
    width,
    height,
    quality,
    onProgress,
  });
}

async function mergeRepairCandidates({
  patchPixels,
  openCvPixels,
  originalPixels,
  fillMask,
  distanceToKnown,
  maskedPixelCount,
  width,
  height,
  quality,
  onProgress,
}: {
  patchPixels: Uint8ClampedArray;
  openCvPixels: Uint8ClampedArray | null;
  originalPixels: Uint8ClampedArray;
  fillMask: Uint8Array;
  distanceToKnown: Float32Array;
  maskedPixelCount: number;
  width: number;
  height: number;
  quality: ImageRepairQuality;
  onProgress?: (progress: number, message: string) => void;
}) {
  if (!openCvPixels) {
    return patchPixels;
  }

  const output = new Uint8ClampedArray(patchPixels);
  const boundaryDepth = quality === 'hq' ? 7 : 5;
  const yieldEvery = resolveLoopYieldEvery(maskedPixelCount, quality);
  let processed = 0;

  for (let index = 0; index < fillMask.length; index += 1) {
    if (fillMask[index] === 0) {
      continue;
    }

    const x = index % width;
    const y = Math.floor(index / width);
    const localVariance = maskedLumaVariance(originalPixels, fillMask, width, height, x, y, 3);
    const smoothFactor = localVariance === null
      ? 0.32
      : 1 - clampNumber(localVariance / (quality === 'hq' ? 1800 : 1400), 0, 1);
    const boundaryFactor = 1 - clampNumber((distanceToKnown[index] - 1) / boundaryDepth, 0, 1);
    const alpha = clampNumber(
      (quality === 'hq' ? 0.16 : 0.12) + boundaryFactor * 0.34 + smoothFactor * 0.22,
      0.12,
      quality === 'hq' ? 0.68 : 0.54,
    );

    const rgbaIndex = index * 4;
    output[rgbaIndex] = blendChannel(output[rgbaIndex], openCvPixels[rgbaIndex], alpha);
    output[rgbaIndex + 1] = blendChannel(output[rgbaIndex + 1], openCvPixels[rgbaIndex + 1], alpha);
    output[rgbaIndex + 2] = blendChannel(output[rgbaIndex + 2], openCvPixels[rgbaIndex + 2], alpha);
    output[rgbaIndex + 3] = 255;

    processed += 1;
    if (processed % yieldEvery === 0) {
      onProgress?.(90, '正在合并平滑基底，保持页面可响应…');
      await yieldToBrowser();
    }
  }

  return output;
}

async function mergeSmoothGradientCandidate({
  basePixels,
  smoothPixels,
  originalPixels,
  fillMask,
  distanceToKnown,
  maskedPixelCount,
  width,
  height,
  quality,
  preserveBaseDetail,
  onProgress,
}: {
  basePixels: Uint8ClampedArray;
  smoothPixels: Uint8ClampedArray;
  originalPixels: Uint8ClampedArray;
  fillMask: Uint8Array;
  distanceToKnown: Float32Array;
  maskedPixelCount: number;
  width: number;
  height: number;
  quality: ImageRepairQuality;
  preserveBaseDetail?: boolean;
  onProgress?: (progress: number, message: string) => void;
}) {
  const output = new Uint8ClampedArray(basePixels);
  const yieldEvery = resolveLoopYieldEvery(maskedPixelCount, quality);
  let processed = 0;

  for (let index = 0; index < fillMask.length; index += 1) {
    if (fillMask[index] === 0) {
      continue;
    }

    const x = index % width;
    const y = Math.floor(index / width);
    const localVariance = maskedLumaVariance(originalPixels, fillMask, width, height, x, y, quality === 'hq' ? 4 : 3);
    const smoothBias = localVariance === null
      ? 0.82
      : 1 - clampNumber(localVariance / (quality === 'hq' ? 2600 : 2000), 0, 1);
    const centerFactor = smoothstep(0.8, quality === 'hq' ? 6.4 : 4.8, distanceToKnown[index]);
    const texturePreserveFactor = preserveBaseDetail && localVariance !== null
      ? clampNumber(localVariance / (quality === 'hq' ? 1300 : 1000), 0, 0.68)
      : 0;
    const smoothWeight = preserveBaseDetail
      ? (quality === 'hq' ? 0.18 : 0.14)
      : (quality === 'hq' ? 0.58 : 0.48);
    const centerWeight = preserveBaseDetail ? 0.06 : 0.24;
    const alpha = clampNumber(
      ((quality === 'hq' ? 0.08 : 0.06) + smoothBias * smoothWeight + centerFactor * centerWeight) * (1 - texturePreserveFactor),
      0.06,
      preserveBaseDetail ? (quality === 'hq' ? 0.32 : 0.26) : (quality === 'hq' ? 0.92 : 0.82),
    );

    const rgbaIndex = index * 4;
    output[rgbaIndex] = blendChannel(output[rgbaIndex], smoothPixels[rgbaIndex], alpha);
    output[rgbaIndex + 1] = blendChannel(output[rgbaIndex + 1], smoothPixels[rgbaIndex + 1], alpha);
    output[rgbaIndex + 2] = blendChannel(output[rgbaIndex + 2], smoothPixels[rgbaIndex + 2], alpha);
    output[rgbaIndex + 3] = 255;

    processed += 1;
    if (processed % yieldEvery === 0) {
      onProgress?.(91, '正在压低锯齿和拼接条带…');
      await yieldToBrowser();
    }
  }

  return output;
}

async function harmonizeBoundaryTransition({
  pixels,
  originalPixels,
  fillMask,
  protectedMask,
  distanceToKnown,
  maskedPixelCount,
  width,
  height,
  quality,
  onProgress,
}: {
  pixels: Uint8ClampedArray;
  originalPixels: Uint8ClampedArray;
  fillMask: Uint8Array;
  protectedMask: Uint8Array;
  distanceToKnown: Float32Array;
  maskedPixelCount: number;
  width: number;
  height: number;
  quality: ImageRepairQuality;
  onProgress?: (progress: number, message: string) => void;
}) {
  const output = new Uint8ClampedArray(pixels);
  const maxDepth = quality === 'hq' ? 6 : 4;
  const yieldEvery = resolveLoopYieldEvery(maskedPixelCount, quality);
  let processed = 0;

  for (let index = 0; index < fillMask.length; index += 1) {
    if (fillMask[index] === 0) {
      continue;
    }

    const depth = distanceToKnown[index];
    if (depth > maxDepth) {
      continue;
    }

    const x = index % width;
    const y = Math.floor(index / width);
    const sampleRadius = Math.max(2, Math.min(maxDepth + 1, Math.round(depth) + 2));
    const boundaryAverage = averageKnownNeighbors(originalPixels, fillMask, width, height, x, y, sampleRadius);
    if (!boundaryAverage) {
      continue;
    }

    const boundaryFactor = 1 - clampNumber((depth - 1) / maxDepth, 0, 1);
    const protectedFactor = protectedMask[index] !== 0 ? 0.58 : 1;
    const alpha = clampNumber(
      (quality === 'hq' ? 0.14 : 0.1) + boundaryFactor * (quality === 'hq' ? 0.28 : 0.22) * protectedFactor,
      0.06,
      quality === 'hq' ? 0.34 : 0.26,
    );
    const rgbaIndex = index * 4;
    output[rgbaIndex] = blendChannel(output[rgbaIndex], boundaryAverage.r, alpha);
    output[rgbaIndex + 1] = blendChannel(output[rgbaIndex + 1], boundaryAverage.g, alpha);
    output[rgbaIndex + 2] = blendChannel(output[rgbaIndex + 2], boundaryAverage.b, alpha);
    output[rgbaIndex + 3] = 255;

    processed += 1;
    if (processed % yieldEvery === 0) {
      onProgress?.(92, '正在统一边界色调和过渡感…');
      await yieldToBrowser();
    }
  }

  return output;
}

async function smoothFilledRegion({
  pixels,
  fillMask,
  distanceToKnown,
  maskedPixelCount,
  width,
  height,
  quality,
  onProgress,
}: {
  pixels: Uint8ClampedArray;
  fillMask: Uint8Array;
  distanceToKnown: Float32Array;
  maskedPixelCount: number;
  width: number;
  height: number;
  quality: ImageRepairQuality;
  onProgress?: (progress: number, message: string) => void;
}) {
  const iterations = quality === 'hq' && maskedPixelCount < 45_000 ? 2 : 1;
  const maxSmoothDepth = maskedPixelCount > (quality === 'hq' ? 60_000 : 36_000) ? (quality === 'hq' ? 5 : 4) : Number.POSITIVE_INFINITY;
  const radius = quality === 'hq' && maskedPixelCount < 40_000 ? 2 : 1;
  const sigmaColor = quality === 'hq' ? 46 : 38;
  const yieldEvery = resolveLoopYieldEvery(maskedPixelCount, quality);
  let current = new Uint8ClampedArray(pixels);

  for (let iteration = 0; iteration < iterations; iteration += 1) {
    const next = new Uint8ClampedArray(current);
    let processed = 0;

    for (let index = 0; index < fillMask.length; index += 1) {
      if (fillMask[index] === 0) {
        continue;
      }

      if (distanceToKnown[index] > maxSmoothDepth) {
        continue;
      }

      const x = index % width;
      const y = Math.floor(index / width);
      const smoothed = bilateralNeighborColor(current, width, height, x, y, radius, sigmaColor);
      const diffused = diffusionNeighborColor(current, width, height, x, y, 1);
      if (!smoothed && !diffused) {
        continue;
      }

      const variance = localLumaVariance(current, width, height, x, y, 1);
      const smoothFactor = 1 - clampNumber(variance / (quality === 'hq' ? 1500 : 1200), 0, 1);
      const boundaryFactor = 1 - clampNumber((distanceToKnown[index] - 1) / (quality === 'hq' ? 6 : 4), 0, 1);
      const diffusionWeight = clampNumber(smoothFactor * 0.72 + boundaryFactor * 0.18, 0, 0.9);
      const targetColor = smoothed && diffused
        ? mixRgbColors(smoothed, diffused, diffusionWeight)
        : (diffused ?? smoothed);
      if (!targetColor) {
        continue;
      }
      const alpha = clampNumber(
        ((quality === 'hq' ? 0.14 : 0.1) + smoothFactor * 0.2 + boundaryFactor * 0.12) * (iteration === 0 ? 1 : 0.72),
        0.06,
        quality === 'hq' ? 0.4 : 0.3,
      );

      const rgbaIndex = index * 4;
      next[rgbaIndex] = blendChannel(current[rgbaIndex], targetColor.r, alpha);
      next[rgbaIndex + 1] = blendChannel(current[rgbaIndex + 1], targetColor.g, alpha);
      next[rgbaIndex + 2] = blendChannel(current[rgbaIndex + 2], targetColor.b, alpha);
      next[rgbaIndex + 3] = 255;

      processed += 1;
      if (processed % yieldEvery === 0) {
        onProgress?.(95, '正在清理局部锯齿和生硬线条…');
        await yieldToBrowser();
      }
    }

    current = next;
  }

  return current;
}

async function reinjectDirectionalTexture({
  pixels,
  originalPixels,
  fillMask,
  distanceToKnown,
  maskedPixelCount,
  width,
  height,
  quality,
  onProgress,
}: {
  pixels: Uint8ClampedArray;
  originalPixels: Uint8ClampedArray;
  fillMask: Uint8Array;
  distanceToKnown: Float32Array;
  maskedPixelCount: number;
  width: number;
  height: number;
  quality: ImageRepairQuality;
  onProgress?: (progress: number, message: string) => void;
}) {
  const output = new Uint8ClampedArray(pixels);
  const searchRadius = quality === 'hq'
    ? Math.min(Math.max(width, height), 220)
    : Math.min(Math.max(width, height), 140);
  const yieldEvery = resolveLoopYieldEvery(maskedPixelCount, quality);
  let processed = 0;

  for (let index = 0; index < fillMask.length; index += 1) {
    if (fillMask[index] === 0) {
      continue;
    }

    const x = index % width;
    const y = Math.floor(index / width);
    const directional = directionalKnownColor(originalPixels, fillMask, width, height, x, y, searchRadius);
    if (!directional) {
      continue;
    }

    const localVariance = maskedLumaVariance(originalPixels, fillMask, width, height, x, y, quality === 'hq' ? 5 : 4);
    const textureFactor = localVariance === null
      ? 0.12
      : clampNumber(localVariance / (quality === 'hq' ? 1200 : 1500), 0.08, 0.74);
    const centerFactor = smoothstep(1.2, quality === 'hq' ? 7.5 : 5.5, distanceToKnown[index]);
    const alpha = clampNumber(
      (quality === 'hq' ? 0.035 : 0.025) + textureFactor * (quality === 'hq' ? 0.08 : 0.06) + centerFactor * 0.035,
      quality === 'hq' ? 0.03 : 0.02,
      quality === 'hq' ? 0.14 : 0.1,
    );

    const rgbaIndex = index * 4;
    output[rgbaIndex] = blendChannel(output[rgbaIndex], directional.r, alpha);
    output[rgbaIndex + 1] = blendChannel(output[rgbaIndex + 1], directional.g, alpha);
    output[rgbaIndex + 2] = blendChannel(output[rgbaIndex + 2], directional.b, alpha);
    output[rgbaIndex + 3] = 255;

    processed += 1;
    if (processed % yieldEvery === 0) {
      onProgress?.(96, '正在回填邻域纹理，让修复区域贴近周围背景…');
      await yieldToBrowser();
    }
  }

  return output;
}

async function suppressMaskBoundaryEcho({
  pixels,
  originalPixels,
  fillMask,
  baseMask,
  distanceInsideSelection,
  distanceToSelection,
  maskedPixelCount,
  width,
  height,
  quality,
  onProgress,
}: {
  pixels: Uint8ClampedArray;
  originalPixels: Uint8ClampedArray;
  fillMask: Uint8Array;
  baseMask: Uint8Array;
  distanceInsideSelection: Float32Array;
  distanceToSelection: Float32Array;
  maskedPixelCount: number;
  width: number;
  height: number;
  quality: ImageRepairQuality;
  onProgress?: (progress: number, message: string) => void;
}) {
  const output = new Uint8ClampedArray(pixels);
  const bandDepth = quality === 'hq' ? 5.5 : 3.8;
  const searchRadius = quality === 'hq'
    ? Math.min(Math.max(width, height), 220)
    : Math.min(Math.max(width, height), 140);
  const yieldEvery = resolveLoopYieldEvery(maskedPixelCount, quality);
  let processed = 0;

  for (let index = 0; index < fillMask.length; index += 1) {
    if (fillMask[index] === 0) {
      continue;
    }

    const insideSelection = baseMask[index] !== 0;
    const distanceToHardEdge = insideSelection ? distanceInsideSelection[index] : distanceToSelection[index];
    if (distanceToHardEdge > bandDepth) {
      continue;
    }

    const x = index % width;
    const y = Math.floor(index / width);
    const directional = directionalKnownColor(originalPixels, fillMask, width, height, x, y, searchRadius);
    if (!directional) {
      continue;
    }

    const edgeStrength = 1 - smoothstep(0.8, bandDepth, distanceToHardEdge + (hashUnit(x + 29, y - 23) - 0.5) * 0.9);
    const insideMaxAlpha = quality === 'hq' ? 0.3 : 0.24;
    const outsideMaxAlpha = quality === 'hq' ? 0.16 : 0.12;
    const alpha = clampNumber(
      edgeStrength * (insideSelection ? insideMaxAlpha : outsideMaxAlpha),
      0.02,
      insideSelection ? insideMaxAlpha : outsideMaxAlpha,
    );

    const rgbaIndex = index * 4;
    output[rgbaIndex] = blendChannel(output[rgbaIndex], directional.r, alpha);
    output[rgbaIndex + 1] = blendChannel(output[rgbaIndex + 1], directional.g, alpha);
    output[rgbaIndex + 2] = blendChannel(output[rgbaIndex + 2], directional.b, alpha);
    output[rgbaIndex + 3] = 255;

    processed += 1;
    if (processed % yieldEvery === 0) {
      onProgress?.(96, '正在压制选区边界回声，避免留下矩形线…');
      await yieldToBrowser();
    }
  }

  return output;
}

async function matchFilledTone(
  pixels: Uint8ClampedArray,
  originalPixels: Uint8ClampedArray,
  fillMask: Uint8Array,
  width: number,
  height: number,
  quality: ImageRepairQuality,
  onProgress?: (progress: number, message: string) => void,
) {
  const output = new Uint8ClampedArray(pixels);
  const outerRing = subtractMasks(dilateMask(fillMask, width, height, quality === 'hq' ? 2 : 1), fillMask);
  const innerCore = erodeMask(fillMask, width, height, 1);
  const innerEdge = subtractMasks(fillMask, innerCore);
  const targetStats = computeStats(originalPixels, outerRing);
  const fillStats = computeStats(output, innerEdge);

  if (targetStats.count < 12 || fillStats.count < 12) {
    return output;
  }

  const gains = {
    r: clampNumber(targetStats.std.r / Math.max(fillStats.std.r, 1), 0.88, 1.16),
    g: clampNumber(targetStats.std.g / Math.max(fillStats.std.g, 1), 0.88, 1.16),
    b: clampNumber(targetStats.std.b / Math.max(fillStats.std.b, 1), 0.88, 1.16),
  };
  const offsets = {
    r: targetStats.mean.r - fillStats.mean.r * gains.r,
    g: targetStats.mean.g - fillStats.mean.g * gains.g,
    b: targetStats.mean.b - fillStats.mean.b * gains.b,
  };
  const alpha = quality === 'hq' ? 0.48 : 0.34;
  const maskedPixelCount = countMasked(fillMask);
  const yieldEvery = resolveLoopYieldEvery(maskedPixelCount, quality);
  let processed = 0;

  for (let index = 0; index < fillMask.length; index += 1) {
    if (fillMask[index] === 0) {
      continue;
    }

    const rgbaIndex = index * 4;
    output[rgbaIndex] = blendChannel(output[rgbaIndex], clampByte(output[rgbaIndex] * gains.r + offsets.r), alpha);
    output[rgbaIndex + 1] = blendChannel(output[rgbaIndex + 1], clampByte(output[rgbaIndex + 1] * gains.g + offsets.g), alpha);
    output[rgbaIndex + 2] = blendChannel(output[rgbaIndex + 2], clampByte(output[rgbaIndex + 2] * gains.b + offsets.b), alpha);
    processed += 1;
    if (processed % yieldEvery === 0) {
      onProgress?.(84, '正在校准局部色调，避免修复区发假…');
      await yieldToBrowser();
    }
  }

  return output;
}

async function alignFilledRegionToBoundaryField({
  pixels,
  originalPixels,
  fillMask,
  distanceToKnown,
  maskedPixelCount,
  width,
  height,
  quality,
  onProgress,
}: {
  pixels: Uint8ClampedArray;
  originalPixels: Uint8ClampedArray;
  fillMask: Uint8Array;
  distanceToKnown: Float32Array;
  maskedPixelCount: number;
  width: number;
  height: number;
  quality: ImageRepairQuality;
  onProgress?: (progress: number, message: string) => void;
}) {
  const output = new Uint8ClampedArray(pixels);
  const searchRadius = quality === 'hq'
    ? Math.min(Math.max(width, height), 260)
    : Math.min(Math.max(width, height), 160);
  const yieldEvery = resolveLoopYieldEvery(maskedPixelCount, quality);
  let processed = 0;

  for (let index = 0; index < fillMask.length; index += 1) {
    if (fillMask[index] === 0) {
      continue;
    }

    const x = index % width;
    const y = Math.floor(index / width);
    const boundaryField = interpolatedBoundaryColor(originalPixels, fillMask, width, height, x, y, searchRadius)
      ?? directionalKnownColor(originalPixels, fillMask, width, height, x, y, searchRadius);
    if (!boundaryField) {
      continue;
    }

    const localVariance = maskedLumaVariance(originalPixels, fillMask, width, height, x, y, quality === 'hq' ? 6 : 4);
    const smoothArea = localVariance === null
      ? 0.72
      : 1 - clampNumber(localVariance / (quality === 'hq' ? 1900 : 1500), 0, 1);
    const boundaryFactor = 1 - smoothstep(0.8, quality === 'hq' ? 9.5 : 6.5, distanceToKnown[index]);
    const centerFactor = smoothstep(4, quality === 'hq' ? 18 : 12, distanceToKnown[index]);
    const alpha = clampNumber(
      (quality === 'hq' ? 0.16 : 0.12)
        + boundaryFactor * (quality === 'hq' ? 0.3 : 0.22)
        + smoothArea * (quality === 'hq' ? 0.18 : 0.13)
        + centerFactor * (quality === 'hq' ? 0.16 : 0.11),
      quality === 'hq' ? 0.08 : 0.06,
      quality === 'hq' ? 0.76 : 0.58,
    );

    const rgbaIndex = index * 4;
    output[rgbaIndex] = blendChannel(output[rgbaIndex], boundaryField.r, alpha);
    output[rgbaIndex + 1] = blendChannel(output[rgbaIndex + 1], boundaryField.g, alpha);
    output[rgbaIndex + 2] = blendChannel(output[rgbaIndex + 2], boundaryField.b, alpha);
    output[rgbaIndex + 3] = 255;

    processed += 1;
    if (processed % yieldEvery === 0) {
      onProgress?.(96, '正在连续匹配周围色场，弱化可见矩形区域…');
      await yieldToBrowser();
    }
  }

  return output;
}

async function restoreLinearBoundaryFeatures({
  pixels,
  originalPixels,
  fillMask,
  distanceToKnown,
  maskedPixelCount,
  width,
  height,
  quality,
  onProgress,
}: {
  pixels: Uint8ClampedArray;
  originalPixels: Uint8ClampedArray;
  fillMask: Uint8Array;
  distanceToKnown: Float32Array;
  maskedPixelCount: number;
  width: number;
  height: number;
  quality: ImageRepairQuality;
  onProgress?: (progress: number, message: string) => void;
}) {
  const output = new Uint8ClampedArray(pixels);
  const searchRadius = quality === 'hq'
    ? Math.min(Math.max(width, height), 260)
    : Math.min(Math.max(width, height), 160);
  const yieldEvery = resolveLoopYieldEvery(maskedPixelCount, quality);
  let processed = 0;

  for (let index = 0; index < fillMask.length; index += 1) {
    if (fillMask[index] === 0) {
      continue;
    }

    const x = index % width;
    const y = Math.floor(index / width);
    const feature = findLinearFeatureColor(pixels, originalPixels, fillMask, width, height, x, y, searchRadius, quality);
    if (!feature) {
      continue;
    }

    const depthFactor = 0.72 + smoothstep(1.2, quality === 'hq' ? 8 : 5.5, distanceToKnown[index]) * 0.28;
    const alpha = clampNumber(feature.strength * depthFactor, 0.08, quality === 'hq' ? 0.68 : 0.52);
    const rgbaIndex = index * 4;
    output[rgbaIndex] = blendChannel(output[rgbaIndex], feature.color.r, alpha);
    output[rgbaIndex + 1] = blendChannel(output[rgbaIndex + 1], feature.color.g, alpha);
    output[rgbaIndex + 2] = blendChannel(output[rgbaIndex + 2], feature.color.b, alpha);
    output[rgbaIndex + 3] = 255;

    processed += 1;
    if (processed % yieldEvery === 0) {
      onProgress?.(96, '正在延续邻域线条和边缘结构，减少矩形断层…');
      await yieldToBrowser();
    }
  }

  return output;
}

async function regrainFilledRegion({
  pixels,
  originalPixels,
  fillMask,
  distanceToKnown,
  maskedPixelCount,
  width,
  height,
  quality,
  onProgress,
}: {
  pixels: Uint8ClampedArray;
  originalPixels: Uint8ClampedArray;
  fillMask: Uint8Array;
  distanceToKnown: Float32Array;
  maskedPixelCount: number;
  width: number;
  height: number;
  quality: ImageRepairQuality;
  onProgress?: (progress: number, message: string) => void;
}) {
  const output = new Uint8ClampedArray(pixels);
  const textureRing = subtractMasks(dilateMask(fillMask, width, height, quality === 'hq' ? 4 : 3), fillMask);
  const textureStats = computeStats(originalPixels, textureRing);
  const fallbackVariance = textureStats.count > 0
    ? ((textureStats.std.r + textureStats.std.g + textureStats.std.b) / 3) ** 2
    : 0;
  const yieldEvery = resolveLoopYieldEvery(maskedPixelCount, quality);
  let processed = 0;

  for (let index = 0; index < fillMask.length; index += 1) {
    if (fillMask[index] === 0) {
      continue;
    }

    const x = index % width;
    const y = Math.floor(index / width);
    const localVariance = maskedLumaVariance(originalPixels, fillMask, width, height, x, y, quality === 'hq' ? 5 : 4);
    const textureVariance = localVariance ?? fallbackVariance;
    if (textureVariance <= 0) {
      continue;
    }

    const boundaryFade = 0.62 + smoothstep(1.2, quality === 'hq' ? 9 : 6, distanceToKnown[index]) * 0.38;
    const amplitude = clampNumber(
      Math.sqrt(textureVariance) * (quality === 'hq' ? 0.13 : 0.1) * boundaryFade,
      0.45,
      quality === 'hq' ? 5.8 : 4.2,
    );
    const noise =
      valueNoise2d(x * 0.34, y * 0.34, 13) * 0.64
      + valueNoise2d(x * 0.72, y * 0.72, 37) * 0.26
      + (hashUnit(x + 101, y - 53) - 0.5) * 0.2;
    const delta = clampNumber(noise, -1, 1) * amplitude;
    const rgbaIndex = index * 4;
    output[rgbaIndex] = clampByte(output[rgbaIndex] + delta);
    output[rgbaIndex + 1] = clampByte(output[rgbaIndex + 1] + delta);
    output[rgbaIndex + 2] = clampByte(output[rgbaIndex + 2] + delta);
    output[rgbaIndex + 3] = 255;

    processed += 1;
    if (processed % yieldEvery === 0) {
      onProgress?.(96, '正在匹配周围细微颗粒，避免处理区过度平滑…');
      await yieldToBrowser();
    }
  }

  return output;
}

async function cleanupResidualSelectionForeground({
  pixels,
  originalPixels,
  selectionMask,
  baseMask,
  maskedPixelCount,
  width,
  height,
  quality,
  onProgress,
}: {
  pixels: Uint8ClampedArray;
  originalPixels: Uint8ClampedArray;
  selectionMask: Uint8Array;
  baseMask: Uint8Array;
  maskedPixelCount: number;
  width: number;
  height: number;
  quality: ImageRepairQuality;
  onProgress?: (progress: number, message: string) => void;
}) {
  const bounds = getBinaryMaskBounds(selectionMask, width, height);
  if (!bounds) {
    return pixels;
  }

  const selectionArea = countMasked(selectionMask);
  if (selectionArea < 24) {
    return pixels;
  }

  const rectWidth = bounds.right - bounds.left + 1;
  const rectHeight = bounds.bottom - bounds.top + 1;
  const stripRatio = Math.max(rectWidth / Math.max(1, rectHeight), rectHeight / Math.max(1, rectWidth));
  const isLongTextStrip = stripRatio >= 3.2 && Math.max(rectWidth, rectHeight) >= 80;
  const ringSize = quality === 'hq' ? 4 : 3;
  const outsideRing = subtractMasks(dilateMask(selectionMask, width, height, ringSize), selectionMask);
  const borderRing = subtractMasks(selectionMask, erodeMask(selectionMask, width, height, ringSize));
  const outsideStats = computeStats(pixels, outsideRing);
  const borderStats = computeStats(pixels, borderRing);
  const backgroundStats = outsideStats.count >= Math.max(16, ringSize * 8) ? outsideStats : borderStats;
  if (backgroundStats.count < 12) {
    return pixels;
  }

  const backgroundStd = (backgroundStats.std.r + backgroundStats.std.g + backgroundStats.std.b) / 3;
  const backgroundLuma = luma(backgroundStats.mean.r, backgroundStats.mean.g, backgroundStats.mean.b);
  const colorThreshold = Math.max(
    isLongTextStrip ? (quality === 'hq' ? 2.6 : 4.2) : (quality === 'hq' ? 13 : 17),
    backgroundStd * (isLongTextStrip ? 0.14 : 0.58),
  );
  const lumaThreshold = isLongTextStrip ? (quality === 'hq' ? 1.8 : 2.8) : (quality === 'hq' ? 9 : 12);
  const residualMask = new Uint8Array(selectionMask.length);
  const yieldEvery = resolveLoopYieldEvery(maskedPixelCount, quality);
  let residualCount = 0;
  let processed = 0;

  for (let index = 0; index < selectionMask.length; index += 1) {
    if (selectionMask[index] === 0) {
      continue;
    }

    const rgbaIndex = index * 4;
    const originalDistance = colorDistanceToMean(originalPixels, rgbaIndex, backgroundStats.mean);
    const currentDistance = colorDistanceToMean(pixels, rgbaIndex, backgroundStats.mean);
    const originalLuma = luma(originalPixels[rgbaIndex], originalPixels[rgbaIndex + 1], originalPixels[rgbaIndex + 2]);
    const currentLuma = luma(pixels[rgbaIndex], pixels[rgbaIndex + 1], pixels[rgbaIndex + 2]);
    const originalLooksLikeText =
      originalDistance > colorThreshold ||
      Math.abs(originalLuma - backgroundLuma) > lumaThreshold;
    const currentStillVisible =
      currentDistance > colorThreshold * 0.72 ||
      Math.abs(currentLuma - backgroundLuma) > lumaThreshold * 0.72;
    const currentEdgeLike = isLongTextStrip
      ? localLumaVariance(pixels, width, height, index % width, Math.floor(index / width), 1) > (quality === 'hq' ? 4.5 : 7)
      : false;
    const underOriginalMask = baseMask[index] !== 0;

    if (
      originalLooksLikeText && (currentStillVisible || underOriginalMask || currentEdgeLike) ||
      isLongTextStrip && currentEdgeLike
    ) {
      residualMask[index] = 1;
      residualCount += 1;
    }

    processed += 1;
    if (processed % yieldEvery === 0) {
      onProgress?.(98, '正在二次扫描残留文字，补清淡色笔画…');
      await yieldToBrowser();
    }
  }

  const maxResidualRatio = isLongTextStrip ? 0.95 : 0.32;
  if (residualCount < Math.max(2, Math.round(selectionArea * (isLongTextStrip ? 0.0006 : 0.002)))) {
    return pixels;
  }

  if (!isLongTextStrip && residualCount > selectionArea * maxResidualRatio) {
    return pixels;
  }

  const expandedResidual = isLongTextStrip && residualCount > selectionArea * 0.58
    ? new Uint8Array(selectionMask)
    : dilateMask(residualMask, width, height, isLongTextStrip ? (quality === 'hq' ? 5 : 3) : (quality === 'hq' ? 2 : 1));
  const expandedResidualCount = countMasked(expandedResidual);
  if (expandedResidualCount === 0 || (!isLongTextStrip && expandedResidualCount > selectionArea * 0.46)) {
    return pixels;
  }

  const repairedPixels = await generateDiffusionCandidate({
    originalPixels: pixels,
    fillMask: expandedResidual,
    maskedPixelCount: expandedResidualCount,
    width,
    height,
    quality,
    onProgress,
  });
  const distanceToResidual = computeDistanceMap(expandedResidual, width, height, 0);
  const output = new Uint8ClampedArray(pixels);
  processed = 0;

  for (let index = 0; index < expandedResidual.length; index += 1) {
    if (expandedResidual[index] === 0) {
      continue;
    }

    const directResidual = residualMask[index] !== 0;
    const alpha = directResidual
      ? (quality === 'hq' ? 0.995 : 0.975)
      : clampNumber(0.52 * (1 - smoothstep(1, isLongTextStrip ? 4.5 : 3.2, distanceToResidual[index])), 0.12, 0.52);
    const rgbaIndex = index * 4;
    output[rgbaIndex] = blendChannel(output[rgbaIndex], repairedPixels[rgbaIndex], alpha);
    output[rgbaIndex + 1] = blendChannel(output[rgbaIndex + 1], repairedPixels[rgbaIndex + 1], alpha);
    output[rgbaIndex + 2] = blendChannel(output[rgbaIndex + 2], repairedPixels[rgbaIndex + 2], alpha);
    output[rgbaIndex + 3] = 255;

    processed += 1;
    if (processed % yieldEvery === 0) {
      onProgress?.(99, '正在用邻域像素覆盖残留笔画…');
      await yieldToBrowser();
    }
  }

  return output;
}

async function harmonizeTransparentOverlayResult({
  pixels,
  originalPixels,
  selectionMask,
  outsideStats,
  width,
  height,
  quality,
  onProgress,
}: {
  pixels: Uint8ClampedArray;
  originalPixels: Uint8ClampedArray;
  selectionMask: Uint8Array;
  outsideStats: RegionStats;
  width: number;
  height: number;
  quality: ImageRepairQuality;
  onProgress?: (progress: number, message: string) => void;
}) {
  const output = new Uint8ClampedArray(pixels);
  const selectedStats = computeStats(output, selectionMask);
  if (selectedStats.count < 12 || outsideStats.count < 12) {
    return output;
  }

  const selectedLuma = luma(selectedStats.mean.r, selectedStats.mean.g, selectedStats.mean.b);
  const outsideLuma = luma(outsideStats.mean.r, outsideStats.mean.g, outsideStats.mean.b);
  const lumaOffset = clampNumber(outsideLuma - selectedLuma, -18, 18);
  const distanceInside = computeDistanceMap(selectionMask, width, height, 0);
  const selectedCount = countMasked(selectionMask);
  const yieldEvery = resolveLoopYieldEvery(selectedCount, quality);
  let processed = 0;

  for (let index = 0; index < selectionMask.length; index += 1) {
    if (selectionMask[index] === 0) {
      continue;
    }

    const edgeFactor = smoothstep(0.6, quality === 'hq' ? 7 : 4.5, distanceInside[index]);
    const toneAlpha = clampNumber((1 - edgeFactor) * 0.16 + 0.1, 0.08, 0.22);
    const rgbaIndex = index * 4;
    output[rgbaIndex] = blendChannel(output[rgbaIndex], clampByte(output[rgbaIndex] + lumaOffset), toneAlpha);
    output[rgbaIndex + 1] = blendChannel(output[rgbaIndex + 1], clampByte(output[rgbaIndex + 1] + lumaOffset), toneAlpha);
    output[rgbaIndex + 2] = blendChannel(output[rgbaIndex + 2], clampByte(output[rgbaIndex + 2] + lumaOffset), toneAlpha);

    const originalLuma = luma(originalPixels[rgbaIndex], originalPixels[rgbaIndex + 1], originalPixels[rgbaIndex + 2]);
    const restoredLuma = luma(output[rgbaIndex], output[rgbaIndex + 1], output[rgbaIndex + 2]);
    const detailDelta = clampNumber((originalLuma - restoredLuma) * 0.08, -3.5, 3.5);
    output[rgbaIndex] = clampByte(output[rgbaIndex] + detailDelta);
    output[rgbaIndex + 1] = clampByte(output[rgbaIndex + 1] + detailDelta);
    output[rgbaIndex + 2] = clampByte(output[rgbaIndex + 2] + detailDelta);
    output[rgbaIndex + 3] = 255;

    processed += 1;
    if (processed % yieldEvery === 0) {
      onProgress?.(90, '正在微调透明水印区域色调，避免留下发白块…');
      await yieldToBrowser();
    }
  }

  return output;
}

async function suppressTransparentOverlayResidual({
  pixels,
  originalPixels,
  selectionMask,
  lowFrequencyBackgroundPixels,
  width,
  height,
  quality,
  onProgress,
}: {
  pixels: Uint8ClampedArray;
  originalPixels: Uint8ClampedArray;
  selectionMask: Uint8Array;
  lowFrequencyBackgroundPixels: Uint8ClampedArray;
  width: number;
  height: number;
  quality: ImageRepairQuality;
  onProgress?: (progress: number, message: string) => void;
}) {
  const output = new Uint8ClampedArray(pixels);
  const distanceInside = computeDistanceMap(selectionMask, width, height, 0);
  const selectedCount = countMasked(selectionMask);
  const yieldEvery = resolveLoopYieldEvery(selectedCount, quality);
  const passes = 1;
  let processed = 0;

  for (let pass = 0; pass < passes; pass += 1) {
    for (let index = 0; index < selectionMask.length; index += 1) {
      if (selectionMask[index] === 0) {
        continue;
      }

      const x = index % width;
      const y = Math.floor(index / width);
      const rgbaIndex = index * 4;
      const background = {
        r: lowFrequencyBackgroundPixels[rgbaIndex],
        g: lowFrequencyBackgroundPixels[rgbaIndex + 1],
        b: lowFrequencyBackgroundPixels[rgbaIndex + 2],
      };
      const original = {
        r: originalPixels[rgbaIndex],
        g: originalPixels[rgbaIndex + 1],
        b: originalPixels[rgbaIndex + 2],
      };
      const current = {
        r: output[rgbaIndex],
        g: output[rgbaIndex + 1],
        b: output[rgbaIndex + 2],
      };
      const originalSmooth = bilateralNeighborColor(originalPixels, width, height, x, y, quality === 'hq' ? 2 : 1, 72);
      const detailScale = quality === 'hq' ? 1.08 : 1.02;
      const target = originalSmooth
        ? {
            r: clampByte(background.r + (original.r - originalSmooth.r) * detailScale),
            g: clampByte(background.g + (original.g - originalSmooth.g) * detailScale),
            b: clampByte(background.b + (original.b - originalSmooth.b) * detailScale),
          }
        : background;

      const currentLuma = luma(current.r, current.g, current.b);
      const targetLuma = luma(target.r, target.g, target.b);
      const backgroundLuma = luma(background.r, background.g, background.b);
      const originalLuma = luma(original.r, original.g, original.b);
      const edgeBoost = 1 - smoothstep(0.7, quality === 'hq' ? 5.2 : 3.8, distanceInside[index]);
      const originalOverlayLift = Math.max(0, originalLuma - backgroundLuma);
      const removedLuma = Math.max(0, originalLuma - currentLuma);
      const removalDeficit = Math.max(0, originalOverlayLift - removedLuma);
      const alreadyRemoved = originalOverlayLift > 6 && removedLuma >= originalOverlayLift * (quality === 'hq' ? 0.86 : 0.8);
      if (alreadyRemoved) {
        continue;
      }

      const lumaDelta = Math.max(currentLuma - targetLuma, removalDeficit * 0.72);
      const residualThreshold = quality === 'hq'
        ? (1.8 - edgeBoost * 0.8)
        : (2.6 - edgeBoost * 0.9);
      if (lumaDelta <= residualThreshold) {
        continue;
      }

      const residualNeed = smoothstep(
        residualThreshold,
        quality === 'hq' ? 18 : 24,
        lumaDelta,
      );
      const extraAlpha = clampNumber(
        (lumaDelta / Math.max(22, 252 - targetLuma)) * (quality === 'hq' ? 1.28 : 1.12),
        0.015,
        quality === 'hq' ? 0.34 : 0.26,
      );
      const inverse = 1 / Math.max(0.28, 1 - extraAlpha);
      const dewhitened = {
        r: clampByte((current.r - 252 * extraAlpha) * inverse),
        g: clampByte((current.g - 252 * extraAlpha) * inverse),
        b: clampByte((current.b - 252 * extraAlpha) * inverse),
      };
      const correctionTarget = mixRgbColors(dewhitened, target, 0.12 + edgeBoost * 0.1);
      const alpha = clampNumber(
        residualNeed * 0.58 * (0.88 + edgeBoost * 0.1),
        0,
        quality === 'hq' ? 0.68 : 0.54,
      );

      if (alpha <= 0.025) {
        continue;
      }

      output[rgbaIndex] = blendChannel(current.r, correctionTarget.r, alpha);
      output[rgbaIndex + 1] = blendChannel(current.g, correctionTarget.g, alpha);
      output[rgbaIndex + 2] = blendChannel(current.b, correctionTarget.b, alpha);
      output[rgbaIndex + 3] = 255;

      processed += 1;
      if (processed % yieldEvery === 0) {
        onProgress?.(92, '正在二次压制透明水印残留轮廓，恢复底图纹理…');
        await yieldToBrowser();
      }
    }
  }

  return output;
}

async function suppressTransparentOverlayBoundaryHalo({
  pixels,
  originalPixels,
  selectionMask,
  width,
  height,
  quality,
  onProgress,
}: {
  pixels: Uint8ClampedArray;
  originalPixels: Uint8ClampedArray;
  selectionMask: Uint8Array;
  width: number;
  height: number;
  quality: ImageRepairQuality;
  onProgress?: (progress: number, message: string) => void;
}) {
  const innerWidth = quality === 'hq' ? 8 : 5;
  const outerWidth = quality === 'hq' ? 9 : 6;
  const innerEdge = subtractMasks(selectionMask, erodeMask(selectionMask, width, height, innerWidth));
  const outerEdge = subtractMasks(dilateMask(selectionMask, width, height, outerWidth), selectionMask);
  const haloMask = new Uint8Array(selectionMask.length);

  for (let index = 0; index < haloMask.length; index += 1) {
    haloMask[index] = innerEdge[index] !== 0 || outerEdge[index] !== 0 ? 1 : 0;
  }

  const haloCount = countMasked(haloMask);
  if (haloCount < 24) {
    return pixels;
  }

  const output = new Uint8ClampedArray(pixels);
  const distanceInside = computeDistanceMap(selectionMask, width, height, 0);
  const distanceToSelection = computeDistanceMap(selectionMask, width, height, 1);
  const searchRadius = quality === 'hq'
    ? Math.min(Math.max(width, height), 120)
    : Math.min(Math.max(width, height), 80);
  const yieldEvery = resolveLoopYieldEvery(haloCount, quality);
  let processed = 0;

  for (let index = 0; index < haloMask.length; index += 1) {
    if (haloMask[index] === 0) {
      continue;
    }

    const x = index % width;
    const y = Math.floor(index / width);
    const rgbaIndex = index * 4;
    const current = {
      r: output[rgbaIndex],
      g: output[rgbaIndex + 1],
      b: output[rgbaIndex + 2],
    };
    const interpolated = interpolatedBoundaryColor(output, haloMask, width, height, x, y, searchRadius)
      ?? directionalKnownColor(output, haloMask, width, height, x, y, searchRadius)
      ?? bilateralNeighborColor(output, width, height, x, y, quality === 'hq' ? 2 : 1, 48)
      ?? medianNeighborColor(output, width, height, x, y, 1);

    if (!interpolated) {
      continue;
    }

    const original = {
      r: originalPixels[rgbaIndex],
      g: originalPixels[rgbaIndex + 1],
      b: originalPixels[rgbaIndex + 2],
    };
    const originalSmooth = bilateralNeighborColor(originalPixels, width, height, x, y, quality === 'hq' ? 2 : 1, 64);
    const originalMismatch = colorDistance(original, interpolated);
    const detailScale =
      (quality === 'hq' ? 0.07 : 0.05) *
      (1 - smoothstep(quality === 'hq' ? 3 : 4, quality === 'hq' ? 16 : 22, originalMismatch));
    const detailMatched = originalSmooth
      ? {
          r: clampByte(interpolated.r + clampNumber(originalPixels[rgbaIndex] - originalSmooth.r, -4, 4) * detailScale),
          g: clampByte(interpolated.g + clampNumber(originalPixels[rgbaIndex + 1] - originalSmooth.g, -4, 4) * detailScale),
          b: clampByte(interpolated.b + clampNumber(originalPixels[rgbaIndex + 2] - originalSmooth.b, -4, 4) * detailScale),
        }
      : interpolated;

    const currentLuma = luma(current.r, current.g, current.b);
    const targetLuma = luma(interpolated.r, interpolated.g, interpolated.b);
    const lumaDelta = Math.abs(currentLuma - targetLuma);
    const chromaDelta = colorDistance(current, interpolated);
    const edgeDistance = selectionMask[index] !== 0 ? distanceInside[index] : distanceToSelection[index];
    const originalLuma = luma(original.r, original.g, original.b);
    const contourWeight = 1 - smoothstep(0.3, quality === 'hq' ? 8.2 : 5.4, edgeDistance);
    const residualNeed = smoothstep(
      quality === 'hq' ? 0.8 : 1.2,
      quality === 'hq' ? 11 : 15,
      lumaDelta + chromaDelta * 0.08,
    );
    const originalResidualNeed = smoothstep(
      quality === 'hq' ? 1.2 : 1.8,
      quality === 'hq' ? 16 : 22,
      Math.abs(originalLuma - targetLuma) + originalMismatch * 0.08,
    );
    const baseFeather = quality === 'hq' ? 0.22 : 0.15;
    const alpha = clampNumber(
      contourWeight * (baseFeather + residualNeed * (quality === 'hq' ? 0.62 : 0.46) + originalResidualNeed * 0.24),
      0,
      quality === 'hq' ? 0.86 : 0.66,
    );

    if (alpha <= 0.01) {
      continue;
    }

    const target = mixRgbColors(current, detailMatched, 0.95);
    output[rgbaIndex] = blendChannel(current.r, target.r, alpha);
    output[rgbaIndex + 1] = blendChannel(current.g, target.g, alpha);
    output[rgbaIndex + 2] = blendChannel(current.b, target.b, alpha);
    output[rgbaIndex + 3] = 255;

    processed += 1;
    if (processed % yieldEvery === 0) {
      onProgress?.(93, '正在软化透明水印边界轮廓，消除残留描边…');
      await yieldToBrowser();
    }
  }

  return output;
}

async function repairOutlierArtifacts(
  pixels: Uint8ClampedArray,
  fillMask: Uint8Array,
  width: number,
  height: number,
  quality: ImageRepairQuality,
  onProgress?: (progress: number, message: string) => void,
) {
  const output = new Uint8ClampedArray(pixels);
  const threshold = quality === 'hq' ? 76 : 92;
  const alpha = quality === 'hq' ? 0.34 : 0.26;
  const maskedPixelCount = countMasked(fillMask);
  const yieldEvery = resolveLoopYieldEvery(maskedPixelCount, quality);
  let processed = 0;

  for (let index = 0; index < fillMask.length; index += 1) {
    if (fillMask[index] === 0) {
      continue;
    }

    const x = index % width;
    const y = Math.floor(index / width);
    const median = medianNeighborColor(pixels, width, height, x, y, 1);
    if (!median) {
      continue;
    }

    const variance = localLumaVariance(pixels, width, height, x, y, 1);
    const rgbaIndex = index * 4;
    const delta =
      Math.abs(pixels[rgbaIndex] - median.r) +
      Math.abs(pixels[rgbaIndex + 1] - median.g) +
      Math.abs(pixels[rgbaIndex + 2] - median.b);

    if (delta > threshold && (variance < 1700 || delta > threshold * 1.7)) {
      output[rgbaIndex] = blendChannel(pixels[rgbaIndex], median.r, alpha);
      output[rgbaIndex + 1] = blendChannel(pixels[rgbaIndex + 1], median.g, alpha);
      output[rgbaIndex + 2] = blendChannel(pixels[rgbaIndex + 2], median.b, alpha);
    }

    processed += 1;
    if (processed % yieldEvery === 0) {
      onProgress?.(85, '正在清理零星伪影，保持页面可响应…');
      await yieldToBrowser();
    }
  }

  return output;
}

async function blendWithOriginal({
  originalPixels,
  rebuiltPixels,
  baseMask,
  fillMask,
  protectedMask,
  repairConfidence,
  distanceToKnown,
  distanceInsideSelection,
  distanceToSelection,
  maskedPixelCount,
  width,
  height,
  quality,
  onProgress,
}: {
  originalPixels: Uint8ClampedArray;
  rebuiltPixels: Uint8ClampedArray;
  baseMask: Uint8Array;
  fillMask: Uint8Array;
  protectedMask: Uint8Array;
  repairConfidence: Float32Array;
  distanceToKnown: Float32Array;
  distanceInsideSelection: Float32Array;
  distanceToSelection: Float32Array;
  maskedPixelCount: number;
  width: number;
  height: number;
  quality: ImageRepairQuality;
  onProgress?: (progress: number, message: string) => void;
}) {
  const output = new Uint8ClampedArray(rebuiltPixels);
  const baseBounds = getBinaryMaskBounds(baseMask, width, height);
  const baseRectArea = baseBounds
    ? Math.max(1, (baseBounds.right - baseBounds.left + 1) * (baseBounds.bottom - baseBounds.top + 1))
    : 1;
  const baseFillRatio = countMasked(baseMask) / baseRectArea;
  const sparseForegroundMask = baseFillRatio < 0.58;
  const edgeOverlayMask = subtractMasks(baseMask, erodeMask(baseMask, width, height, quality === 'hq' ? 3 : 2));
  const edgeOutsideMask = subtractMasks(dilateMask(baseMask, width, height, quality === 'hq' ? 3 : 2), baseMask);
  const edgeStats = computeStats(originalPixels, edgeOverlayMask);
  const edgeOutsideStats = computeStats(originalPixels, edgeOutsideMask);
  const edgeLuma = luma(edgeStats.mean.r, edgeStats.mean.g, edgeStats.mean.b);
  const edgeOutsideLuma = luma(edgeOutsideStats.mean.r, edgeOutsideStats.mean.g, edgeOutsideStats.mean.b);
  const edgeStd = (edgeStats.std.r + edgeStats.std.g + edgeStats.std.b) / 3;
  const edgeLooksLikeOverlay =
    !sparseForegroundMask &&
    edgeStats.count > 12 &&
    edgeOutsideStats.count > 12 &&
    colorDistance(edgeStats.mean, edgeOutsideStats.mean) > (quality === 'hq' ? 32 : 40) &&
    edgeLuma - edgeOutsideLuma > 24 &&
    edgeStd < 52 &&
    Math.min(edgeStats.mean.r, edgeStats.mean.g, edgeStats.mean.b) > 135;
  const innerFeatherDepth = sparseForegroundMask
    ? (quality === 'hq' ? 5.5 : 4)
    : edgeLooksLikeOverlay
      ? (quality === 'hq' ? 6.5 : 4.5)
      : (quality === 'hq' ? 17 : 11);
  const outerFeatherDepth = sparseForegroundMask
    ? (quality === 'hq' ? 8 : 5.5)
    : edgeLooksLikeOverlay
      ? (quality === 'hq' ? 9 : 6)
      : (quality === 'hq' ? 18 : 12);
  const yieldEvery = resolveLoopYieldEvery(maskedPixelCount, quality);
  let processed = 0;

  for (let index = 0; index < fillMask.length; index += 1) {
    if (fillMask[index] === 0) {
      continue;
    }

    const rgbaIndex = index * 4;
    let alpha: number;
    if (baseMask[index] !== 0) {
      const confidence = repairConfidence[index];
      const x = index % width;
      const y = Math.floor(index / width);
      const edgeJitter = (hashUnit(x, y) - 0.5) * (
        sparseForegroundMask ? (quality === 'hq' ? 1.2 : 0.8) : (quality === 'hq' ? 5.2 : 3.4)
      );
      const innerEdge = smoothstep(0.9, innerFeatherDepth, distanceInsideSelection[index] + edgeJitter);
      const redDelta = originalPixels[rgbaIndex] - rebuiltPixels[rgbaIndex];
      const greenDelta = originalPixels[rgbaIndex + 1] - rebuiltPixels[rgbaIndex + 1];
      const blueDelta = originalPixels[rgbaIndex + 2] - rebuiltPixels[rgbaIndex + 2];
      const removalNeed = smoothstep(
        quality === 'hq' ? 18 : 22,
        quality === 'hq' ? 58 : 68,
        Math.sqrt(redDelta * redDelta + greenDelta * greenDelta + blueDelta * blueDelta),
      );
      const effectiveRemovalNeed = sparseForegroundMask || edgeLooksLikeOverlay
        ? removalNeed
        : removalNeed * smoothstep(2.4, innerFeatherDepth, distanceInsideSelection[index]);
      const protectedBoost = protectedMask[index] !== 0
        ? smoothstep(innerFeatherDepth * 0.45, innerFeatherDepth + 2.5, distanceInsideSelection[index]) * ((sparseForegroundMask || edgeLooksLikeOverlay) ? 0.06 : 0.14)
        : 0;
      const edgeFloor = sparseForegroundMask || edgeLooksLikeOverlay
        ? (quality === 'hq' ? 0.96 : 0.94)
        : (quality === 'hq' ? 0.2 : 0.24);
      const softEdgeAlpha = edgeFloor + (1 - edgeFloor) * Math.max(innerEdge, effectiveRemovalNeed);
      alpha = confidence * clampNumber(
        softEdgeAlpha + protectedBoost,
        edgeFloor,
        (sparseForegroundMask || edgeLooksLikeOverlay) ? 0.999 : 0.995,
      );
    } else {
      const x = index % width;
      const y = Math.floor(index / width);
      const edgeJitter = (hashUnit(x + 17, y - 11) - 0.5) * (
        sparseForegroundMask ? (quality === 'hq' ? 1.6 : 1) : (quality === 'hq' ? 5.8 : 3.6)
      );
      const outerBlend = 1 - smoothstep(0.6, outerFeatherDepth, distanceToSelection[index] + edgeJitter);
      const knownBlend = 1 - smoothstep(0.3, outerFeatherDepth + 1.5, distanceToKnown[index]);
      const outerMaxAlpha = sparseForegroundMask
        ? (quality === 'hq' ? 0.48 : 0.38)
        : edgeLooksLikeOverlay
          ? (quality === 'hq' ? 0.52 : 0.42)
        : (quality === 'hq' ? 0.26 : 0.2);
      alpha = clampNumber(
        0.006 + outerBlend * outerMaxAlpha * (0.64 + knownBlend * 0.2),
        0,
        outerMaxAlpha,
      );
    }

    output[rgbaIndex] = blendChannel(originalPixels[rgbaIndex], output[rgbaIndex], alpha);
    output[rgbaIndex + 1] = blendChannel(originalPixels[rgbaIndex + 1], output[rgbaIndex + 1], alpha);
    output[rgbaIndex + 2] = blendChannel(originalPixels[rgbaIndex + 2], output[rgbaIndex + 2], alpha);
    output[rgbaIndex + 3] = 255;

    processed += 1;
    if (processed % yieldEvery === 0) {
      onProgress?.(97, '正在做软边融合，避免边界发硬…');
      await yieldToBrowser();
    }
  }

  return softenBlendSeam({
    pixels: output,
    fillMask,
    baseMask,
    distanceInsideSelection,
    distanceToSelection,
    maskedPixelCount,
    width,
    height,
    quality,
    onProgress,
  });
}

async function softenBlendSeam({
  pixels,
  fillMask,
  baseMask,
  distanceInsideSelection,
  distanceToSelection,
  maskedPixelCount,
  width,
  height,
  quality,
  onProgress,
}: {
  pixels: Uint8ClampedArray;
  fillMask: Uint8Array;
  baseMask: Uint8Array;
  distanceInsideSelection: Float32Array;
  distanceToSelection: Float32Array;
  maskedPixelCount: number;
  width: number;
  height: number;
  quality: ImageRepairQuality;
  onProgress?: (progress: number, message: string) => void;
}) {
  const output = new Uint8ClampedArray(pixels);
  const bandDepth = quality === 'hq' ? 7.5 : 5;
  const outerBandDepth = quality === 'hq' ? 3.2 : 2.2;
  const distanceToFill = computeDistanceMap(fillMask, width, height, 1);
  const yieldEvery = resolveLoopYieldEvery(maskedPixelCount, quality);
  let processed = 0;

  for (let index = 0; index < fillMask.length; index += 1) {
    const insideFill = fillMask[index] !== 0;
    if (!insideFill && distanceToFill[index] > outerBandDepth) {
      continue;
    }

    const distanceToHardEdge = insideFill
      ? (baseMask[index] !== 0 ? distanceInsideSelection[index] : distanceToSelection[index])
      : distanceToFill[index];
    if (distanceToHardEdge > bandDepth) {
      continue;
    }

    const x = index % width;
    const y = Math.floor(index / width);
    const smoothed = bilateralNeighborColor(pixels, width, height, x, y, quality === 'hq' ? 2 : 1, quality === 'hq' ? 56 : 46);
    if (!smoothed) {
      continue;
    }

    const edgeStrength = 1 - smoothstep(0.7, bandDepth, distanceToHardEdge + (hashUnit(x - 5, y + 9) - 0.5) * 0.8);
    const maxAlpha = insideFill
      ? (quality === 'hq' ? 0.3 : 0.22)
      : (quality === 'hq' ? 0.12 : 0.08);
    const alpha = clampNumber(edgeStrength * maxAlpha, insideFill ? 0.03 : 0.01, maxAlpha);
    const rgbaIndex = index * 4;
    output[rgbaIndex] = blendChannel(output[rgbaIndex], smoothed.r, alpha);
    output[rgbaIndex + 1] = blendChannel(output[rgbaIndex + 1], smoothed.g, alpha);
    output[rgbaIndex + 2] = blendChannel(output[rgbaIndex + 2], smoothed.b, alpha);
    output[rgbaIndex + 3] = 255;

    processed += 1;
    if (processed % yieldEvery === 0) {
      onProgress?.(98, '正在软化矩形边界，消除直线和硬角…');
      await yieldToBrowser();
    }
  }

  return output;
}

function frontierPriority(mask: Uint8Array, width: number, height: number, index: number) {
  return countKnownNeighbors(mask, width, height, index) * 4 + countPatchKnownSamples(mask, width, height, index, 2);
}

function countKnownNeighbors(mask: Uint8Array, width: number, height: number, index: number) {
  const x = index % width;
  const y = Math.floor(index / width);
  let total = 0;

  for (let offsetY = -1; offsetY <= 1; offsetY += 1) {
    const sampleY = y + offsetY;
    if (sampleY < 0 || sampleY >= height) {
      continue;
    }

    for (let offsetX = -1; offsetX <= 1; offsetX += 1) {
      const sampleX = x + offsetX;
      if (sampleX < 0 || sampleX >= width || (offsetX === 0 && offsetY === 0)) {
        continue;
      }

      if (mask[sampleY * width + sampleX] === 0) {
        total += 1;
      }
    }
  }

  return total;
}

function countPatchKnownSamples(mask: Uint8Array, width: number, height: number, index: number, patchRadius: number) {
  const x = index % width;
  const y = Math.floor(index / width);
  let total = 0;

  for (let offsetY = -patchRadius; offsetY <= patchRadius; offsetY += 1) {
    const sampleY = y + offsetY;
    if (sampleY < 0 || sampleY >= height) {
      continue;
    }

    for (let offsetX = -patchRadius; offsetX <= patchRadius; offsetX += 1) {
      const sampleX = x + offsetX;
      if (sampleX < 0 || sampleX >= width) {
        continue;
      }

      if (mask[sampleY * width + sampleX] === 0) {
        total += 1;
      }
    }
  }

  return total;
}

function findBestDonorPatch({
  pixels,
  donorPixels,
  mask,
  width,
  height,
  targetX,
  targetY,
  searchRadius,
  patchRadius,
  candidateStep,
  minKnownSamples,
}: {
  pixels: Uint8ClampedArray;
  donorPixels: Uint8ClampedArray;
  mask: Uint8Array;
  width: number;
  height: number;
  targetX: number;
  targetY: number;
  searchRadius: number;
  patchRadius: number;
  candidateStep: number;
  minKnownSamples: number;
}) {
  const scanWindow = (radius: number, step: number) => {
    let best: PatchMatch | null = null;

    for (let sampleY = Math.max(patchRadius, targetY - radius); sampleY <= Math.min(height - patchRadius - 1, targetY + radius); sampleY += step) {
      for (let sampleX = Math.max(patchRadius, targetX - radius); sampleX <= Math.min(width - patchRadius - 1, targetX + radius); sampleX += step) {
        if (!isPatchKnown(mask, width, height, sampleX, sampleY, patchRadius)) {
          continue;
        }

        const score = scorePatchCandidate({
          pixels,
          donorPixels,
          mask,
          width,
          height,
          targetX,
          targetY,
          sampleX,
          sampleY,
          patchRadius,
          minKnownSamples,
        });

        if (!score || (best && score.score >= best.score)) {
          continue;
        }

        best = { x: sampleX, y: sampleY, score: score.score, adjustment: score.adjustment };
      }
    }

    return best;
  };

  return scanWindow(searchRadius, candidateStep)
    ?? scanWindow(Math.min(Math.max(width, height), Math.round(searchRadius * 1.8)), candidateStep + 1);
}

function scorePatchCandidate({
  pixels,
  donorPixels,
  mask,
  width,
  height,
  targetX,
  targetY,
  sampleX,
  sampleY,
  patchRadius,
  minKnownSamples,
}: {
  pixels: Uint8ClampedArray;
  donorPixels: Uint8ClampedArray;
  mask: Uint8Array;
  width: number;
  height: number;
  targetX: number;
  targetY: number;
  sampleX: number;
  sampleY: number;
  patchRadius: number;
  minKnownSamples: number;
}) {
  let knownSamples = 0;
  let totalWeight = 0;
  let diffR = 0;
  let diffG = 0;
  let diffB = 0;
  let targetMeanLuma = 0;
  let donorMeanLuma = 0;

  for (let offsetY = -patchRadius; offsetY <= patchRadius; offsetY += 1) {
    const targetSampleY = targetY + offsetY;
    const donorSampleY = sampleY + offsetY;
    if (targetSampleY < 0 || targetSampleY >= height || donorSampleY < 0 || donorSampleY >= height) {
      continue;
    }

    for (let offsetX = -patchRadius; offsetX <= patchRadius; offsetX += 1) {
      const targetSampleX = targetX + offsetX;
      const donorSampleX = sampleX + offsetX;
      if (targetSampleX < 0 || targetSampleX >= width || donorSampleX < 0 || donorSampleX >= width) {
        continue;
      }

      const targetIndex = targetSampleY * width + targetSampleX;
      const donorIndex = donorSampleY * width + donorSampleX;
      if (mask[targetIndex] !== 0 || mask[donorIndex] !== 0) {
        continue;
      }

      const weight = 1 / (Math.abs(offsetX) + Math.abs(offsetY) + 1);
      const targetRgbaIndex = targetIndex * 4;
      const donorRgbaIndex = donorIndex * 4;
      diffR += (pixels[targetRgbaIndex] - donorPixels[donorRgbaIndex]) * weight;
      diffG += (pixels[targetRgbaIndex + 1] - donorPixels[donorRgbaIndex + 1]) * weight;
      diffB += (pixels[targetRgbaIndex + 2] - donorPixels[donorRgbaIndex + 2]) * weight;
      targetMeanLuma += luma(pixels[targetRgbaIndex], pixels[targetRgbaIndex + 1], pixels[targetRgbaIndex + 2]) * weight;
      donorMeanLuma += luma(donorPixels[donorRgbaIndex], donorPixels[donorRgbaIndex + 1], donorPixels[donorRgbaIndex + 2]) * weight;
      totalWeight += weight;
      knownSamples += 1;
    }
  }

  if (knownSamples < minKnownSamples || totalWeight === 0) {
    return null;
  }

  const adjustment = {
    r: clampNumber(diffR / totalWeight, -40, 40),
    g: clampNumber(diffG / totalWeight, -40, 40),
    b: clampNumber(diffB / totalWeight, -40, 40),
  };
  const targetLumaAverage = targetMeanLuma / totalWeight;
  const donorLumaAverage = donorMeanLuma / totalWeight;
  let totalError = 0;
  let targetVariance = 0;
  let donorVariance = 0;

  for (let offsetY = -patchRadius; offsetY <= patchRadius; offsetY += 1) {
    const targetSampleY = targetY + offsetY;
    const donorSampleY = sampleY + offsetY;
    if (targetSampleY < 0 || targetSampleY >= height || donorSampleY < 0 || donorSampleY >= height) {
      continue;
    }

    for (let offsetX = -patchRadius; offsetX <= patchRadius; offsetX += 1) {
      const targetSampleX = targetX + offsetX;
      const donorSampleX = sampleX + offsetX;
      if (targetSampleX < 0 || targetSampleX >= width || donorSampleX < 0 || donorSampleX >= width) {
        continue;
      }

      const targetIndex = targetSampleY * width + targetSampleX;
      const donorIndex = donorSampleY * width + donorSampleX;
      if (mask[targetIndex] !== 0 || mask[donorIndex] !== 0) {
        continue;
      }

      const weight = 1 / (Math.abs(offsetX) + Math.abs(offsetY) + 1);
      const targetRgbaIndex = targetIndex * 4;
      const donorRgbaIndex = donorIndex * 4;
      const donorR = clampByte(donorPixels[donorRgbaIndex] + adjustment.r);
      const donorG = clampByte(donorPixels[donorRgbaIndex + 1] + adjustment.g);
      const donorB = clampByte(donorPixels[donorRgbaIndex + 2] + adjustment.b);
      const deltaR = pixels[targetRgbaIndex] - donorR;
      const deltaG = pixels[targetRgbaIndex + 1] - donorG;
      const deltaB = pixels[targetRgbaIndex + 2] - donorB;
      totalError += (deltaR * deltaR + deltaG * deltaG + deltaB * deltaB) * weight;

      const targetL = luma(pixels[targetRgbaIndex], pixels[targetRgbaIndex + 1], pixels[targetRgbaIndex + 2]) - targetLumaAverage;
      const donorL = luma(donorR, donorG, donorB) - donorLumaAverage;
      targetVariance += targetL * targetL * weight;
      donorVariance += donorL * donorL * weight;
    }
  }

  const distancePenalty = ((targetX - sampleX) ** 2 + (targetY - sampleY) ** 2) * 0.015;
  const variancePenalty = Math.abs(targetVariance - donorVariance) / totalWeight;

  return {
    score: totalError / totalWeight + variancePenalty * 0.35 + distancePenalty,
    adjustment,
  };
}

function copyDonorPatchIntoMask({
  pixels,
  mask,
  donorPixels,
  width,
  height,
  targetX,
  targetY,
  donorX,
  donorY,
  patchRadius,
  adjustment,
}: {
  pixels: Uint8ClampedArray;
  mask: Uint8Array;
  donorPixels: Uint8ClampedArray;
  width: number;
  height: number;
  targetX: number;
  targetY: number;
  donorX: number;
  donorY: number;
  patchRadius: number;
  adjustment: Rgb;
}) {
  let writes = 0;

  for (let offsetY = -patchRadius; offsetY <= patchRadius; offsetY += 1) {
    const targetSampleY = targetY + offsetY;
    const donorSampleY = donorY + offsetY;
    if (targetSampleY < 0 || targetSampleY >= height || donorSampleY < 0 || donorSampleY >= height) {
      continue;
    }

    for (let offsetX = -patchRadius; offsetX <= patchRadius; offsetX += 1) {
      const targetSampleX = targetX + offsetX;
      const donorSampleX = donorX + offsetX;
      if (targetSampleX < 0 || targetSampleX >= width || donorSampleX < 0 || donorSampleX >= width) {
        continue;
      }

      const targetIndex = targetSampleY * width + targetSampleX;
      if (mask[targetIndex] === 0) {
        continue;
      }

      const donorIndex = donorSampleY * width + donorSampleX;
      if (mask[donorIndex] !== 0) {
        continue;
      }

      const targetRgbaIndex = targetIndex * 4;
      const donorRgbaIndex = donorIndex * 4;
      pixels[targetRgbaIndex] = clampByte(donorPixels[donorRgbaIndex] + adjustment.r);
      pixels[targetRgbaIndex + 1] = clampByte(donorPixels[donorRgbaIndex + 1] + adjustment.g);
      pixels[targetRgbaIndex + 2] = clampByte(donorPixels[donorRgbaIndex + 2] + adjustment.b);
      pixels[targetRgbaIndex + 3] = 255;
      mask[targetIndex] = 0;
      writes += 1;
    }
  }

  return writes;
}

function sampleAdjustedDonorPixel(donorPixels: Uint8ClampedArray, rgbaIndex: number, adjustment: Rgb) {
  return {
    r: clampByte(donorPixels[rgbaIndex] + adjustment.r),
    g: clampByte(donorPixels[rgbaIndex + 1] + adjustment.g),
    b: clampByte(donorPixels[rgbaIndex + 2] + adjustment.b),
  };
}

function fillRemainingPixels(
  pixels: Uint8ClampedArray,
  mask: Uint8Array,
  width: number,
  height: number,
  searchRadius: number,
) {
  let remaining = countMasked(mask);

  while (remaining > 0) {
    let writes = 0;

    for (let index = 0; index < mask.length; index += 1) {
      if (mask[index] === 0) {
        continue;
      }

      const x = index % width;
      const y = Math.floor(index / width);
      const average = averageKnownNeighbors(pixels, mask, width, height, x, y, Math.min(4, Math.max(2, Math.round(searchRadius / 6))))
        ?? directionalKnownColor(pixels, mask, width, height, x, y, Math.max(24, searchRadius));
      if (!average) {
        continue;
      }

      const rgbaIndex = index * 4;
      pixels[rgbaIndex] = clampByte(average.r);
      pixels[rgbaIndex + 1] = clampByte(average.g);
      pixels[rgbaIndex + 2] = clampByte(average.b);
      pixels[rgbaIndex + 3] = 255;
      mask[index] = 0;
      writes += 1;
    }

    if (writes === 0) {
      const fallbackColor = averageBoundaryColor(pixels, mask, width, height);
      for (let index = 0; index < mask.length; index += 1) {
        if (mask[index] === 0) {
          continue;
        }

        const rgbaIndex = index * 4;
        pixels[rgbaIndex] = clampByte(fallbackColor.r);
        pixels[rgbaIndex + 1] = clampByte(fallbackColor.g);
        pixels[rgbaIndex + 2] = clampByte(fallbackColor.b);
        pixels[rgbaIndex + 3] = 255;
        mask[index] = 0;
      }
      return;
    }

    remaining = countMasked(mask);
  }
}

function averageBoundaryColor(pixels: Uint8ClampedArray, mask: Uint8Array, width: number, height: number) {
  let totalWeight = 0;
  let red = 0;
  let green = 0;
  let blue = 0;

  for (let index = 0; index < mask.length; index += 1) {
    if (mask[index] === 0 || !hasKnownNeighbor(mask, width, height, index)) {
      continue;
    }

    const x = index % width;
    const y = Math.floor(index / width);
    const average = averageKnownNeighbors(pixels, mask, width, height, x, y, 1);
    if (!average) {
      continue;
    }

    totalWeight += 1;
    red += average.r;
    green += average.g;
    blue += average.b;
  }

  if (totalWeight === 0) {
    return { r: 127, g: 127, b: 127 };
  }

  return { r: red / totalWeight, g: green / totalWeight, b: blue / totalWeight };
}

function directionalKnownColor(
  pixels: Uint8ClampedArray,
  mask: Uint8Array,
  width: number,
  height: number,
  x: number,
  y: number,
  maxDistance: number,
) {
  const interpolated = interpolatedBoundaryColor(pixels, mask, width, height, x, y, maxDistance);
  if (interpolated) {
    return interpolated;
  }

  const samples = collectDirectionalKnownSamples(pixels, mask, width, height, x, y, maxDistance);
  if (samples.length === 0) {
    return null;
  }

  return weightedDirectionalColor(samples);
}

function interpolatedBoundaryColor(
  pixels: Uint8ClampedArray,
  mask: Uint8Array,
  width: number,
  height: number,
  x: number,
  y: number,
  maxDistance: number,
) {
  const directionPairs = [
    [[-1, 0], [1, 0], 1],
    [[0, -1], [0, 1], 1],
    [[-1, -1], [1, 1], 0.76],
    [[1, -1], [-1, 1], 0.76],
  ] as const;
  let totalWeight = 0;
  let red = 0;
  let green = 0;
  let blue = 0;

  for (const [leftDirection, rightDirection, pairWeight] of directionPairs) {
    const left = traceKnownBoundarySample(
      pixels,
      mask,
      width,
      height,
      x,
      y,
      leftDirection[0],
      leftDirection[1],
      maxDistance,
    );
    const right = traceKnownBoundarySample(
      pixels,
      mask,
      width,
      height,
      x,
      y,
      rightDirection[0],
      rightDirection[1],
      maxDistance,
    );

    if (left && right) {
      const distanceSum = left.distance + right.distance;
      if (distanceSum <= 0) {
        continue;
      }

      const interpolated = mixRgbColors(left, right, left.distance / distanceSum);
      const continuity = 1 - clampNumber(colorDistance(left, right) / 190, 0, 0.58);
      const averageDistance = Math.max(1, distanceSum * 0.5);
      const weight = pairWeight * continuity / Math.sqrt(averageDistance);
      totalWeight += weight;
      red += interpolated.r * weight;
      green += interpolated.g * weight;
      blue += interpolated.b * weight;
      continue;
    }

    const single = left ?? right;
    if (!single) {
      continue;
    }

    const weight = pairWeight * 0.28 / Math.max(1, single.distance);
    totalWeight += weight;
    red += single.r * weight;
    green += single.g * weight;
    blue += single.b * weight;
  }

  if (totalWeight === 0) {
    return null;
  }

  return { r: red / totalWeight, g: green / totalWeight, b: blue / totalWeight };
}

function findLinearFeatureColor(
  currentPixels: Uint8ClampedArray,
  originalPixels: Uint8ClampedArray,
  mask: Uint8Array,
  width: number,
  height: number,
  x: number,
  y: number,
  maxDistance: number,
  quality: ImageRepairQuality,
) {
  const directionPairs = [
    [[-1, 0], [1, 0], 0.82],
    [[0, -1], [0, 1], 0.82],
    [[-1, -1], [1, 1], 0.92],
    [[1, -1], [-1, 1], 0.92],
    [[-2, 1], [2, -1], 1],
    [[2, 1], [-2, -1], 1],
    [[-3, 1], [3, -1], 0.9],
    [[3, 1], [-3, -1], 0.9],
    [[-3, 2], [3, -2], 0.86],
    [[3, 2], [-3, -2], 0.86],
  ] as const;
  const rgbaIndex = (y * width + x) * 4;
  const currentColor = {
    r: currentPixels[rgbaIndex],
    g: currentPixels[rgbaIndex + 1],
    b: currentPixels[rgbaIndex + 2],
  };
  let best: { color: Rgb; strength: number; score: number } | null = null;

  for (const [leftDirection, rightDirection, directionWeight] of directionPairs) {
    const left = traceKnownBoundarySample(
      originalPixels,
      mask,
      width,
      height,
      x,
      y,
      leftDirection[0],
      leftDirection[1],
      maxDistance,
    );
    const right = traceKnownBoundarySample(
      originalPixels,
      mask,
      width,
      height,
      x,
      y,
      rightDirection[0],
      rightDirection[1],
      maxDistance,
    );

    if (!left || !right) {
      continue;
    }

    const pairDelta = colorDistance(left, right);
    const continuityLimit = quality === 'hq' ? 78 : 66;
    if (pairDelta > continuityLimit) {
      continue;
    }

    const distanceSum = Math.max(1, left.distance + right.distance);
    const color = mixRgbColors(left, right, left.distance / distanceSum);
    const contrast = colorDistance(color, currentColor);
    const chroma = Math.max(color.r, color.g, color.b) - Math.min(color.r, color.g, color.b);
    const continuity = 1 - pairDelta / continuityLimit;
    const distancePenalty = 1 / Math.sqrt(distanceSum * 0.5);
    const score = (contrast * 0.62 + chroma * 0.38) * continuity * directionWeight * distancePenalty;
    const minimumScore = quality === 'hq' ? 7.5 : 9;
    if (score < minimumScore) {
      continue;
    }

    const strength = clampNumber((score - minimumScore) / (quality === 'hq' ? 34 : 42), 0.08, quality === 'hq' ? 0.68 : 0.52);
    if (!best || score > best.score) {
      best = { color, strength, score };
    }
  }

  return best;
}

function traceKnownBoundarySample(
  pixels: Uint8ClampedArray,
  mask: Uint8Array,
  width: number,
  height: number,
  x: number,
  y: number,
  directionX: number,
  directionY: number,
  maxDistance: number,
): BoundarySample | null {
  for (let distance = 1; distance <= maxDistance; distance += 1) {
    const sampleX = x + directionX * distance;
    const sampleY = y + directionY * distance;
    if (sampleX < 0 || sampleX >= width || sampleY < 0 || sampleY >= height) {
      break;
    }

    const sampleIndex = sampleY * width + sampleX;
    if (mask[sampleIndex] !== 0) {
      continue;
    }

    const rgbaIndex = sampleIndex * 4;
    return {
      r: pixels[rgbaIndex],
      g: pixels[rgbaIndex + 1],
      b: pixels[rgbaIndex + 2],
      distance,
      diagonal: directionX !== 0 && directionY !== 0,
    };
  }

  return null;
}

function collectDirectionalKnownSamples(
  pixels: Uint8ClampedArray,
  mask: Uint8Array,
  width: number,
  height: number,
  x: number,
  y: number,
  maxDistance: number,
) {
  const directions = [
    [1, 0],
    [-1, 0],
    [0, 1],
    [0, -1],
    [1, 1],
    [-1, 1],
    [1, -1],
    [-1, -1],
  ];
  const samples: BoundarySample[] = [];

  for (const [directionX, directionY] of directions) {
    for (let distance = 1; distance <= maxDistance; distance += 1) {
      const sampleX = x + directionX * distance;
      const sampleY = y + directionY * distance;
      if (sampleX < 0 || sampleX >= width || sampleY < 0 || sampleY >= height) {
        break;
      }

      const sampleIndex = sampleY * width + sampleX;
      if (mask[sampleIndex] !== 0) {
        continue;
      }

      const rgbaIndex = sampleIndex * 4;
      samples.push({
        r: pixels[rgbaIndex],
        g: pixels[rgbaIndex + 1],
        b: pixels[rgbaIndex + 2],
        distance,
        diagonal: directionX !== 0 && directionY !== 0,
      });
      break;
    }
  }

  return samples;
}

function weightedDirectionalColor(samples: BoundarySample[]) {
  let totalWeight = 0;
  let red = 0;
  let green = 0;
  let blue = 0;

  for (const sample of samples) {
    const diagonalWeight = sample.diagonal ? 0.72 : 1;
    const weight = diagonalWeight / Math.max(1, sample.distance);
    totalWeight += weight;
    red += sample.r * weight;
    green += sample.g * weight;
    blue += sample.b * weight;
  }

  if (totalWeight === 0) {
    return null;
  }

  return { r: red / totalWeight, g: green / totalWeight, b: blue / totalWeight };
}

function averageKnownNeighbors(
  pixels: Uint8ClampedArray,
  mask: Uint8Array,
  width: number,
  height: number,
  x: number,
  y: number,
  radius: number,
) {
  let totalWeight = 0;
  let red = 0;
  let green = 0;
  let blue = 0;

  for (let offsetY = -radius; offsetY <= radius; offsetY += 1) {
    const sampleY = y + offsetY;
    if (sampleY < 0 || sampleY >= height) {
      continue;
    }

    for (let offsetX = -radius; offsetX <= radius; offsetX += 1) {
      const sampleX = x + offsetX;
      if (sampleX < 0 || sampleX >= width || (offsetX === 0 && offsetY === 0)) {
        continue;
      }

      const sampleIndex = sampleY * width + sampleX;
      if (mask[sampleIndex] !== 0) {
        continue;
      }

      const distance = Math.hypot(offsetX, offsetY);
      const weight = 1.8 / (distance + 0.4);
      const rgbaIndex = sampleIndex * 4;
      totalWeight += weight;
      red += pixels[rgbaIndex] * weight;
      green += pixels[rgbaIndex + 1] * weight;
      blue += pixels[rgbaIndex + 2] * weight;
    }
  }

  if (totalWeight === 0) {
    return null;
  }

  return { r: red / totalWeight, g: green / totalWeight, b: blue / totalWeight };
}

function bilateralNeighborColor(
  pixels: Uint8ClampedArray,
  width: number,
  height: number,
  x: number,
  y: number,
  radius: number,
  sigmaColor: number,
) {
  const centerIndex = (y * width + x) * 4;
  const centerR = pixels[centerIndex];
  const centerG = pixels[centerIndex + 1];
  const centerB = pixels[centerIndex + 2];
  let totalWeight = 0;
  let red = 0;
  let green = 0;
  let blue = 0;

  for (let offsetY = -radius; offsetY <= radius; offsetY += 1) {
    const sampleY = y + offsetY;
    if (sampleY < 0 || sampleY >= height) {
      continue;
    }

    for (let offsetX = -radius; offsetX <= radius; offsetX += 1) {
      const sampleX = x + offsetX;
      if (sampleX < 0 || sampleX >= width || (offsetX === 0 && offsetY === 0)) {
        continue;
      }

      const sampleIndex = (sampleY * width + sampleX) * 4;
      const distance = Math.hypot(offsetX, offsetY);
      const deltaR = pixels[sampleIndex] - centerR;
      const deltaG = pixels[sampleIndex + 1] - centerG;
      const deltaB = pixels[sampleIndex + 2] - centerB;
      const spatialWeight = 1 / (distance + 0.55);
      const colorWeight = Math.exp(-(deltaR * deltaR + deltaG * deltaG + deltaB * deltaB) / (2 * sigmaColor * sigmaColor));
      const weight = spatialWeight * colorWeight;

      totalWeight += weight;
      red += pixels[sampleIndex] * weight;
      green += pixels[sampleIndex + 1] * weight;
      blue += pixels[sampleIndex + 2] * weight;
    }
  }

  if (totalWeight === 0) {
    return null;
  }

  return { r: red / totalWeight, g: green / totalWeight, b: blue / totalWeight };
}

function diffusionNeighborColor(
  pixels: Uint8ClampedArray,
  width: number,
  height: number,
  x: number,
  y: number,
  radius: number,
) {
  let totalWeight = 0;
  let red = 0;
  let green = 0;
  let blue = 0;

  for (let offsetY = -radius; offsetY <= radius; offsetY += 1) {
    const sampleY = y + offsetY;
    if (sampleY < 0 || sampleY >= height) {
      continue;
    }

    for (let offsetX = -radius; offsetX <= radius; offsetX += 1) {
      const sampleX = x + offsetX;
      if (sampleX < 0 || sampleX >= width || (offsetX === 0 && offsetY === 0)) {
        continue;
      }

      const sampleIndex = (sampleY * width + sampleX) * 4;
      const weight = (offsetX === 0 || offsetY === 0) ? 1 : 0.82;
      totalWeight += weight;
      red += pixels[sampleIndex] * weight;
      green += pixels[sampleIndex + 1] * weight;
      blue += pixels[sampleIndex + 2] * weight;
    }
  }

  if (totalWeight === 0) {
    return null;
  }

  return { r: red / totalWeight, g: green / totalWeight, b: blue / totalWeight };
}

function mixRgbColors(base: Rgb, target: Rgb, alpha: number) {
  return {
    r: base.r * (1 - alpha) + target.r * alpha,
    g: base.g * (1 - alpha) + target.g * alpha,
    b: base.b * (1 - alpha) + target.b * alpha,
  };
}

function hashUnit(x: number, y: number) {
  const value = Math.sin(x * 12.9898 + y * 78.233) * 43758.5453;
  return value - Math.floor(value);
}

function valueNoise2d(x: number, y: number, seed: number) {
  const left = Math.floor(x);
  const top = Math.floor(y);
  const tx = smoothstep(0, 1, x - left);
  const ty = smoothstep(0, 1, y - top);
  const topLeft = hashUnit(left + seed * 17, top - seed * 11);
  const topRight = hashUnit(left + 1 + seed * 17, top - seed * 11);
  const bottomLeft = hashUnit(left + seed * 17, top + 1 - seed * 11);
  const bottomRight = hashUnit(left + 1 + seed * 17, top + 1 - seed * 11);
  const topMix = topLeft * (1 - tx) + topRight * tx;
  const bottomMix = bottomLeft * (1 - tx) + bottomRight * tx;
  return (topMix * (1 - ty) + bottomMix * ty) * 2 - 1;
}

function medianNeighborColor(
  pixels: Uint8ClampedArray,
  width: number,
  height: number,
  x: number,
  y: number,
  radius: number,
) {
  const reds: number[] = [];
  const greens: number[] = [];
  const blues: number[] = [];

  for (let offsetY = -radius; offsetY <= radius; offsetY += 1) {
    const sampleY = y + offsetY;
    if (sampleY < 0 || sampleY >= height) {
      continue;
    }

    for (let offsetX = -radius; offsetX <= radius; offsetX += 1) {
      const sampleX = x + offsetX;
      if (sampleX < 0 || sampleX >= width || (offsetX === 0 && offsetY === 0)) {
        continue;
      }

      const rgbaIndex = (sampleY * width + sampleX) * 4;
      reds.push(pixels[rgbaIndex]);
      greens.push(pixels[rgbaIndex + 1]);
      blues.push(pixels[rgbaIndex + 2]);
    }
  }

  if (reds.length === 0) {
    return null;
  }

  reds.sort((left, right) => left - right);
  greens.sort((left, right) => left - right);
  blues.sort((left, right) => left - right);
  const middle = Math.floor(reds.length / 2);

  return { r: reds[middle], g: greens[middle], b: blues[middle] };
}

function localLumaVariance(
  pixels: Uint8ClampedArray,
  width: number,
  height: number,
  x: number,
  y: number,
  radius: number,
) {
  const values: number[] = [];

  for (let offsetY = -radius; offsetY <= radius; offsetY += 1) {
    const sampleY = y + offsetY;
    if (sampleY < 0 || sampleY >= height) {
      continue;
    }

    for (let offsetX = -radius; offsetX <= radius; offsetX += 1) {
      const sampleX = x + offsetX;
      if (sampleX < 0 || sampleX >= width) {
        continue;
      }

      const rgbaIndex = (sampleY * width + sampleX) * 4;
      values.push(luma(pixels[rgbaIndex], pixels[rgbaIndex + 1], pixels[rgbaIndex + 2]));
    }
  }

  if (values.length === 0) {
    return 0;
  }

  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  let variance = 0;

  for (const value of values) {
    variance += (value - mean) ** 2;
  }

  return variance / values.length;
}

function maskedLumaVariance(
  pixels: Uint8ClampedArray,
  mask: Uint8Array,
  width: number,
  height: number,
  x: number,
  y: number,
  radius: number,
) {
  const values: number[] = [];

  for (let offsetY = -radius; offsetY <= radius; offsetY += 1) {
    const sampleY = y + offsetY;
    if (sampleY < 0 || sampleY >= height) {
      continue;
    }

    for (let offsetX = -radius; offsetX <= radius; offsetX += 1) {
      const sampleX = x + offsetX;
      if (sampleX < 0 || sampleX >= width) {
        continue;
      }

      const sampleIndex = sampleY * width + sampleX;
      if (mask[sampleIndex] !== 0) {
        continue;
      }

      const rgbaIndex = sampleIndex * 4;
      values.push(luma(pixels[rgbaIndex], pixels[rgbaIndex + 1], pixels[rgbaIndex + 2]));
    }
  }

  if (values.length < 4) {
    return null;
  }

  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  let variance = 0;

  for (const value of values) {
    variance += (value - mean) ** 2;
  }

  return variance / values.length;
}

function computeRepairConfidenceMap({
  selectionMask,
  width,
  height: _height,
  quality,
}: {
  selectionMask: Uint8Array;
  width: number;
  height: number;
  quality: ImageRepairQuality;
}) {
  const confidence = new Float32Array(selectionMask.length);
  const distanceInside = computeDistanceMap(selectionMask, width, _height, 0);

  for (let index = 0; index < selectionMask.length; index += 1) {
    if (selectionMask[index] === 0) {
      continue;
    }

    confidence[index] = 0.97 + 0.03 * smoothstep(0.5, quality === 'hq' ? 3.6 : 2.8, distanceInside[index]);
  }

  return confidence;
}

function computeStats(pixels: Uint8ClampedArray, mask: Uint8Array): RegionStats {
  let count = 0;
  let red = 0;
  let green = 0;
  let blue = 0;

  for (let index = 0; index < mask.length; index += 1) {
    if (mask[index] === 0) {
      continue;
    }

    const rgbaIndex = index * 4;
    red += pixels[rgbaIndex];
    green += pixels[rgbaIndex + 1];
    blue += pixels[rgbaIndex + 2];
    count += 1;
  }

  if (count === 0) {
    return { count: 0, mean: { r: 0, g: 0, b: 0 }, std: { r: 0, g: 0, b: 0 } };
  }

  const mean = { r: red / count, g: green / count, b: blue / count };
  let redVariance = 0;
  let greenVariance = 0;
  let blueVariance = 0;

  for (let index = 0; index < mask.length; index += 1) {
    if (mask[index] === 0) {
      continue;
    }

    const rgbaIndex = index * 4;
    redVariance += (pixels[rgbaIndex] - mean.r) ** 2;
    greenVariance += (pixels[rgbaIndex + 1] - mean.g) ** 2;
    blueVariance += (pixels[rgbaIndex + 2] - mean.b) ** 2;
  }

  return {
    count,
    mean,
    std: {
      r: Math.sqrt(redVariance / count),
      g: Math.sqrt(greenVariance / count),
      b: Math.sqrt(blueVariance / count),
    },
  };
}

function computeMeanSaturation(pixels: Uint8ClampedArray, mask: Uint8Array) {
  let count = 0;
  let total = 0;

  for (let index = 0; index < mask.length; index += 1) {
    if (mask[index] === 0) {
      continue;
    }

    const rgbaIndex = index * 4;
    total += saturation(pixels[rgbaIndex], pixels[rgbaIndex + 1], pixels[rgbaIndex + 2]);
    count += 1;
  }

  return count === 0 ? 0 : total / count;
}

function buildTransparentOverlayRepairMask({
  originalPixels,
  backgroundPixels,
  selectionMask,
  width,
  height,
  quality,
}: {
  originalPixels: Uint8ClampedArray;
  backgroundPixels: Uint8ClampedArray;
  selectionMask: Uint8Array;
  width: number;
  height: number;
  quality: ImageRepairQuality;
}) {
  const bounds = getBinaryMaskBounds(selectionMask, width, height);
  if (!bounds) {
    return null;
  }

  const selectionArea = countMasked(selectionMask);
  if (selectionArea < 120) {
    return null;
  }
  const referenceRingWidth = quality === 'hq' ? 5 : 4;
  const referenceMask = subtractMasks(dilateMask(selectionMask, width, height, referenceRingWidth), selectionMask);
  for (let y = bounds.top; y <= bounds.bottom; y += 1) {
    for (let x = bounds.left; x <= bounds.right; x += 1) {
      const index = y * width + x;
      if (selectionMask[index] === 0) {
        continue;
      }

      const borderDistance = Math.min(x - bounds.left, bounds.right - x, y - bounds.top, bounds.bottom - y);
      if (borderDistance < referenceRingWidth) {
        referenceMask[index] = 1;
      }
    }
  }

  const referenceStats = computeStats(originalPixels, referenceMask);
  if (referenceStats.count < 32) {
    return null;
  }

  const referenceLuma = luma(referenceStats.mean.r, referenceStats.mean.g, referenceStats.mean.b);
  const referenceStd = (referenceStats.std.r + referenceStats.std.g + referenceStats.std.b) / 3;
  const referenceSaturation = computeMeanSaturation(originalPixels, referenceMask);
  const globalSoftLiftThreshold = Math.max(quality === 'hq' ? 22 : 27, referenceStd * 0.82);
  const globalSeedLiftThreshold = Math.max(quality === 'hq' ? 32 : 38, referenceStd * 1.16);
  const candidate = new Uint8Array(selectionMask.length);
  const seed = new Uint8Array(selectionMask.length);
  const signals = new Float32Array(selectionMask.length);
  const seedThreshold = quality === 'hq' ? 8.5 : 11;
  const strongThreshold = quality === 'hq' ? 13 : 17;
  let candidateCount = 0;
  let seedCount = 0;

  for (let y = bounds.top; y <= bounds.bottom; y += 1) {
    for (let x = bounds.left; x <= bounds.right; x += 1) {
      const index = y * width + x;
      if (selectionMask[index] === 0) {
        continue;
      }

      const rgbaIndex = index * 4;
      const observed = {
        r: originalPixels[rgbaIndex],
        g: originalPixels[rgbaIndex + 1],
        b: originalPixels[rgbaIndex + 2],
      };
      const background = {
        r: backgroundPixels[rgbaIndex],
        g: backgroundPixels[rgbaIndex + 1],
        b: backgroundPixels[rgbaIndex + 2],
      };
      const observedLuma = luma(observed.r, observed.g, observed.b);
      const backgroundLuma = luma(background.r, background.g, background.b);
      const observedSaturation = saturation(observed.r, observed.g, observed.b);
      const globalLift = observedLuma - referenceLuma;
      const globalSaturationDrop = referenceSaturation - observedSaturation;
      const lift = observedLuma - backgroundLuma;
      const saturationDrop =
        saturation(background.r, background.g, background.b) -
        observedSaturation;
      const minimumChannelLift =
        Math.min(observed.r, observed.g, observed.b) -
        Math.min(background.r, background.g, background.b);
      const signal =
        lift +
        Math.max(0, saturationDrop) * 0.2 +
        Math.max(0, minimumChannelLift) * 0.16;
      const channelFloor = Math.min(observed.r, observed.g, observed.b) > 30;
      const desaturatedEnough = saturationDrop > -8 || globalSaturationDrop > -10 || observedSaturation < 18;
      const globallyBrightOverlay =
        globalLift > globalSoftLiftThreshold &&
        globalSaturationDrop > -14 &&
        Math.min(observed.r, observed.g, observed.b) > 42;
      const isCandidate =
        channelFloor &&
        desaturatedEnough &&
        globallyBrightOverlay &&
        signal > (quality === 'hq' ? -1.5 : 0.5) &&
        lift > (quality === 'hq' ? -2.5 : -1);

      if (!isCandidate) {
        continue;
      }

      candidate[index] = 1;
      signals[index] = signal;
      candidateCount += 1;

      if (
        globalLift > globalSeedLiftThreshold ||
        (signal > seedThreshold && globalLift > globalSoftLiftThreshold * 0.82 && lift > (quality === 'hq' ? 0 : 1.5))
      ) {
        seed[index] = 1;
        seedCount += 1;
      }
    }
  }

  if (
    candidateCount < Math.max(24, Math.round(selectionArea * 0.006)) ||
    seedCount < Math.max(8, Math.round(selectionArea * 0.0012))
  ) {
    return null;
  }

  const visited = new Uint8Array(selectionMask.length);
  const result = new Uint8Array(selectionMask.length);
  const minComponentArea = Math.max(18, Math.round(selectionArea * 0.0015));
  const acceptedComponents: Array<{
    indexes: number[];
    area: number;
    left: number;
    top: number;
    right: number;
    bottom: number;
  }> = [];

  for (let index = 0; index < candidate.length; index += 1) {
    if (candidate[index] === 0 || visited[index] !== 0) {
      continue;
    }

    const queue = [index];
    const component: number[] = [];
    let componentSeedCount = 0;
    let strongCount = 0;
    let signalSum = 0;
    let left = width;
    let right = 0;
    let top = height;
    let bottom = 0;
    visited[index] = 1;

    while (queue.length > 0) {
      const currentIndex = queue.pop()!;
      component.push(currentIndex);
      const x = currentIndex % width;
      const y = Math.floor(currentIndex / width);
      left = Math.min(left, x);
      right = Math.max(right, x);
      top = Math.min(top, y);
      bottom = Math.max(bottom, y);

      const signal = signals[currentIndex];
      signalSum += signal;
      if (seed[currentIndex] !== 0) {
        componentSeedCount += 1;
      }
      if (signal > strongThreshold) {
        strongCount += 1;
      }

      for (let offsetY = -1; offsetY <= 1; offsetY += 1) {
        const sampleY = y + offsetY;
        if (sampleY < bounds.top || sampleY > bounds.bottom) {
          continue;
        }

        for (let offsetX = -1; offsetX <= 1; offsetX += 1) {
          const sampleX = x + offsetX;
          if (sampleX < bounds.left || sampleX > bounds.right || (offsetX === 0 && offsetY === 0)) {
            continue;
          }

          const sampleIndex = sampleY * width + sampleX;
          if (candidate[sampleIndex] === 0 || visited[sampleIndex] !== 0) {
            continue;
          }

          visited[sampleIndex] = 1;
          queue.push(sampleIndex);
        }
      }
    }

    const area = component.length;
    if (area < minComponentArea || componentSeedCount === 0) {
      continue;
    }

    const componentWidth = right - left + 1;
    const componentHeight = bottom - top + 1;
    const componentAspect = Math.max(
      componentWidth / Math.max(1, componentHeight),
      componentHeight / Math.max(1, componentWidth),
    );
    const componentFillRatio = area / Math.max(1, componentWidth * componentHeight);
    const meanSignal = signalSum / area;
    const strongRatio = strongCount / area;
    const largeCoherentShape = area > selectionArea * 0.025 && componentAspect < 3.8 && meanSignal > 4.2;
    const compactSignalShape =
      componentAspect < 4.8 &&
      componentFillRatio > 0.1 &&
      (strongRatio > 0.045 || meanSignal > (quality === 'hq' ? 7 : 9));

    if (!largeCoherentShape && !compactSignalShape) {
      continue;
    }

    acceptedComponents.push({ indexes: component, area, left, top, right, bottom });
  }

  if (acceptedComponents.length === 0) {
    return null;
  }

  let anchorComponent = acceptedComponents[0];
  for (const component of acceptedComponents) {
    if (component.area > anchorComponent.area) {
      anchorComponent = component;
    }
  }

  const anchorPadding = quality === 'hq' ? 8 : 10;
  for (const component of acceptedComponents) {
    const nearAnchor =
      component.right >= anchorComponent.left - anchorPadding &&
      component.left <= anchorComponent.right + anchorPadding &&
      component.bottom >= anchorComponent.top - anchorPadding &&
      component.top <= anchorComponent.bottom + anchorPadding;
    const substantialNeighbor = component.area >= Math.max(120, anchorComponent.area * 0.012);
    if (component !== anchorComponent && (!nearAnchor || !substantialNeighbor)) {
      continue;
    }

    for (const componentIndex of component.indexes) {
      result[componentIndex] = 1;
    }
  }

  let resultArea = countMasked(result);
  if (resultArea < Math.max(32, Math.round(selectionArea * 0.01))) {
    return null;
  }

  if (resultArea > selectionArea * 0.66) {
    for (let index = 0; index < result.length; index += 1) {
      result[index] = seed[index];
    }
    resultArea = countMasked(result);
    if (resultArea < Math.max(32, Math.round(selectionArea * 0.006))) {
      return null;
    }
  }

  const closed = dilateMask(result, width, height, quality === 'hq' ? 1 : 1);
  for (let index = 0; index < closed.length; index += 1) {
    if (selectionMask[index] === 0) {
      closed[index] = 0;
    }
  }

  const closedArea = countMasked(closed);
  if (closedArea < Math.max(32, Math.round(selectionArea * 0.01)) || closedArea > selectionArea * 0.72) {
    return null;
  }

  return closed;
}

function computeTransparentOverlayLocalStats(
  originalPixels: Uint8ClampedArray,
  backgroundPixels: Uint8ClampedArray,
  mask: Uint8Array,
) {
  let selectedCount = 0;
  let positiveCount = 0;
  let strongCount = 0;
  let negativeCount = 0;
  let positiveLift = 0;

  for (let index = 0; index < mask.length; index += 1) {
    if (mask[index] === 0) {
      continue;
    }

    const rgbaIndex = index * 4;
    const observed = {
      r: originalPixels[rgbaIndex],
      g: originalPixels[rgbaIndex + 1],
      b: originalPixels[rgbaIndex + 2],
    };
    const background = {
      r: backgroundPixels[rgbaIndex],
      g: backgroundPixels[rgbaIndex + 1],
      b: backgroundPixels[rgbaIndex + 2],
    };
    const lift = luma(observed.r, observed.g, observed.b) - luma(background.r, background.g, background.b);
    const saturationDrop =
      saturation(background.r, background.g, background.b) -
      saturation(observed.r, observed.g, observed.b);
    const minimumChannelLift =
      Math.min(observed.r, observed.g, observed.b) -
      Math.min(background.r, background.g, background.b);
    const overlaySignal =
      lift +
      Math.max(0, saturationDrop) * 0.16 +
      Math.max(0, minimumChannelLift) * 0.1;

    selectedCount += 1;
    if (overlaySignal > 3.5) {
      positiveCount += 1;
      positiveLift += Math.max(0, overlaySignal);
    }
    if (overlaySignal > 11) {
      strongCount += 1;
    }
    if (overlaySignal < -5.5 || lift < -7) {
      negativeCount += 1;
    }
  }

  return {
    positiveRatio: selectedCount === 0 ? 0 : positiveCount / selectedCount,
    strongRatio: selectedCount === 0 ? 0 : strongCount / selectedCount,
    negativeRatio: selectedCount === 0 ? 0 : negativeCount / selectedCount,
    meanPositiveLift: positiveCount === 0 ? 0 : positiveLift / positiveCount,
  };
}

function computeMeanLocalLumaVariance(
  pixels: Uint8ClampedArray,
  mask: Uint8Array,
  width: number,
  height: number,
  sampleStep: number,
) {
  let count = 0;
  let total = 0;

  for (let index = 0; index < mask.length; index += sampleStep) {
    if (mask[index] === 0) {
      continue;
    }

    const x = index % width;
    const y = Math.floor(index / width);
    total += localLumaVariance(pixels, width, height, x, y, 1);
    count += 1;
  }

  return count === 0 ? 0 : total / count;
}

function subtractMasks(base: Uint8Array, remove: Uint8Array) {
  const result = new Uint8Array(base.length);

  for (let index = 0; index < base.length; index += 1) {
    result[index] = base[index] !== 0 && remove[index] === 0 ? 1 : 0;
  }

  return result;
}

function erodeMask(mask: Uint8Array, width: number, height: number, iterations: number) {
  let current = new Uint8Array(mask);

  for (let iteration = 0; iteration < iterations; iteration += 1) {
    const next = new Uint8Array(current);

    for (let index = 0; index < current.length; index += 1) {
      if (current[index] === 0) {
        continue;
      }

      const x = index % width;
      const y = Math.floor(index / width);
      let shrink = false;

      for (let offsetY = -1; offsetY <= 1 && !shrink; offsetY += 1) {
        const sampleY = y + offsetY;
        if (sampleY < 0 || sampleY >= height) {
          shrink = true;
          break;
        }

        for (let offsetX = -1; offsetX <= 1; offsetX += 1) {
          const sampleX = x + offsetX;
          if (sampleX < 0 || sampleX >= width || current[sampleY * width + sampleX] === 0) {
            shrink = true;
            break;
          }
        }
      }

      if (shrink) {
        next[index] = 0;
      }
    }

    current = next;
  }

  return current;
}

function isPatchKnown(mask: Uint8Array, width: number, height: number, centerX: number, centerY: number, patchRadius: number) {
  for (let offsetY = -patchRadius; offsetY <= patchRadius; offsetY += 1) {
    const sampleY = centerY + offsetY;
    if (sampleY < 0 || sampleY >= height) {
      return false;
    }

    for (let offsetX = -patchRadius; offsetX <= patchRadius; offsetX += 1) {
      const sampleX = centerX + offsetX;
      if (sampleX < 0 || sampleX >= width || mask[sampleY * width + sampleX] !== 0) {
        return false;
      }
    }
  }

  return true;
}

function imageDataToMask(maskImage: ImageData) {
  const result = new Uint8Array(maskImage.width * maskImage.height);
  for (let index = 0; index < result.length; index += 1) {
    result[index] = maskImage.data[index * 4 + 3] > 0 ? 1 : 0;
  }
  return result;
}

function dilateMask(mask: Uint8Array, width: number, height: number, iterations: number) {
  let current = new Uint8Array(mask);

  for (let iteration = 0; iteration < iterations; iteration += 1) {
    const next = new Uint8Array(current);

    for (let index = 0; index < current.length; index += 1) {
      if (current[index] !== 0) {
        continue;
      }

      const x = index % width;
      const y = Math.floor(index / width);
      let expand = false;

      for (let offsetY = -1; offsetY <= 1 && !expand; offsetY += 1) {
        const sampleY = y + offsetY;
        if (sampleY < 0 || sampleY >= height) {
          continue;
        }

        for (let offsetX = -1; offsetX <= 1; offsetX += 1) {
          const sampleX = x + offsetX;
          if (sampleX < 0 || sampleX >= width) {
            continue;
          }

          if (current[sampleY * width + sampleX] !== 0) {
            expand = true;
            break;
          }
        }
      }

      if (expand) {
        next[index] = 1;
      }
    }

    current = next;
  }

  return current;
}

function computeDistanceMap(mask: Uint8Array, width: number, height: number, targetValue: 0 | 1) {
  const distance = new Float32Array(mask.length);
  const diagonal = Math.SQRT2;
  const infinity = 1e9;

  for (let index = 0; index < mask.length; index += 1) {
    distance[index] = mask[index] === targetValue ? 0 : infinity;
  }

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const index = y * width + x;
      let best = distance[index];

      if (x > 0) {
        best = Math.min(best, distance[index - 1] + 1);
      }
      if (y > 0) {
        best = Math.min(best, distance[index - width] + 1);
      }
      if (x > 0 && y > 0) {
        best = Math.min(best, distance[index - width - 1] + diagonal);
      }
      if (x + 1 < width && y > 0) {
        best = Math.min(best, distance[index - width + 1] + diagonal);
      }

      distance[index] = best;
    }
  }

  for (let y = height - 1; y >= 0; y -= 1) {
    for (let x = width - 1; x >= 0; x -= 1) {
      const index = y * width + x;
      let best = distance[index];

      if (x + 1 < width) {
        best = Math.min(best, distance[index + 1] + 1);
      }
      if (y + 1 < height) {
        best = Math.min(best, distance[index + width] + 1);
      }
      if (x + 1 < width && y + 1 < height) {
        best = Math.min(best, distance[index + width + 1] + diagonal);
      }
      if (x > 0 && y + 1 < height) {
        best = Math.min(best, distance[index + width - 1] + diagonal);
      }

      distance[index] = best;
    }
  }

  return distance;
}

function getBinaryMaskBounds(mask: Uint8Array, width: number, height: number) {
  let left = width;
  let top = height;
  let right = -1;
  let bottom = -1;

  for (let index = 0; index < mask.length; index += 1) {
    if (mask[index] === 0) {
      continue;
    }

    const x = index % width;
    const y = Math.floor(index / width);
    left = Math.min(left, x);
    top = Math.min(top, y);
    right = Math.max(right, x);
    bottom = Math.max(bottom, y);
  }

  if (right < left || bottom < top) {
    return null;
  }

  return { left, top, right, bottom };
}

function resizeMaskArray(
  mask: Uint8Array,
  sourceWidth: number,
  sourceHeight: number,
  targetWidth: number,
  targetHeight: number,
) {
  const sourceImage = new ImageData(sourceWidth, sourceHeight);
  for (let index = 0; index < mask.length; index += 1) {
    const alpha = mask[index] !== 0 ? 255 : 0;
    const rgbaIndex = index * 4;
    sourceImage.data[rgbaIndex] = 255;
    sourceImage.data[rgbaIndex + 1] = 255;
    sourceImage.data[rgbaIndex + 2] = 255;
    sourceImage.data[rgbaIndex + 3] = alpha;
  }

  return imageDataToMask(resizeImageData(sourceImage, targetWidth, targetHeight, false));
}

function hasKnownNeighbor(mask: Uint8Array, width: number, height: number, index: number) {
  const x = index % width;
  const y = Math.floor(index / width);

  for (let offsetY = -1; offsetY <= 1; offsetY += 1) {
    const sampleY = y + offsetY;
    if (sampleY < 0 || sampleY >= height) {
      continue;
    }

    for (let offsetX = -1; offsetX <= 1; offsetX += 1) {
      const sampleX = x + offsetX;
      if (sampleX < 0 || sampleX >= width || (offsetX === 0 && offsetY === 0)) {
        continue;
      }

      if (mask[sampleY * width + sampleX] === 0) {
        return true;
      }
    }
  }

  return false;
}

function resizeForProcessing(source: ImageData, mask: ImageData, quality: ImageRepairQuality): WorkingRoi {
  const maxPixels = quality === 'hq' ? 320_000 : 220_000;
  const maxDimension = quality === 'hq' ? 960 : 760;
  const currentPixels = source.width * source.height;
  const currentMaxDimension = Math.max(source.width, source.height);

  let scale = 1;

  if (currentPixels > maxPixels) {
    scale = Math.min(scale, Math.sqrt(maxPixels / currentPixels));
  }

  if (currentMaxDimension > maxDimension) {
    scale = Math.min(scale, maxDimension / currentMaxDimension);
  }

  if (scale >= 1) {
    return { source, mask, scale: 1 };
  }

  const nextWidth = Math.max(1, Math.round(source.width * scale));
  const nextHeight = Math.max(1, Math.round(source.height * scale));

  return {
    source: resizeImageData(source, nextWidth, nextHeight, true),
    mask: resizeImageData(mask, nextWidth, nextHeight, false),
    scale,
  };
}

function resizeImageData(imageData: ImageData, width: number, height: number, smoothing: boolean) {
  const sourceCanvas = document.createElement('canvas');
  sourceCanvas.width = imageData.width;
  sourceCanvas.height = imageData.height;
  sourceCanvas.getContext('2d')?.putImageData(imageData, 0, 0);

  const targetCanvas = document.createElement('canvas');
  targetCanvas.width = width;
  targetCanvas.height = height;
  const targetContext = targetCanvas.getContext('2d');
  if (!targetContext) {
    throw new Error('无法缩放局部画面。');
  }

  targetContext.imageSmoothingEnabled = smoothing;
  targetContext.clearRect(0, 0, width, height);
  targetContext.drawImage(sourceCanvas, 0, 0, width, height);

  if (!smoothing) {
    const result = targetContext.getImageData(0, 0, width, height);
    for (let index = 0; index < result.data.length; index += 4) {
      result.data[index] = 255;
      result.data[index + 1] = 255;
      result.data[index + 2] = 255;
      result.data[index + 3] = result.data[index + 3] > 8 ? 255 : 0;
    }
    return result;
  }

  return targetContext.getImageData(0, 0, width, height);
}

function getMaskBounds(mask: ImageData): MaskBounds | null {
  const { width, height, data } = mask;
  let left = width;
  let top = height;
  let right = -1;
  let bottom = -1;

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const alphaIndex = (y * width + x) * 4 + 3;
      if (data[alphaIndex] === 0) {
        continue;
      }

      left = Math.min(left, x);
      top = Math.min(top, y);
      right = Math.max(right, x);
      bottom = Math.max(bottom, y);
    }
  }

  if (right < left || bottom < top) {
    return null;
  }

  return { left, top, right, bottom };
}

function expandBounds(bounds: MaskBounds, imageWidth: number, imageHeight: number, margin: number): RoiRect {
  const x = Math.max(0, bounds.left - margin);
  const y = Math.max(0, bounds.top - margin);
  const right = Math.min(imageWidth, bounds.right + margin + 1);
  const bottom = Math.min(imageHeight, bounds.bottom + margin + 1);

  return {
    x,
    y,
    width: Math.max(1, right - x),
    height: Math.max(1, bottom - y),
  };
}

function resolveMargin(quality: ImageRepairQuality, strength: number) {
  const normalized = clampStrength(strength) / 100;
  return quality === 'hq'
    ? Math.round(18 + normalized * 42)
    : Math.round(10 + normalized * 24);
}

function resolveDilateIterations(quality: ImageRepairQuality, strength: number, scale: number) {
  const normalized = clampStrength(strength) / 100;
  const raw = quality === 'hq' ? 3 + normalized * 5.2 : 1.8 + normalized * 2.8;
  return Math.max(1, Math.round(raw * Math.max(scale, 0.65)));
}

function resolveSearchRadius(quality: ImageRepairQuality, strength: number, width: number, height: number) {
  const normalized = clampStrength(strength) / 100;
  const base = quality === 'hq' ? 11 : 7;
  const maxBySize = Math.max(6, Math.min(22, Math.round(Math.max(width, height) / 12)));
  return Math.min(maxBySize, base + Math.round(normalized * (quality === 'hq' ? 7 : 4)));
}

function resolvePatchRadius(quality: ImageRepairQuality, strength: number) {
  const normalized = clampStrength(strength) / 100;
  return quality === 'hq'
    ? Math.max(2, Math.min(3, Math.round(2 + normalized * 1.2)))
    : Math.max(1, Math.min(2, Math.round(1 + normalized * 0.9)));
}

function countMasked(mask: Uint8Array) {
  let total = 0;
  for (const value of mask) {
    total += value;
  }
  return total;
}

function resolveProtectedCoreMask(mask: Uint8Array, width: number, height: number, quality: ImageRepairQuality) {
  const totalMasked = countMasked(mask);
  const preferredIterations = quality === 'hq' && totalMasked > 9_000 ? 2 : 1;
  let core = erodeMask(mask, width, height, preferredIterations);

  if (countMasked(core) < Math.max(12, Math.round(totalMasked * 0.12))) {
    core = preferredIterations > 1 ? erodeMask(mask, width, height, 1) : new Uint8Array(mask);
  }

  if (countMasked(core) === 0) {
    return new Uint8Array(mask);
  }

  return core;
}

function clampStrength(value: number) {
  return Math.max(1, Math.min(100, Math.round(value)));
}

function clampByte(value: number) {
  return Math.max(0, Math.min(255, Math.round(value)));
}

function clampNumber(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}

function resolveLoopYieldEvery(maskedPixelCount: number, quality: ImageRepairQuality) {
  const base = quality === 'hq' ? 360 : 520;
  return Math.max(96, Math.min(720, Math.round(base - Math.min(maskedPixelCount, 90_000) / 220)));
}

function shouldUseLightweightRemovalPipeline({
  selectionMask,
  baseMask,
  maskedPixelCount,
  width,
  height,
  quality,
}: {
  selectionMask: Uint8Array;
  baseMask: Uint8Array;
  maskedPixelCount: number;
  width: number;
  height: number;
  quality: ImageRepairQuality;
}) {
  if (FORCE_LIGHTWEIGHT_BROWSER_PIPELINE) {
    return true;
  }

  const selectionArea = countMasked(selectionMask);
  const refinedArea = countMasked(baseMask);
  const totalPixels = width * height;
  const selectionBounds = getBinaryMaskBounds(selectionMask, width, height);

  if (maskedPixelCount > (quality === 'hq' ? 8_000 : 5_500)) {
    return true;
  }

  if (totalPixels > (quality === 'hq' ? 140_000 : 100_000)) {
    return true;
  }

  if (!selectionBounds) {
    return false;
  }

  const rectArea = Math.max(1, (selectionBounds.right - selectionBounds.left + 1) * (selectionBounds.bottom - selectionBounds.top + 1));
  const fillRatio = selectionArea / rectArea;
  const refinedRatio = refinedArea / Math.max(selectionArea, 1);

  return fillRatio > 0.9 || refinedRatio < 0.82;
}

function resolveGrabCutScale(width: number, height: number, selectionArea: number, quality: ImageRepairQuality) {
  if (!ENABLE_OPENCV_BROWSER_REFINEMENT) {
    return 0;
  }

  const totalPixels = width * height;
  const maxPixels = quality === 'hq' ? 80_000 : 52_000;
  const maxSelectionArea = quality === 'hq' ? 4_500 : 2_800;
  const maxDimension = quality === 'hq' ? 320 : 260;

  if (totalPixels > maxPixels * 2.2 || selectionArea > maxSelectionArea * 1.8) {
    return 0;
  }

  let scale = 1;
  if (totalPixels > maxPixels) {
    scale = Math.min(scale, Math.sqrt(maxPixels / totalPixels));
  }

  const currentMaxDimension = Math.max(width, height);
  if (currentMaxDimension > maxDimension) {
    scale = Math.min(scale, maxDimension / currentMaxDimension);
  }

  return clampNumber(scale, 0, 1);
}

function resolveGrabCutIterations(totalPixels: number, selectionArea: number, quality: ImageRepairQuality) {
  if (totalPixels > 120_000 || selectionArea > 40_000) {
    return 1;
  }

  return quality === 'hq' ? 2 : 1;
}

function shouldUseOpenCvRefinement({
  width,
  height,
  maskedPixelCount,
  quality,
}: {
  width: number;
  height: number;
  maskedPixelCount: number;
  quality: ImageRepairQuality;
}) {
  if (!ENABLE_OPENCV_BROWSER_REFINEMENT) {
    return false;
  }

  const totalPixels = width * height;
  const maskRatio = maskedPixelCount / Math.max(totalPixels, 1);
  const maxPixels = quality === 'hq' ? 360_000 : 180_000;
  const maxMaskedPixels = quality === 'hq' ? 54_000 : 24_000;
  const maxMaskRatio = quality === 'hq' ? 0.2 : 0.14;

  return totalPixels <= maxPixels && maskedPixelCount <= maxMaskedPixels && maskRatio <= maxMaskRatio;
}

function smoothstep(edge0: number, edge1: number, value: number) {
  if (edge0 === edge1) {
    return value < edge0 ? 0 : 1;
  }

  const t = clampNumber((value - edge0) / (edge1 - edge0), 0, 1);
  return t * t * (3 - 2 * t);
}

function blendChannel(base: number, target: number, alpha: number) {
  return clampByte(base * (1 - alpha) + target * alpha);
}

function colorDistanceToMean(pixels: Uint8ClampedArray, rgbaIndex: number, mean: Rgb) {
  const redDelta = pixels[rgbaIndex] - mean.r;
  const greenDelta = pixels[rgbaIndex + 1] - mean.g;
  const blueDelta = pixels[rgbaIndex + 2] - mean.b;
  return Math.sqrt(redDelta * redDelta + greenDelta * greenDelta + blueDelta * blueDelta);
}

function colorDistance(left: Rgb, right: Rgb) {
  const redDelta = left.r - right.r;
  const greenDelta = left.g - right.g;
  const blueDelta = left.b - right.b;
  return Math.sqrt(redDelta * redDelta + greenDelta * greenDelta + blueDelta * blueDelta);
}

function colorDistanceBetweenPixels(pixels: Uint8ClampedArray, leftRgbaIndex: number, rightRgbaIndex: number) {
  const redDelta = pixels[leftRgbaIndex] - pixels[rightRgbaIndex];
  const greenDelta = pixels[leftRgbaIndex + 1] - pixels[rightRgbaIndex + 1];
  const blueDelta = pixels[leftRgbaIndex + 2] - pixels[rightRgbaIndex + 2];
  return Math.sqrt(redDelta * redDelta + greenDelta * greenDelta + blueDelta * blueDelta);
}

function luma(r: number, g: number, b: number) {
  return r * 0.299 + g * 0.587 + b * 0.114;
}

function saturation(r: number, g: number, b: number) {
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  return max === 0 ? 0 : (max - min) / max * 255;
}

function waitForFrame() {
  return new Promise<void>((resolve) => window.requestAnimationFrame(() => resolve()));
}

async function yieldToBrowser() {
  await new Promise<void>((resolve) => window.setTimeout(resolve, 0));
  await waitForFrame();
}

function promiseWithTimeout<T>(promise: Promise<T>, timeoutMs: number) {
  return new Promise<T>((resolve, reject) => {
    const timeoutId = window.setTimeout(() => {
      reject(new Error('Timed out'));
    }, timeoutMs);

    promise
      .then((value) => {
        window.clearTimeout(timeoutId);
        resolve(value);
      })
      .catch((error) => {
        window.clearTimeout(timeoutId);
        reject(error);
      });
  });
}
