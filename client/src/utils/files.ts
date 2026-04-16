const jpegTypes = new Set(['image/jpeg', 'image/jpg']);
const supportedImageTypes = new Set(['image/png', 'image/jpeg', 'image/jpg', 'image/webp', 'image/bmp']);
const supportedVideoTypes = new Set(['video/mp4', 'video/quicktime', 'video/webm', 'video/x-matroska']);
const supportedImageExtensions = ['.png', '.jpg', '.jpeg', '.webp', '.bmp'];
const supportedVideoExtensions = ['.mp4', '.mov', '.webm', '.mkv'];

export function isSupportedImageFile(file: File) {
  return supportedImageTypes.has(file.type) || supportedImageExtensions.some((extension) => file.name.toLowerCase().endsWith(extension));
}

export function isSupportedVideoFile(file: File) {
  return supportedVideoTypes.has(file.type) || supportedVideoExtensions.some((extension) => file.name.toLowerCase().endsWith(extension));
}

export function formatBytes(bytes: number) {
  if (bytes < 1024) {
    return `${bytes} B`;
  }

  const units = ['KB', 'MB', 'GB'];
  let value = bytes / 1024;
  let unitIndex = 0;

  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }

  return `${value.toFixed(value >= 100 ? 0 : 1)} ${units[unitIndex]}`;
}

export function downloadBlob(blob: Blob, filename: string) {
  const objectUrl = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = objectUrl;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(objectUrl);
}

export function sanitizeFilename(value: string) {
  return value.replace(/[\\/:*?"<>|]+/g, '-');
}

export function getPreferredImageMimeType(type: string) {
  if (jpegTypes.has(type)) {
    return 'image/jpeg';
  }

  if (type === 'image/png' || type === 'image/webp') {
    return type;
  }

  return 'image/png';
}

export function getImageExtension(type: string) {
  if (type === 'image/jpeg') {
    return 'jpg';
  }

  if (type === 'image/png') {
    return 'png';
  }

  if (type === 'image/webp') {
    return 'webp';
  }

  return 'png';
}

export function canvasToBlob(canvas: HTMLCanvasElement, type: string, quality?: number) {
  return new Promise<Blob>((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (!blob) {
        reject(new Error('导出失败'));
        return;
      }

      resolve(blob);
    }, type, quality);
  });
}
