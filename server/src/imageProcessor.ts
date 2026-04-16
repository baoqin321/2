import { spawn } from 'node:child_process';
import { createHash, randomUUID } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const serverRoot = path.resolve(fileURLToPath(new URL('..', import.meta.url)));
const classicScriptPath = path.join(serverRoot, 'scripts', 'inpaint_image.py');
const aiScriptPath = path.join(serverRoot, 'scripts', 'inpaint_image_ai.py');
const pythonBin = process.env.PYTHON_BIN ?? 'python';
const aiPythonBin = process.env.AI_PYTHON_BIN ?? (process.platform === 'win32' ? 'py' : 'python3.11');

const supportedExtensions = new Set(['.png', '.jpg', '.jpeg', '.webp', '.bmp']);
const IMAGE_PROCESS_CACHE_VERSION = 'detail-v4-shared-mask';

export interface ImageProcessResult {
  body: Buffer;
  outputName: string;
  mimeType: string;
  tempOutputPath: string | null;
}

export interface ImageProcessOptions {
  strength?: number;
  mode?: 'rect' | 'brush';
  quality?: 'fast' | 'hq';
}

interface CachedImageResult {
  body: Buffer;
  extension: string;
  mimeType: string;
  expiresAt: number;
}

const CACHE_TTL_MS = 10 * 60 * 1000;
const CACHE_MAX_ITEMS = 12;
const imageProcessCache = new Map<string, CachedImageResult>();

export async function processImageWithMask(
  imageFile: Express.Multer.File,
  maskFile: Express.Multer.File,
  options: ImageProcessOptions = {},
): Promise<ImageProcessResult> {
  const [inputBuffer, maskBuffer] = await Promise.all([
    readFile(imageFile.path),
    readFile(maskFile.path),
  ]);
  const cacheKey = createCacheKey(inputBuffer, maskBuffer, options);
  const cached = getCachedResult(cacheKey);
  const safeExtension = normalizeExtension(path.extname(imageFile.originalname));
  const outputName = `${sanitizeBaseName(path.parse(imageFile.originalname).name || 'image')}-clean`;

  if (cached) {
    return {
      body: Buffer.from(cached.body),
      mimeType: cached.mimeType,
      outputName: `${outputName}${cached.extension}`,
      tempOutputPath: null,
    };
  }

  const inputExtension = normalizeExtension(path.extname(imageFile.originalname));
  const outputPath = path.join(path.dirname(imageFile.path), `${randomUUID()}${inputExtension}`);
  const payload = await runPythonProcess(imageFile.path, maskFile.path, outputPath, options.mode, options.strength, options.quality);
  const body = await readFile(outputPath);
  const extension = path.extname(outputPath) || safeExtension;
  setCachedResult(cacheKey, {
    body,
    extension,
    mimeType: payload.mimeType,
    expiresAt: Date.now() + CACHE_TTL_MS,
  });

  return {
    body,
    mimeType: payload.mimeType,
    outputName: `${outputName}${extension}`,
    tempOutputPath: outputPath,
  };
}

function runPythonProcess(
  inputPath: string,
  maskPath: string,
  outputPath: string,
  mode?: 'rect' | 'brush',
  strength?: number,
  quality?: 'fast' | 'hq',
) {
  return new Promise<{ mimeType: string }>((resolve, reject) => {
    const normalizedMode = mode === 'brush' ? 'brush' : 'rect';
    const normalizedStrength = String(typeof strength === 'number' ? clampStrength(strength) : 55);
    const normalizedQuality = quality === 'hq' ? 'hq' : 'fast';
    const isAiMode = normalizedQuality === 'hq';
    const command = isAiMode ? aiPythonBin : pythonBin;
    const args = isAiMode
      ? buildAiArgs(inputPath, maskPath, outputPath, normalizedMode, normalizedStrength)
      : [classicScriptPath, inputPath, maskPath, outputPath, normalizedMode, normalizedStrength, normalizedQuality];

    const child = spawn(command, args, {
      env: {
        ...process.env,
        PYTHONIOENCODING: 'utf-8',
        PYTHONUTF8: '1',
      },
      windowsHide: true,
    });

    let stdout = '';
    let stderr = '';

    child.stdout.on('data', (chunk: Buffer) => {
      stdout += chunk.toString();
    });

    child.stderr.on('data', (chunk: Buffer) => {
      stderr += chunk.toString();
    });

    child.once('error', (error) => {
      reject(error);
    });

    child.once('close', (code) => {
      if (code !== 0) {
        reject(new Error(stderr.trim() || '图片处理失败'));
        return;
      }

      try {
        const payload = JSON.parse(stdout) as { mime_type: string };
        resolve({
          mimeType: payload.mime_type,
        });
      } catch (error) {
        reject(error);
      }
    });
  });
}

function buildAiArgs(inputPath: string, maskPath: string, outputPath: string, mode: 'rect' | 'brush', strength: string) {
  if (process.platform === 'win32') {
    return ['-3.11', aiScriptPath, inputPath, maskPath, outputPath, mode, strength];
  }

  return [aiScriptPath, inputPath, maskPath, outputPath, mode, strength];
}

function normalizeExtension(extension: string) {
  const normalized = extension.toLowerCase();
  return supportedExtensions.has(normalized) ? normalized : '.png';
}

function sanitizeBaseName(value: string) {
  return value
    .replace(/[^\w\u4e00-\u9fa5-]+/g, '-')
    .replace(/-{2,}/g, '-')
    .replace(/^-|-$/g, '') || 'image';
}

function clampStrength(value: number) {
  return Math.max(1, Math.min(100, Math.round(value)));
}

function createCacheKey(imageBuffer: Buffer, maskBuffer: Buffer, options: ImageProcessOptions) {
  const hash = createHash('sha1');
  hash.update(imageBuffer);
  hash.update('|');
  hash.update(maskBuffer);
  hash.update('|');
  hash.update(IMAGE_PROCESS_CACHE_VERSION);
  hash.update('|');
  hash.update(String(typeof options.strength === 'number' ? clampStrength(options.strength) : 55));
  hash.update('|');
  hash.update(options.quality === 'hq' ? 'hq' : 'fast');
  return hash.digest('hex');
}

function getCachedResult(cacheKey: string) {
  const cached = imageProcessCache.get(cacheKey);
  if (!cached) {
    return null;
  }

  if (cached.expiresAt <= Date.now()) {
    imageProcessCache.delete(cacheKey);
    return null;
  }

  imageProcessCache.delete(cacheKey);
  imageProcessCache.set(cacheKey, cached);
  return cached;
}

function setCachedResult(cacheKey: string, payload: CachedImageResult) {
  imageProcessCache.set(cacheKey, payload);

  while (imageProcessCache.size > CACHE_MAX_ITEMS) {
    const oldestKey = imageProcessCache.keys().next().value;
    if (!oldestKey) {
      break;
    }
    imageProcessCache.delete(oldestKey);
  }
}
