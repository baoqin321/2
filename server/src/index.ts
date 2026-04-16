import { existsSync } from 'node:fs';
import { rm, unlink } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import cors from 'cors';
import express from 'express';
import multer from 'multer';

import { processImageWithMask } from './imageProcessor.js';
import {
  createTask,
  ensureVideoRuntime,
  getTaskOutput,
  getUploadRoot,
  listTasks,
  moveUploadedFiles,
  scheduleTaskCleanup,
  startVideoTask,
} from './videoTasks.js';

const app = express();

await ensureVideoRuntime();
scheduleTaskCleanup();

const upload = multer({
  dest: getUploadRoot(),
  limits: {
    fileSize: 1024 * 1024 * 1024,
    files: 2,
  },
});

app.use(cors());
app.use(express.json());

app.get('/api/health', (_request, response) => {
  response.json({ ok: true });
});

app.post(
  '/api/image/process',
  upload.fields([
    { name: 'image', maxCount: 1 },
    { name: 'mask', maxCount: 1 },
  ]),
  async (request, response) => {
    const files = request.files as Record<string, Express.Multer.File[] | undefined>;
    const imageFile = files.image?.[0];
    const maskFile = files.mask?.[0];
    const strength = parseStrength(request.body?.strength);
    const mode = parseMode(request.body?.mode);
    const quality = parseQuality(request.body?.quality);

    if (!imageFile || !maskFile) {
      response.status(400).json({ message: '缺少图片文件或蒙版文件' });
      return;
    }

    let outputPath: string | null = null;

    try {
      const result = await processImageWithMask(imageFile, maskFile, { strength, mode, quality });
      outputPath = result.tempOutputPath;
      response.setHeader('Content-Type', result.mimeType);
      response.setHeader('Content-Disposition', `inline; filename*=UTF-8''${encodeURIComponent(result.outputName)}`);
      response.send(result.body);
    } catch (error) {
      const message = error instanceof Error ? error.message : '图片处理失败';
      response.status(500).json({ message });
    } finally {
      await Promise.allSettled([
        unlink(imageFile.path),
        unlink(maskFile.path),
        ...(outputPath ? [rm(outputPath, { force: true })] : []),
      ]);
    }
  },
);

app.post(
  '/api/video/tasks',
  upload.fields([
    { name: 'video', maxCount: 1 },
    { name: 'mask', maxCount: 1 },
  ]),
  async (request, response) => {
    const files = request.files as Record<string, Express.Multer.File[] | undefined>;
    const videoFile = files.video?.[0];
    const maskFile = files.mask?.[0];

    if (!videoFile || !maskFile) {
      response.status(400).json({ message: '缺少视频文件或蒙版文件' });
      return;
    }

    try {
      const task = createTask(videoFile.originalname || 'video.mp4');
      const moved = await moveUploadedFiles(task, videoFile, maskFile);

      void startVideoTask(task, moved.inputPath, moved.maskPath, moved.outputPath);

      response.status(202).json(task);
    } catch (error) {
      await Promise.allSettled([
        unlink(videoFile.path),
        unlink(maskFile.path),
      ]);

      const message = error instanceof Error ? error.message : '上传失败';
      response.status(500).json({ message });
    }
  },
);

app.get('/api/video/tasks/:taskId', (request, response) => {
  const task = listTasks().get(request.params.taskId);
  if (!task) {
    response.status(404).json({ message: '任务不存在' });
    return;
  }

  response.json(task);
});

app.get('/api/video/tasks/:taskId/download', async (request, response) => {
  const result = await getTaskOutput(request.params.taskId);

  if (!result) {
    response.status(404).json({ message: '处理结果不存在或尚未完成' });
    return;
  }

  response.setHeader('Content-Type', 'video/mp4');
  response.setHeader('Content-Disposition', `attachment; filename*=UTF-8''${encodeURIComponent(result.task.outputName)}`);
  result.stream.pipe(response);
});

const serverRoot = path.resolve(fileURLToPath(new URL('..', import.meta.url)));
const workspaceRoot = path.resolve(serverRoot, '..');
const clientDist = path.join(workspaceRoot, 'client', 'dist');

if (existsSync(clientDist)) {
  app.use(express.static(clientDist));

  app.get(/^(?!\/api).*/, (_request, response) => {
    response.sendFile(path.join(clientDist, 'index.html'));
  });
}

const port = Number(process.env.PORT ?? 3001);

app.listen(port, () => {
  console.log(`API server listening on http://localhost:${port}`);
});

function parseStrength(value: unknown) {
  if (typeof value !== 'string' || value.trim() === '') {
    return undefined;
  }

  const parsed = Number.parseInt(value, 10);
  if (!Number.isFinite(parsed)) {
    return undefined;
  }

  return Math.max(1, Math.min(100, parsed));
}

function parseMode(value: unknown): 'rect' | 'brush' {
  return value === 'brush' ? 'brush' : 'rect';
}

function parseQuality(value: unknown): 'fast' | 'hq' {
  return value === 'hq' ? 'hq' : 'fast';
}
