import { spawn } from 'node:child_process';
import { createReadStream, existsSync } from 'node:fs';
import { mkdir, readdir, rename, rm, stat } from 'node:fs/promises';
import path from 'node:path';
import { randomUUID } from 'node:crypto';
import { fileURLToPath } from 'node:url';

import ffmpegInstaller from '@ffmpeg-installer/ffmpeg';
import ffprobeInstaller from '@ffprobe-installer/ffprobe';

export type TaskPhase = 'processing' | 'completed' | 'failed';

export interface VideoTask {
  id: string;
  sourceName: string;
  outputName: string;
  phase: TaskPhase;
  progress: number;
  message: string;
  createdAt: string;
  updatedAt: string;
  durationSeconds?: number;
  width?: number;
  height?: number;
  downloadUrl?: string;
}

const serverRoot = path.resolve(fileURLToPath(new URL('..', import.meta.url)));
const workspaceRoot = path.resolve(serverRoot, '..');
const runtimeRoot = path.join(workspaceRoot, 'runtime');
const uploadRoot = path.join(runtimeRoot, 'uploads');
const tasksRoot = path.join(runtimeRoot, 'tasks');
const retentionMs = 12 * 60 * 60 * 1000;

const taskStore = new Map<string, VideoTask>();

export function getUploadRoot() {
  return uploadRoot;
}

export async function ensureVideoRuntime() {
  await mkdir(uploadRoot, { recursive: true });
  await mkdir(tasksRoot, { recursive: true });
}

export function listTasks() {
  return taskStore;
}

export function createTask(sourceName: string) {
  const id = randomUUID();
  const baseName = sanitizeBaseName(path.parse(sourceName).name || 'video');
  const task: VideoTask = {
    id,
    sourceName,
    outputName: `${baseName}-clean.mp4`,
    phase: 'processing',
    progress: 2,
    message: '视频已上传，后端开始处理',
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
  };

  taskStore.set(id, task);
  return task;
}

export async function moveUploadedFiles(task: VideoTask, videoFile: Express.Multer.File, maskFile: Express.Multer.File) {
  const taskDir = path.join(tasksRoot, task.id);
  await mkdir(taskDir, { recursive: true });

  const inputExtension = path.extname(videoFile.originalname) || '.mp4';
  const inputPath = path.join(taskDir, `input${inputExtension}`);
  const maskPath = path.join(taskDir, 'mask.png');

  await rename(videoFile.path, inputPath);
  await rename(maskFile.path, maskPath);

  return {
    taskDir,
    inputPath,
    maskPath,
    outputPath: path.join(taskDir, 'output.mp4'),
  };
}

export async function startVideoTask(task: VideoTask, inputPath: string, maskPath: string, outputPath: string) {
  try {
    const mediaInfo = await probeVideo(inputPath);
    task.durationSeconds = mediaInfo.durationSeconds;
    task.width = mediaInfo.width;
    task.height = mediaInfo.height;
    updateTask(task, {
      message: `后端处理中，适合固定位置水印，视频时长 ${mediaInfo.durationSeconds.toFixed(1)} 秒`,
      progress: 5,
    });

    await runFfmpeg(task, inputPath, maskPath, outputPath, mediaInfo.durationSeconds);

    updateTask(task, {
      phase: 'completed',
      progress: 100,
      message: '处理完成，可导出视频',
      downloadUrl: `/api/video/tasks/${task.id}/download`,
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : '视频处理失败';
    updateTask(task, {
      phase: 'failed',
      progress: 100,
      message,
      downloadUrl: undefined,
    });
  }
}

export async function getTaskOutput(taskId: string) {
  const task = taskStore.get(taskId);
  if (!task) {
    return null;
  }

  const outputPath = path.join(tasksRoot, taskId, 'output.mp4');
  if (!existsSync(outputPath)) {
    return null;
  }

  return {
    task,
    outputPath,
    stream: createReadStream(outputPath),
  };
}

export function scheduleTaskCleanup() {
  const timer = setInterval(() => {
    void cleanupExpiredTasks();
  }, 60 * 60 * 1000);

  timer.unref();
}

async function cleanupExpiredTasks() {
  const now = Date.now();
  const entries = await readdir(tasksRoot, { withFileTypes: true });

  for (const entry of entries) {
    if (!entry.isDirectory()) {
      continue;
    }

    const taskId = entry.name;
    const taskPath = path.join(tasksRoot, taskId);
    const taskStats = await stat(taskPath);
    const task = taskStore.get(taskId);

    if (task?.phase === 'processing') {
      continue;
    }

    if (now - taskStats.mtimeMs < retentionMs) {
      continue;
    }

    taskStore.delete(taskId);
    await rm(taskPath, { recursive: true, force: true });
  }
}

function updateTask(task: VideoTask, patch: Partial<VideoTask>) {
  Object.assign(task, patch, { updatedAt: new Date().toISOString() });
}

function sanitizeBaseName(value: string) {
  return value
    .replace(/[^\w\u4e00-\u9fa5-]+/g, '-')
    .replace(/-{2,}/g, '-')
    .replace(/^-|-$/g, '') || 'video';
}

async function probeVideo(filePath: string) {
  const ffprobePath = ffprobeInstaller.path;

  return new Promise<{ durationSeconds: number; width: number; height: number }>((resolve, reject) => {
    const child = spawn(ffprobePath, ['-v', 'error', '-print_format', 'json', '-show_streams', '-show_format', filePath], {
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
        reject(new Error(stderr || '无法读取视频信息'));
        return;
      }

      try {
        const payload = JSON.parse(stdout) as {
          streams?: Array<{ codec_type?: string; width?: number; height?: number; duration?: string }>;
          format?: { duration?: string };
        };
        const videoStream = payload.streams?.find((stream) => stream.codec_type === 'video');

        if (!videoStream?.width || !videoStream?.height) {
          reject(new Error('无法识别视频分辨率'));
          return;
        }

        const durationRaw = payload.format?.duration ?? videoStream.duration ?? '0';
        resolve({
          durationSeconds: Number.parseFloat(durationRaw),
          width: videoStream.width,
          height: videoStream.height,
        });
      } catch (error) {
        reject(error);
      }
    });
  });
}

async function runFfmpeg(
  task: VideoTask,
  inputPath: string,
  maskPath: string,
  outputPath: string,
  durationSeconds: number,
) {
  const ffmpegPath = ffmpegInstaller.path;
  const taskDir = path.dirname(outputPath);
  const lastLogs: string[] = [];

  const args = [
    '-y',
    '-i',
    path.basename(inputPath),
    '-vf',
    `removelogo=f=${path.basename(maskPath)}`,
    '-map',
    '0:v:0',
    '-map',
    '0:a?',
    '-c:v',
    'libx264',
    '-preset',
    'medium',
    '-crf',
    '18',
    '-pix_fmt',
    'yuv420p',
    '-movflags',
    '+faststart',
    '-c:a',
    'aac',
    '-b:a',
    '192k',
    '-progress',
    'pipe:1',
    '-nostats',
    path.basename(outputPath),
  ];

  return new Promise<void>((resolve, reject) => {
    const child = spawn(ffmpegPath, args, {
      cwd: taskDir,
      windowsHide: true,
    });

    let stdoutBuffer = '';

    child.stdout.on('data', (chunk: Buffer) => {
      stdoutBuffer += chunk.toString();

      let newlineIndex = stdoutBuffer.indexOf('\n');
      while (newlineIndex >= 0) {
        const rawLine = stdoutBuffer.slice(0, newlineIndex).trim();
        stdoutBuffer = stdoutBuffer.slice(newlineIndex + 1);

        if (rawLine) {
          handleProgressLine(task, rawLine, durationSeconds);
        }

        newlineIndex = stdoutBuffer.indexOf('\n');
      }
    });

    child.stderr.on('data', (chunk: Buffer) => {
      const lines = chunk.toString().split(/\r?\n/).filter(Boolean);
      for (const line of lines) {
        lastLogs.push(line);
        if (lastLogs.length > 24) {
          lastLogs.shift();
        }
      }
    });

    child.once('error', (error) => {
      reject(error);
    });

    child.once('close', (code) => {
      if (code === 0) {
        resolve();
        return;
      }

      const detail = lastLogs.at(-1) ?? 'FFmpeg 处理失败';
      reject(new Error(detail));
    });
  });
}

function handleProgressLine(task: VideoTask, line: string, durationSeconds: number) {
  const [key, value] = line.split('=');
  if (!key || !value) {
    return;
  }

  if (key === 'out_time') {
    const elapsed = parseTimecodeToSeconds(value);
    const ratio = durationSeconds > 0 ? elapsed / durationSeconds : 0;
    updateTask(task, {
      progress: Math.max(task.progress, Math.min(98, Math.round(ratio * 100))),
      message: '后端正在处理视频，请稍候',
    });
  }

  if (key === 'progress' && value === 'end') {
    updateTask(task, {
      progress: 99,
      message: '正在封装输出视频',
    });
  }
}

function parseTimecodeToSeconds(timecode: string) {
  const [hours, minutes, seconds] = timecode.split(':').map((segment) => Number.parseFloat(segment));
  return hours * 3600 + minutes * 60 + seconds;
}
