import { useEffect, useRef, useState } from 'react';
import type { ReactNode } from 'react';

import { AnnotationCanvas, type AnnotationCanvasHandle } from './components/AnnotationCanvas';
import type { EditMode, EditorStateSnapshot, ImageRepairQuality, WorkflowDescriptor, WorkflowPhase } from './types';
import { processImageLocally } from './utils/inpaint';
import {
  canvasToBlob,
  downloadBlob,
  formatBytes,
  getImageExtension,
  getPreferredImageMimeType,
  isSupportedImageFile,
  isSupportedVideoFile,
  sanitizeFilename,
} from './utils/files';

const IMAGE_LIMIT_BYTES = 30 * 1024 * 1024;

const workflowDescriptors: WorkflowDescriptor[] = [
  { phase: 'idle', label: '未上传', tone: 'neutral' },
  { phase: 'uploaded', label: '已上传', tone: 'active' },
  { phase: 'editing', label: '编辑中', tone: 'active' },
  { phase: 'processing', label: '处理中', tone: 'active' },
  { phase: 'completed', label: '处理完成', tone: 'success' },
  { phase: 'exporting', label: '导出中', tone: 'active' },
  { phase: 'upload-error', label: '上传失败', tone: 'danger' },
  { phase: 'process-error', label: '处理失败', tone: 'danger' },
  { phase: 'unsupported', label: '文件不支持', tone: 'danger' },
  { phase: 'too-large', label: '文件过大', tone: 'danger' },
];

interface SelectedImage {
  file: File;
  previewUrl: string;
}

interface ImageMeta {
  width: number;
  height: number;
  mimeType: string;
}

function App() {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const latestPreviewUrlRef = useRef<string | null>(null);
  const [selectedImage, setSelectedImage] = useState<SelectedImage | null>(null);
  const [landingPhase, setLandingPhase] = useState<WorkflowPhase>('idle');
  const [landingMessage, setLandingMessage] = useState(
    '当前免费版直接在浏览器本地处理图片，不上传到服务器。点击上传后会直接进入编辑区。',
  );
  const [dragOver, setDragOver] = useState(false);

  useEffect(() => {
    latestPreviewUrlRef.current = selectedImage?.previewUrl ?? null;
  }, [selectedImage]);

  useEffect(() => {
    const active = Boolean(selectedImage);
    document.body.classList.toggle('editor-active', active);
    document.documentElement.classList.toggle('editor-active', active);

    return () => {
      document.body.classList.remove('editor-active');
      document.documentElement.classList.remove('editor-active');
    };
  }, [selectedImage]);

  useEffect(() => {
    return () => {
      if (latestPreviewUrlRef.current) {
        URL.revokeObjectURL(latestPreviewUrlRef.current);
      }
    };
  }, []);

  const replaceSelectedImage = (next: SelectedImage) => {
    if (selectedImage?.previewUrl) {
      URL.revokeObjectURL(selectedImage.previewUrl);
    }

    setSelectedImage(next);
  };

  const resetToLanding = () => {
    if (selectedImage?.previewUrl) {
      URL.revokeObjectURL(selectedImage.previewUrl);
    }

    setSelectedImage(null);
    setLandingPhase('idle');
    setLandingMessage('当前免费版直接在浏览器本地处理图片，不上传到服务器。点击上传后会直接进入编辑区。');
  };

  const validateFile = (file: File) => {
    if (isSupportedImageFile(file)) {
      if (file.size > IMAGE_LIMIT_BYTES) {
        setLandingPhase('too-large');
        setLandingMessage(`图片文件超过 ${formatBytes(IMAGE_LIMIT_BYTES)}，请压缩后再试。`);
        return;
      }

      replaceSelectedImage({
        file,
        previewUrl: URL.createObjectURL(file),
      });
      setLandingPhase('idle');
      setLandingMessage('图片已就绪，马上进入编辑区。');
      return;
    }

    if (isSupportedVideoFile(file)) {
      setLandingPhase('unsupported');
      setLandingMessage('当前免费版只支持图片去水印，视频去水印暂不支持。');
      return;
    }

    setLandingPhase('unsupported');
    setLandingMessage('当前仅支持 PNG / JPG / WEBP / BMP 图片。');
  };

  const handleFiles = (files: FileList | null) => {
    const file = files?.[0];
    if (!file) {
      return;
    }

    validateFile(file);
  };

  return (
    <div className="app-shell">
      <input
        ref={inputRef}
        className="visually-hidden"
        type="file"
        accept="image/png,image/jpeg,image/webp,image/bmp"
        onChange={(event) => {
          handleFiles(event.target.files);
          event.currentTarget.value = '';
        }}
      />

      <header className="page-header">
        <div className="page-header-copy">
          <p className="eyebrow">图片去水印工具（暂不支持视频去水印）</p>
          <h1>baoqin去水印</h1>
          <p className="page-subtitle">鼠标左键拖动框选，滚轮放大缩小，右键按住进行拖动</p>
        </div>
        <a className="header-link-card" href="http://baoqin.xyz" target="_blank" rel="noreferrer">
          <span className="header-link-label">网站作者的个人博客</span>
          <strong className="header-link-url">baoqin.xyz</strong>
        </a>
      </header>

      {!selectedImage ? (
        <main className="landing-layout">
          <section
            className={`upload-card ${dragOver ? 'is-drag-over' : ''}`}
            onDragOver={(event) => {
              event.preventDefault();
              setDragOver(true);
            }}
            onDragLeave={() => setDragOver(false)}
            onDrop={(event) => {
              event.preventDefault();
              setDragOver(false);
              handleFiles(event.dataTransfer.files);
            }}
          >
            <div className="upload-card-content">
              <p className="upload-kicker">直接上传</p>
              <h2>点击上传或拖拽图片到这里</h2>
              <p>上传后直接进入大画布编辑。当前免费版为纯前端本地处理，只支持图片去水印。</p>
              <div className="upload-actions">
                <button className="button button-primary" onClick={() => inputRef.current?.click()}>
                  上传图片
                </button>
                <span>支持 PNG / JPG / WEBP / BMP，建议不超过 {formatBytes(IMAGE_LIMIT_BYTES)}</span>
              </div>
            </div>
          </section>

          <aside className="sidebar">
            <InstructionCard />
            <StatusCard phase={landingPhase} message={landingMessage} progress={landingPhase === 'idle' ? 0 : 100} />
            <StateLegend currentPhase={landingPhase} />
            <InfoCard title="版本说明">
              <p className="paragraph">当前版本为 GitHub Pages 可部署的免费版，只保留图片去水印主流程。</p>
              <p className="paragraph">去水印由浏览器本地执行，不会把图片上传到后端服务。</p>
            </InfoCard>
          </aside>
        </main>
      ) : (
        <ImageWorkspace
          key={`${selectedImage.file.name}-${selectedImage.file.lastModified}`}
          file={selectedImage.file}
          previewUrl={selectedImage.previewUrl}
          onReplace={() => inputRef.current?.click()}
          onReset={resetToLanding}
        />
      )}
    </div>
  );
}

function ImageWorkspace({
  file,
  previewUrl,
  onReplace,
  onReset,
}: {
  file: File;
  previewUrl: string;
  onReplace: () => void;
  onReset: () => void;
}) {
  const editorRef = useRef<AnnotationCanvasHandle | null>(null);
  const originalCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const sourceCanvasRef = useRef<HTMLCanvasElement | null>(null);

  const [imageMeta, setImageMeta] = useState<ImageMeta | null>(null);
  const [displaySource, setDisplaySource] = useState<CanvasImageSource | null>(null);
  const [phase, setPhase] = useState<WorkflowPhase>('uploaded');
  const [message, setMessage] = useState('图片已上传，先框选或涂抹水印区域。');
  const [progress, setProgress] = useState(0);
  const [mode, setMode] = useState<EditMode>('rect');
  const [qualityMode, setQualityMode] = useState<ImageRepairQuality>('hq');
  const [brushSize, setBrushSize] = useState(32);
  const [repairStrength, setRepairStrength] = useState(56);
  const [editorState, setEditorState] = useState<EditorStateSnapshot>({
    canUndo: false,
    canRedo: false,
    hasMask: false,
  });
  const [resultBlob, setResultBlob] = useState<Blob | null>(null);
  const [renderVersion, setRenderVersion] = useState(0);

  useEffect(() => {
    let active = true;
    const image = new Image();

    image.onload = () => {
      if (!active) {
        return;
      }

      const originalCanvas = document.createElement('canvas');
      originalCanvas.width = image.naturalWidth;
      originalCanvas.height = image.naturalHeight;
      originalCanvas.getContext('2d')?.drawImage(image, 0, 0, image.naturalWidth, image.naturalHeight);

      const sourceCanvas = document.createElement('canvas');
      sourceCanvas.width = image.naturalWidth;
      sourceCanvas.height = image.naturalHeight;
      sourceCanvas.getContext('2d')?.drawImage(originalCanvas, 0, 0, image.naturalWidth, image.naturalHeight);

      originalCanvasRef.current = originalCanvas;
      sourceCanvasRef.current = sourceCanvas;

      setImageMeta({
        width: image.naturalWidth,
        height: image.naturalHeight,
        mimeType: getPreferredImageMimeType(file.type),
      });
      setDisplaySource(sourceCanvas);
      setResultBlob(null);
      setPhase('uploaded');
      setMessage('图片已上传，先框选或涂抹水印区域。');
      setProgress(0);
      setRenderVersion((value) => value + 1);
    };

    image.onerror = () => {
      if (!active) {
        return;
      }

      setPhase('upload-error');
      setMessage('图片加载失败，请重新上传。');
    };

    image.src = previewUrl;

    return () => {
      active = false;
    };
  }, [file, previewUrl]);

  const handleUserEdit = () => {
    if (phase !== 'processing' && phase !== 'exporting') {
      setPhase('editing');
      setMessage('编辑中，可以继续缩放、拖动和修改选区。');
      setProgress(0);
    }
  };

  const handleRestoreOriginal = () => {
    const originalCanvas = originalCanvasRef.current;
    const sourceCanvas = sourceCanvasRef.current;
    if (!originalCanvas || !sourceCanvas) {
      return;
    }

    const context = sourceCanvas.getContext('2d');
    if (!context) {
      return;
    }

    sourceCanvas.width = originalCanvas.width;
    sourceCanvas.height = originalCanvas.height;
    context.clearRect(0, 0, sourceCanvas.width, sourceCanvas.height);
    context.drawImage(originalCanvas, 0, 0, sourceCanvas.width, sourceCanvas.height);

    editorRef.current?.resetMask();
    setResultBlob(null);
    setDisplaySource(sourceCanvas);
    setRenderVersion((value) => value + 1);
    setPhase('uploaded');
    setMessage('已恢复原图，可以重新框选或涂抹后再处理。');
    setProgress(0);
  };

  const exportMimeType = imageMeta?.mimeType ?? 'image/png';

  const handleProcess = async () => {
    const sourceCanvas = sourceCanvasRef.current;
    const maskCanvas = editorRef.current?.getMaskCanvas();

    if (!sourceCanvas || !maskCanvas || !imageMeta) {
      return;
    }

    if (!editorRef.current?.hasMask()) {
      setPhase('editing');
      setMessage('请先标注水印区域，再开始去水印。');
      return;
    }

    try {
      setPhase('processing');
      setMessage('正在准备局部处理区域…');
      setProgress(12);

      await new Promise<void>((resolve) => window.requestAnimationFrame(() => resolve()));

      const outputImageData = await processImageLocally({
        sourceCanvas,
        maskCanvas,
        quality: qualityMode,
        strength: repairStrength,
        onProgress: (nextProgress, nextMessage) => {
          setProgress(nextProgress);
          setMessage(nextMessage);
        },
      });

      const context = sourceCanvas.getContext('2d');
      if (!context) {
        throw new Error('无法写入处理结果。');
      }

      context.putImageData(outputImageData.imageData, outputImageData.roi.x, outputImageData.roi.y);

      const exportBlob = await canvasToBlob(
        sourceCanvas,
        exportMimeType,
        exportMimeType === 'image/jpeg' || exportMimeType === 'image/webp' ? 0.98 : undefined,
      );

      editorRef.current.resetMask();
      setResultBlob(exportBlob);
      setDisplaySource(sourceCanvas);
      setRenderVersion((value) => value + 1);
      setPhase('completed');
      setMessage('处理完成，当前选区已自动清空。继续框选会在现有结果上继续处理。');
      setProgress(100);
    } catch (error) {
      setPhase('process-error');
      setMessage(error instanceof Error ? error.message : '图片处理失败');
      setProgress(100);
    }
  };

  const handleExport = () => {
    if (!resultBlob) {
      setMessage('请先完成去水印处理，再导出结果。');
      return;
    }

    setPhase('exporting');
    setMessage('正在导出图片…');
    setProgress(100);

    downloadBlob(
      resultBlob,
      `${sanitizeFilename(file.name.replace(/\.[^.]+$/, ''))}-clean.${getImageExtension(resultBlob.type || exportMimeType)}`,
    );

    setPhase('completed');
    setMessage('图片已导出。');
  };

  return (
    <main className="workspace-layout">
      <section className="workspace-main">
        <div className="toolbar">
          <button className="button button-secondary" onClick={onReplace}>
            上传图片
          </button>
          <button className="button button-secondary" onClick={onReset}>
            返回首页
          </button>
          <button
            className={`button ${mode === 'rect' ? 'button-primary' : 'button-secondary'}`}
            onClick={() => setMode('rect')}
            type="button"
          >
            框选模式
          </button>
          <button
            className={`button ${mode === 'brush' ? 'button-primary' : 'button-secondary'}`}
            onClick={() => setMode('brush')}
            type="button"
          >
            画笔模式
          </button>
          <button className="button button-secondary" onClick={() => editorRef.current?.undo()} disabled={!editorState.canUndo}>
            撤销
          </button>
          <button
            className="button button-secondary"
            onClick={() => {
              if (resultBlob) {
                handleRestoreOriginal();
                return;
              }

              editorRef.current?.redo();
            }}
            disabled={resultBlob ? false : !editorState.canRedo}
          >
            重做
          </button>
          <button className="button button-secondary" onClick={() => editorRef.current?.clear()} disabled={!editorState.hasMask}>
            清空
          </button>
          <button
            className="button button-primary"
            onClick={() => void handleProcess()}
            disabled={phase === 'processing' || phase === 'exporting'}
          >
            开始去水印
          </button>
          <button className="button button-primary" onClick={handleExport} disabled={!resultBlob || phase === 'processing'}>
            导出结果
          </button>
        </div>

        <div className="brush-row">
          <span className="brush-label">当前模式：{mode === 'rect' ? '矩形框选' : '手动画笔'}</span>
          <div className="quality-switch">
            <span>修复模式</span>
            <div className="quality-switch-buttons">
              <button
                className={`button ${qualityMode === 'fast' ? 'button-primary' : 'button-secondary'}`}
                onClick={() => setQualityMode('fast')}
                type="button"
              >
                快速模式
              </button>
              <button
                className={`button ${qualityMode === 'hq' ? 'button-primary' : 'button-secondary'}`}
                onClick={() => setQualityMode('hq')}
                type="button"
              >
                高质量模式
              </button>
            </div>
          </div>
          <label className="brush-slider">
            <span>去水印强度</span>
            <input
              type="range"
              min="1"
              max="100"
              step="1"
              value={repairStrength}
              onChange={(event) => setRepairStrength(Number(event.target.value))}
            />
            <strong>{repairStrength}%</strong>
          </label>
          {mode === 'brush' && (
            <label className="brush-slider">
              <span>画笔大小</span>
              <input
                type="range"
                min="8"
                max="160"
                step="1"
                value={brushSize}
                onChange={(event) => setBrushSize(Number(event.target.value))}
              />
              <strong>{Math.round(brushSize)} px</strong>
            </label>
          )}
        </div>

        <section className="editor-panel">
          <AnnotationCanvas
            ref={editorRef}
            source={displaySource}
            width={imageMeta?.width ?? 0}
            height={imageMeta?.height ?? 0}
            mode={mode}
            brushSize={brushSize}
            disabled={phase === 'processing' || phase === 'exporting'}
            renderVersion={renderVersion}
            onStateChange={setEditorState}
            onUserEdit={handleUserEdit}
          />
        </section>
      </section>

      <aside className="sidebar">
        <InstructionCard />
        <StatusCard phase={phase} message={message} progress={progress} />
        <StateLegend currentPhase={phase} />
        <InfoCard title="文件信息">
          <div className="meta-line"><span>文件名</span><strong>{file.name}</strong></div>
          <div className="meta-line"><span>文件大小</span><strong>{formatBytes(file.size)}</strong></div>
          <div className="meta-line"><span>原始分辨率</span><strong>{imageMeta ? `${imageMeta.width} × ${imageMeta.height}` : '加载中'}</strong></div>
          <div className="meta-line"><span>当前模式</span><strong>{qualityMode === 'hq' ? '高质量模式' : '快速模式'}</strong></div>
          <div className="meta-line"><span>本地引擎</span><strong>纯前端修复</strong></div>
        </InfoCard>
        <InfoCard title="处理说明">
          <p className="paragraph">当前免费版只处理你框选或涂抹的局部区域，不会对整张图做高开销运算。</p>
          <p className="paragraph">快速模式先彻底清除局部区域，再做一次局部融合；高质量模式会额外做前景细化、邻域色场重建、纹理回填和边缘修正，复杂区域更自然，但速度更慢。</p>
          <p className="paragraph">纯前端免费方案仍然有边界，但当前版本的目标已经改成“先清干净，再按周围纹理重建并融合”，而不是只做模糊淡化。</p>
        </InfoCard>
      </aside>
    </main>
  );
}

function InstructionCard() {
  return (
    <InfoCard title="操作说明">
      <div className="instruction-list">
        <div className="instruction-item"><span>滚轮</span><strong>缩放</strong></div>
        <div className="instruction-item"><span>右键拖动</span><strong>移动画面</strong></div>
        <div className="instruction-item"><span>左键框选</span><strong>选择水印区域</strong></div>
        <div className="instruction-item"><span>画笔模式</span><strong>手动涂抹水印区域</strong></div>
      </div>
      <p className="paragraph">首页直接上传，进入后就是大画布和核心工具，不额外截图，不做复杂工作台。</p>
      <p className="paragraph">处理完成后当前选区会自动清空。下一次再框选时，会在上一次处理结果的基础上继续修复。</p>
    </InfoCard>
  );
}

function StatusCard({ phase, message, progress }: { phase: WorkflowPhase; message: string; progress: number }) {
  const descriptor = workflowDescriptors.find((item) => item.phase === phase) ?? workflowDescriptors[0];

  return (
    <InfoCard title="当前状态">
      <div className={`status-badge tone-${descriptor.tone}`}>{descriptor.label}</div>
      <p className="paragraph">{message}</p>
      <div className="progress">
        <div className="progress-bar" style={{ width: `${Math.min(100, Math.max(0, progress))}%` }} />
      </div>
    </InfoCard>
  );
}

function StateLegend({ currentPhase }: { currentPhase: WorkflowPhase }) {
  return (
    <InfoCard title="状态说明">
      <div className="state-grid">
        {workflowDescriptors.map((descriptor) => (
          <div
            key={descriptor.phase}
            className={`state-chip ${descriptor.phase === currentPhase ? 'is-active' : ''} tone-${descriptor.tone}`}
          >
            {descriptor.label}
          </div>
        ))}
      </div>
    </InfoCard>
  );
}

function InfoCard({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="info-card">
      <h3>{title}</h3>
      {children}
    </section>
  );
}

export default App;
