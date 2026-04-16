import {
  forwardRef,
  useEffect,
  useImperativeHandle,
  useRef,
  useState,
} from 'react';

import type { EditMode, EditorStateSnapshot } from '../types';
import { canvasToBlob } from '../utils/files';

type Point = {
  x: number;
  y: number;
};

type Rect = {
  x: number;
  y: number;
  width: number;
  height: number;
};

type InteractionState = {
  panning: boolean;
  brushDrawing: boolean;
  rectDrawing: boolean;
  panStartClientX: number;
  panStartClientY: number;
  panOriginX: number;
  panOriginY: number;
  rectStart?: Point;
  rectCurrent?: Point;
  lastBrushPoint?: Point;
};

const HISTORY_LIMIT = 20;

export interface AnnotationCanvasHandle {
  undo: () => void;
  redo: () => void;
  clear: () => void;
  resetMask: () => void;
  fitToView: () => void;
  hasMask: () => boolean;
  exportMaskBlob: () => Promise<Blob>;
  getMaskCanvas: () => HTMLCanvasElement | null;
}

interface AnnotationCanvasProps {
  source: CanvasImageSource | null;
  width: number;
  height: number;
  mode: EditMode;
  brushSize: number;
  disabled?: boolean;
  renderVersion?: number;
  onStateChange?: (state: EditorStateSnapshot) => void;
  onUserEdit?: () => void;
}

export const AnnotationCanvas = forwardRef<AnnotationCanvasHandle, AnnotationCanvasProps>(function AnnotationCanvas(
  { source, width, height, mode, brushSize, disabled = false, renderVersion = 0, onStateChange, onUserEdit },
  ref,
) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const maskCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const previewCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const mediaRef = useRef<{ source: CanvasImageSource | null; width: number; height: number }>({
    source: null,
    width: 0,
    height: 0,
  });
  const viewportRef = useRef({ scale: 1, offsetX: 0, offsetY: 0 });
  const interactionRef = useRef<InteractionState>({
    panning: false,
    brushDrawing: false,
    rectDrawing: false,
    panStartClientX: 0,
    panStartClientY: 0,
    panOriginX: 0,
    panOriginY: 0,
  });
  const undoStackRef = useRef<ImageData[]>([]);
  const redoStackRef = useRef<ImageData[]>([]);
  const hasMaskRef = useRef(false);
  const resizeFrameRef = useRef<number | null>(null);
  const [zoomPercent, setZoomPercent] = useState(100);

  useEffect(() => {
    mediaRef.current = { source, width, height };
  }, [source, width, height]);

  const emitState = () => {
    onStateChange?.({
      canUndo: undoStackRef.current.length > 0,
      canRedo: redoStackRef.current.length > 0,
      hasMask: hasMaskRef.current,
    });
  };

  const syncPreviewFromBinary = () => {
    const binaryCanvas = maskCanvasRef.current;
    const previewCanvas = previewCanvasRef.current;
    if (!binaryCanvas || !previewCanvas) {
      return;
    }

    const previewContext = previewCanvas.getContext('2d');
    if (!previewContext) {
      return;
    }

    previewContext.clearRect(0, 0, previewCanvas.width, previewCanvas.height);
    previewContext.drawImage(binaryCanvas, 0, 0);
    previewContext.globalCompositeOperation = 'source-in';
    previewContext.fillStyle = 'rgba(32, 112, 255, 0.32)';
    previewContext.fillRect(0, 0, previewCanvas.width, previewCanvas.height);
    previewContext.globalCompositeOperation = 'source-over';
  };

  const detectMaskContent = () => {
    const binaryCanvas = maskCanvasRef.current;
    const context = binaryCanvas?.getContext('2d');
    if (!binaryCanvas || !context) {
      return false;
    }

    const { data } = context.getImageData(0, 0, binaryCanvas.width, binaryCanvas.height);
    for (let index = 3; index < data.length; index += 4) {
      if (data[index] !== 0) {
        return true;
      }
    }

    return false;
  };

  const syncCanvasResolution = () => {
    const container = containerRef.current;
    const canvas = canvasRef.current;
    if (!container || !canvas) {
      return;
    }

    const bounds = container.getBoundingClientRect();
    const devicePixelRatio = window.devicePixelRatio || 1;
    const nextWidth = Math.max(1, Math.floor(bounds.width * devicePixelRatio));
    const nextHeight = Math.max(1, Math.floor(bounds.height * devicePixelRatio));

    if (canvas.width !== nextWidth || canvas.height !== nextHeight) {
      canvas.width = nextWidth;
      canvas.height = nextHeight;
    }
  };

  const fitToView = () => {
    const { width: currentWidth, height: currentHeight } = mediaRef.current;
    const container = containerRef.current;
    if (!container || !currentWidth || !currentHeight) {
      return;
    }

    const bounds = container.getBoundingClientRect();
    const padding = 32;
    const scale = Math.min(
      Math.max((bounds.width - padding * 2) / currentWidth, 0.05),
      Math.max((bounds.height - padding * 2) / currentHeight, 0.05),
    );

    viewportRef.current.scale = Number.isFinite(scale) ? scale : 1;
    viewportRef.current.offsetX = (bounds.width - currentWidth * viewportRef.current.scale) / 2;
    viewportRef.current.offsetY = (bounds.height - currentHeight * viewportRef.current.scale) / 2;
    setZoomPercent(Math.round(viewportRef.current.scale * 100));
  };

  const toImagePoint = (clientX: number, clientY: number): Point | null => {
    const { width: currentWidth, height: currentHeight } = mediaRef.current;
    const canvas = canvasRef.current;
    if (!canvas || !currentWidth || !currentHeight) {
      return null;
    }

    const bounds = canvas.getBoundingClientRect();
    const x = clientX - bounds.left;
    const y = clientY - bounds.top;

    return {
      x: (x - viewportRef.current.offsetX) / viewportRef.current.scale,
      y: (y - viewportRef.current.offsetY) / viewportRef.current.scale,
    };
  };

  const clampPoint = (point: Point) => ({
    x: Math.max(0, Math.min(mediaRef.current.width, point.x)),
    y: Math.max(0, Math.min(mediaRef.current.height, point.y)),
  });

  const pushUndoState = () => {
    const binaryCanvas = maskCanvasRef.current;
    const context = binaryCanvas?.getContext('2d');
    if (!binaryCanvas || !context) {
      return;
    }

    undoStackRef.current.push(context.getImageData(0, 0, binaryCanvas.width, binaryCanvas.height));
    if (undoStackRef.current.length > HISTORY_LIMIT) {
      undoStackRef.current.shift();
    }
    redoStackRef.current = [];
    emitState();
  };

  const drawBrushStroke = (from: Point, to: Point, size: number) => {
    const binaryCanvas = maskCanvasRef.current;
    const previewCanvas = previewCanvasRef.current;
    if (!binaryCanvas || !previewCanvas) {
      return;
    }

    const binaryContext = binaryCanvas.getContext('2d');
    const previewContext = previewCanvas.getContext('2d');
    if (!binaryContext || !previewContext) {
      return;
    }

    binaryContext.save();
    binaryContext.strokeStyle = '#ffffff';
    binaryContext.fillStyle = '#ffffff';
    binaryContext.lineCap = 'round';
    binaryContext.lineJoin = 'round';
    binaryContext.lineWidth = size;
    binaryContext.beginPath();
    binaryContext.moveTo(from.x, from.y);
    binaryContext.lineTo(to.x, to.y);
    binaryContext.stroke();
    binaryContext.beginPath();
    binaryContext.arc(to.x, to.y, size / 2, 0, Math.PI * 2);
    binaryContext.fill();
    binaryContext.restore();

    previewContext.save();
    previewContext.strokeStyle = 'rgba(32, 112, 255, 0.36)';
    previewContext.fillStyle = 'rgba(32, 112, 255, 0.36)';
    previewContext.lineCap = 'round';
    previewContext.lineJoin = 'round';
    previewContext.lineWidth = size;
    previewContext.beginPath();
    previewContext.moveTo(from.x, from.y);
    previewContext.lineTo(to.x, to.y);
    previewContext.stroke();
    previewContext.beginPath();
    previewContext.arc(to.x, to.y, size / 2, 0, Math.PI * 2);
    previewContext.fill();
    previewContext.restore();
  };

  const fillRect = (rect: Rect) => {
    const binaryCanvas = maskCanvasRef.current;
    const previewCanvas = previewCanvasRef.current;
    if (!binaryCanvas || !previewCanvas) {
      return;
    }

    const binaryContext = binaryCanvas.getContext('2d');
    const previewContext = previewCanvas.getContext('2d');
    if (!binaryContext || !previewContext) {
      return;
    }

    binaryContext.fillStyle = '#ffffff';
    binaryContext.fillRect(rect.x, rect.y, rect.width, rect.height);

    previewContext.fillStyle = 'rgba(32, 112, 255, 0.3)';
    previewContext.fillRect(rect.x, rect.y, rect.width, rect.height);
  };

  const renderCanvas = () => {
    const { source: currentSource, width: currentWidth, height: currentHeight } = mediaRef.current;
    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }

    const context = canvas.getContext('2d');
    if (!context) {
      return;
    }

    const bounds = canvas.getBoundingClientRect();
    const devicePixelRatio = window.devicePixelRatio || 1;
    context.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
    context.clearRect(0, 0, bounds.width, bounds.height);

    context.fillStyle = '#f4f7fb';
    context.fillRect(0, 0, bounds.width, bounds.height);

    if (!currentSource || !currentWidth || !currentHeight) {
      return;
    }

    const drawWidth = currentWidth * viewportRef.current.scale;
    const drawHeight = currentHeight * viewportRef.current.scale;

    context.drawImage(currentSource, viewportRef.current.offsetX, viewportRef.current.offsetY, drawWidth, drawHeight);

    const previewCanvas = previewCanvasRef.current;
    if (previewCanvas) {
      context.drawImage(previewCanvas, viewportRef.current.offsetX, viewportRef.current.offsetY, drawWidth, drawHeight);
    }

    context.strokeStyle = 'rgba(15, 23, 42, 0.14)';
    context.lineWidth = 1;
    context.strokeRect(viewportRef.current.offsetX, viewportRef.current.offsetY, drawWidth, drawHeight);

    const activeRect = normalizeRect(interactionRef.current.rectStart, interactionRef.current.rectCurrent);
    if (activeRect) {
      context.save();
      context.setLineDash([8, 6]);
      context.strokeStyle = '#206fff';
      context.fillStyle = 'rgba(32, 111, 255, 0.16)';
      context.lineWidth = 1.5;
      context.fillRect(
        viewportRef.current.offsetX + activeRect.x * viewportRef.current.scale,
        viewportRef.current.offsetY + activeRect.y * viewportRef.current.scale,
        activeRect.width * viewportRef.current.scale,
        activeRect.height * viewportRef.current.scale,
      );
      context.strokeRect(
        viewportRef.current.offsetX + activeRect.x * viewportRef.current.scale,
        viewportRef.current.offsetY + activeRect.y * viewportRef.current.scale,
        activeRect.width * viewportRef.current.scale,
        activeRect.height * viewportRef.current.scale,
      );
      context.restore();
    }
  };

  useImperativeHandle(ref, () => ({
    undo: () => {
      const binaryCanvas = maskCanvasRef.current;
      if (!binaryCanvas || undoStackRef.current.length === 0) {
        return;
      }

      const context = binaryCanvas.getContext('2d');
      if (!context) {
        return;
      }

      redoStackRef.current.push(context.getImageData(0, 0, binaryCanvas.width, binaryCanvas.height));
      const previous = undoStackRef.current.pop();
      if (!previous) {
        return;
      }

      context.putImageData(previous, 0, 0);
      syncPreviewFromBinary();
      hasMaskRef.current = detectMaskContent();
      onUserEdit?.();
      emitState();
      renderCanvas();
    },
    redo: () => {
      const binaryCanvas = maskCanvasRef.current;
      if (!binaryCanvas || redoStackRef.current.length === 0) {
        return;
      }

      const context = binaryCanvas.getContext('2d');
      if (!context) {
        return;
      }

      undoStackRef.current.push(context.getImageData(0, 0, binaryCanvas.width, binaryCanvas.height));
      const next = redoStackRef.current.pop();
      if (!next) {
        return;
      }

      context.putImageData(next, 0, 0);
      syncPreviewFromBinary();
      hasMaskRef.current = detectMaskContent();
      onUserEdit?.();
      emitState();
      renderCanvas();
    },
    clear: () => {
      const binaryCanvas = maskCanvasRef.current;
      const previewCanvas = previewCanvasRef.current;
      if (!binaryCanvas || !previewCanvas || !hasMaskRef.current) {
        return;
      }

      pushUndoState();

      const binaryContext = binaryCanvas.getContext('2d');
      const previewContext = previewCanvas.getContext('2d');
      if (!binaryContext || !previewContext) {
        return;
      }

      binaryContext.clearRect(0, 0, binaryCanvas.width, binaryCanvas.height);
      previewContext.clearRect(0, 0, previewCanvas.width, previewCanvas.height);
      hasMaskRef.current = false;
      onUserEdit?.();
      emitState();
      renderCanvas();
    },
    resetMask: () => {
      const binaryCanvas = maskCanvasRef.current;
      const previewCanvas = previewCanvasRef.current;
      if (!binaryCanvas || !previewCanvas) {
        return;
      }

      const binaryContext = binaryCanvas.getContext('2d');
      const previewContext = previewCanvas.getContext('2d');
      if (!binaryContext || !previewContext) {
        return;
      }

      interactionRef.current.brushDrawing = false;
      interactionRef.current.rectDrawing = false;
      interactionRef.current.lastBrushPoint = undefined;
      interactionRef.current.rectStart = undefined;
      interactionRef.current.rectCurrent = undefined;

      binaryContext.clearRect(0, 0, binaryCanvas.width, binaryCanvas.height);
      previewContext.clearRect(0, 0, previewCanvas.width, previewCanvas.height);
      undoStackRef.current = [];
      redoStackRef.current = [];
      hasMaskRef.current = false;
      emitState();
      renderCanvas();
    },
    fitToView: () => {
      fitToView();
      renderCanvas();
    },
    hasMask: () => hasMaskRef.current,
    exportMaskBlob: async () => {
      const binaryCanvas = maskCanvasRef.current;
      if (!binaryCanvas) {
        throw new Error('没有可用的蒙版');
      }

      return canvasToBlob(binaryCanvas, 'image/png');
    },
    getMaskCanvas: () => maskCanvasRef.current,
  }));

  useEffect(() => {
    if (!width || !height) {
      return;
    }

    maskCanvasRef.current = document.createElement('canvas');
    maskCanvasRef.current.width = width;
    maskCanvasRef.current.height = height;
    previewCanvasRef.current = document.createElement('canvas');
    previewCanvasRef.current.width = width;
    previewCanvasRef.current.height = height;
    undoStackRef.current = [];
    redoStackRef.current = [];
    hasMaskRef.current = false;
    emitState();
    syncCanvasResolution();
    fitToView();
    renderCanvas();
  }, [width, height]);

  useEffect(() => {
    syncCanvasResolution();
    renderCanvas();
  }, [source, renderVersion]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) {
      return;
    }

    const observer = new ResizeObserver(() => {
      if (resizeFrameRef.current !== null) {
        cancelAnimationFrame(resizeFrameRef.current);
      }

      resizeFrameRef.current = window.requestAnimationFrame(() => {
        resizeFrameRef.current = null;
        syncCanvasResolution();
        renderCanvas();
      });
    });

    observer.observe(container);
    return () => {
      observer.disconnect();
      if (resizeFrameRef.current !== null) {
        cancelAnimationFrame(resizeFrameRef.current);
        resizeFrameRef.current = null;
      }
    };
  }, []);

  useEffect(() => {
    const handleMouseMove = (event: MouseEvent) => {
      if (disabled) {
        return;
      }

      const interaction = interactionRef.current;
      if (!interaction.panning && !interaction.brushDrawing && !interaction.rectDrawing) {
        return;
      }

      if (interaction.panning) {
        viewportRef.current.offsetX = interaction.panOriginX + (event.clientX - interaction.panStartClientX);
        viewportRef.current.offsetY = interaction.panOriginY + (event.clientY - interaction.panStartClientY);
        renderCanvas();
        return;
      }

      const point = toImagePoint(event.clientX, event.clientY);
      if (!point) {
        return;
      }

      const clamped = clampPoint(point);

      if (interaction.brushDrawing) {
        const previous = interaction.lastBrushPoint ?? clamped;
        drawBrushStroke(previous, clamped, brushSize);
        interaction.lastBrushPoint = clamped;
        hasMaskRef.current = true;
        emitState();
        renderCanvas();
        return;
      }

      if (interaction.rectDrawing) {
        interaction.rectCurrent = clamped;
        renderCanvas();
      }
    };

    const handleMouseUp = (event: MouseEvent) => {
      const interaction = interactionRef.current;

      if (event.button === 2) {
        interaction.panning = false;
      }

      if (event.button === 0 && interaction.brushDrawing) {
        interaction.brushDrawing = false;
        interaction.lastBrushPoint = undefined;
      }

      if (event.button === 0 && interaction.rectDrawing) {
        interaction.rectDrawing = false;
        const rect = normalizeRect(interaction.rectStart, interaction.rectCurrent);
        interaction.rectStart = undefined;
        interaction.rectCurrent = undefined;

        if (rect && rect.width > 1 && rect.height > 1) {
          pushUndoState();
          fillRect(rect);
          hasMaskRef.current = true;
          onUserEdit?.();
          emitState();
        }

        renderCanvas();
      }
    };

    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);

    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, [brushSize, disabled, onUserEdit]);

  useEffect(() => {
    const container = containerRef.current;
    const canvas = canvasRef.current;
    if (!container || !canvas) {
      return;
    }

    const handleWheel = (event: WheelEvent) => {
      if (disabled || !mediaRef.current.width || !mediaRef.current.height) {
        return;
      }

      event.preventDefault();
      event.stopPropagation();

      const point = toImagePoint(event.clientX, event.clientY);
      const bounds = canvas.getBoundingClientRect();
      if (!point) {
        return;
      }

      const scaleFactor = event.deltaY < 0 ? 1.1 : 0.92;
      const nextScale = Math.min(24, Math.max(0.05, viewportRef.current.scale * scaleFactor));
      viewportRef.current.offsetX = event.clientX - bounds.left - point.x * nextScale;
      viewportRef.current.offsetY = event.clientY - bounds.top - point.y * nextScale;
      viewportRef.current.scale = nextScale;
      setZoomPercent(Math.round(nextScale * 100));
      renderCanvas();
    };

    container.addEventListener('wheel', handleWheel, { passive: false, capture: true });

    return () => {
      container.removeEventListener('wheel', handleWheel, { capture: true });
    };
  }, [disabled]);

  const cursor =
    disabled
      ? 'not-allowed'
      : mode === 'brush'
        ? 'crosshair'
        : 'default';

  return (
    <div className="editor-canvas-shell">
      <div
        ref={containerRef}
        className="editor-canvas-container"
        onContextMenu={(event) => event.preventDefault()}
      >
        <canvas
          ref={canvasRef}
          className="editor-canvas-surface"
          style={{ cursor }}
          onMouseDown={(event) => {
            if (disabled) {
              return;
            }

            if (event.button === 2) {
              interactionRef.current.panning = true;
              interactionRef.current.panStartClientX = event.clientX;
              interactionRef.current.panStartClientY = event.clientY;
              interactionRef.current.panOriginX = viewportRef.current.offsetX;
              interactionRef.current.panOriginY = viewportRef.current.offsetY;
              return;
            }

            if (event.button !== 0) {
              return;
            }

            const point = toImagePoint(event.clientX, event.clientY);
            if (!point || point.x < 0 || point.y < 0 || point.x > mediaRef.current.width || point.y > mediaRef.current.height) {
              return;
            }

            const clamped = clampPoint(point);

            if (mode === 'brush') {
              pushUndoState();
              interactionRef.current.brushDrawing = true;
              interactionRef.current.lastBrushPoint = clamped;
              drawBrushStroke(clamped, clamped, brushSize);
              hasMaskRef.current = true;
              onUserEdit?.();
              emitState();
              renderCanvas();
              return;
            }

            interactionRef.current.rectDrawing = true;
            interactionRef.current.rectStart = clamped;
            interactionRef.current.rectCurrent = clamped;
            renderCanvas();
          }}
        />
        {!source && <div className="editor-empty-state">上传图片后会在这里显示编辑画布</div>}
      </div>
      <div className="editor-canvas-hud">
        <span>缩放 {zoomPercent}%</span>
        <span>{mode === 'rect' ? '框选模式' : `画笔模式 · ${Math.round(brushSize)} px`}</span>
      </div>
    </div>
  );
});

function normalizeRect(start?: Point, current?: Point): Rect | null {
  if (!start || !current) {
    return null;
  }

  const x = Math.min(start.x, current.x);
  const y = Math.min(start.y, current.y);
  const width = Math.abs(current.x - start.x);
  const height = Math.abs(current.y - start.y);

  return { x, y, width, height };
}
