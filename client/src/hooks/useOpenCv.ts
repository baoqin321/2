import { useEffect, useState } from 'react';

const OPENCV_SCRIPT_URL = 'https://cdn.jsdelivr.net/npm/@techstark/opencv-js@4.12.0-release.1/dist/opencv.js';
const OPENCV_TIMEOUT_MS = 30000;

declare global {
  interface Window {
    cv?: OpenCvNamespace & { onRuntimeInitialized?: () => void };
    __opencvLoadingPromise?: Promise<OpenCvNamespace>;
  }
}

// OpenCV.js exposes a dynamic runtime namespace; this keeps call sites compatible with its generated API.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type OpenCvNamespace = any;

export function useOpenCv(enabled: boolean) {
  const [cv, setCv] = useState<OpenCvNamespace | null>(null);
  const [status, setStatus] = useState<'idle' | 'loading' | 'ready' | 'error'>(enabled ? 'loading' : 'idle');
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!enabled) {
      return;
    }

    let active = true;
    window.queueMicrotask(() => {
      if (!active) {
        return;
      }

      setStatus('loading');
      setError(null);
    });

    loadOpenCv()
      .then((loadedCv) => {
        if (!active) {
          return;
        }

        setCv(loadedCv);
        setStatus('ready');
      })
      .catch((loadError) => {
        if (!active) {
          return;
        }

        setStatus('error');
        setError(loadError instanceof Error ? loadError.message : 'OpenCV.js 加载失败');
      });

    return () => {
      active = false;
    };
  }, [enabled]);

  return { cv, status, error };
}

export function loadOpenCv() {
  if (typeof window === 'undefined') {
    return Promise.reject(new Error('当前环境不支持加载 OpenCV.js'));
  }

  if (window.cv?.Mat) {
    return Promise.resolve(window.cv);
  }

  if (window.__opencvLoadingPromise) {
    return window.__opencvLoadingPromise;
  }

  const loadingPromise = new Promise<OpenCvNamespace>((resolve, reject) => {
    const resolveWhenReady = () => {
      const loadedCv = window.cv;
      if (!loadedCv) {
        reject(new Error('OpenCV.js 初始化失败'));
        return;
      }

      if (loadedCv.Mat) {
        resolve(loadedCv);
        return;
      }

      const previousInit = loadedCv.onRuntimeInitialized;
      const timeoutId = window.setTimeout(() => {
        reject(new Error('OpenCV.js 初始化超时，请刷新页面后重试'));
      }, OPENCV_TIMEOUT_MS);

      loadedCv.onRuntimeInitialized = () => {
        window.clearTimeout(timeoutId);
        previousInit?.();
        resolve(loadedCv);
      };
    };

    const existingScript = document.querySelector<HTMLScriptElement>('script[data-opencv-loader="true"]');
    if (existingScript) {
      if (existingScript.dataset.loaded === 'true') {
        resolveWhenReady();
        return;
      }

      existingScript.addEventListener('load', resolveWhenReady, { once: true });
      existingScript.addEventListener('error', () => reject(new Error('OpenCV.js 脚本加载失败')), { once: true });
      return;
    }

    const script = document.createElement('script');
    script.async = true;
    script.src = OPENCV_SCRIPT_URL;
    script.dataset.opencvLoader = 'true';
    script.addEventListener('load', () => {
      script.dataset.loaded = 'true';
      resolveWhenReady();
    }, { once: true });
    script.addEventListener('error', () => reject(new Error('OpenCV.js 脚本加载失败')), { once: true });
    document.body.appendChild(script);
  });

  window.__opencvLoadingPromise = loadingPromise.catch((error) => {
    window.__opencvLoadingPromise = undefined;
    throw error;
  });

  return window.__opencvLoadingPromise;
}
