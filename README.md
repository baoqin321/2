# baoqin去水印

当前仓库已经调整为适合 `GitHub Pages + 自定义域名` 的免费静态版本，主流程只保留图片去水印。

## 当前版本说明

- 只支持图片去水印
- 去水印在浏览器本地执行，不依赖后端接口
- 支持点击上传和拖拽上传
- 支持滚轮缩放、右键拖动、框选模式、画笔模式
- 支持撤销、重做、清空、再次局部处理
- 处理后当前选区会自动清空，上一轮处理结果会保留

## 本地开发

```bash
npm install
npm --workspace client run dev -- --host 0.0.0.0
```

## 本地构建

```bash
npm --workspace client run build
```

## 免费部署到 GitHub Pages

仓库里已经包含：

- GitHub Actions 工作流：`.github/workflows/deploy-pages.yml`
- 自定义域名文件：`client/public/CNAME`

你还需要在 GitHub 仓库页面做一件事：

1. 进入仓库 `Settings`
2. 打开 `Pages`
3. 把 `Source` 改成 `GitHub Actions`

之后每次推送到 `main`，GitHub 都会自动构建并发布 `client/dist`。

## 域名配置

当前自定义域名已经写入：

```text
qushuiyin.baoqin.xyz
```

DNS 需要保持：

- `qushuiyin.baoqin.xyz` 指向 `baoqin321.github.io`

## 技术说明

- 去水印算法：浏览器端 OpenCV.js `inpaint`
- 快速模式：局部快速修复
- 高质量模式：局部二次修复和边缘融合
- 导出策略：
  - PNG：按原始分辨率导出
  - JPEG / WEBP：按原始分辨率高质量导出
  - BMP：浏览器端改为 PNG 导出

## 保留内容

仓库中的 `server/` 仍然保留，方便以后恢复付费部署或继续扩展视频去水印，但当前 GitHub Pages 免费版不会使用这些后端代码。
