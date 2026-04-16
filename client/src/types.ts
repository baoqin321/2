export type MediaKind = 'image' | 'video';

export type WorkflowPhase =
  | 'idle'
  | 'uploaded'
  | 'editing'
  | 'processing'
  | 'completed'
  | 'exporting'
  | 'upload-error'
  | 'process-error'
  | 'unsupported'
  | 'too-large';

export type EditMode = 'rect' | 'brush';

export type ImageRepairQuality = 'fast' | 'hq';

export interface EditorStateSnapshot {
  canUndo: boolean;
  canRedo: boolean;
  hasMask: boolean;
}

export interface WorkflowDescriptor {
  phase: WorkflowPhase;
  label: string;
  tone: 'neutral' | 'active' | 'success' | 'danger';
}
