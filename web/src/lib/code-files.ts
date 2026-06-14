/**
 * Types + accessor for the /code viewer's source snapshot.
 *
 * The actual file reading happens in scripts/code-snapshot.mjs (a prebuild step),
 * which writes src/generated/code-snapshot.json. We import that here instead of
 * touching fs at module level — Turbopack tries to bundle fs.readdirSync paths as
 * assets and chokes on backend/venv's symlinks.
 */
import snapshot from '@/generated/code-snapshot.json';

export interface FileNode {
  type: 'file';
  name: string;
  path: string;
  lang: string;
}
export interface DirNode {
  type: 'dir';
  name: string;
  path: string;
  children: TreeNode[];
}
export type TreeNode = FileNode | DirNode;

export interface CodeData {
  tree: DirNode[];
  files: Record<string, { content: string; lang: string }>;
}

export function getCodeData(): CodeData {
  return snapshot as unknown as CodeData;
}
