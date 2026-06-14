// Build-time source snapshot for the /code viewer.
//
// Runs as a prebuild step (plain Node — NOT bundled by Turbopack, which otherwise
// tries to treat fs.readdirSync paths as asset references and chokes on backend/venv).
// Walks a curated set of source dirs and writes src/generated/code-snapshot.json.

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));
const WEB_ROOT = path.resolve(SCRIPT_DIR, '..');
const REPO_ROOT = path.resolve(WEB_ROOT, '..');

const ROOTS = [
  // The live Next.js app (deploy root) ...
  { label: 'web/src', dir: path.join(WEB_ROOT, 'src') },
  { label: 'web/scripts', dir: path.join(WEB_ROOT, 'scripts') },
  { label: 'web/supabase', dir: path.join(WEB_ROOT, 'supabase') },
  // ... and the original Python RAG pipeline the corpus was built with.
  { label: 'src', dir: path.join(REPO_ROOT, 'src') },
  { label: 'scripts', dir: path.join(REPO_ROOT, 'scripts') },
];

const EXCLUDE_DIRS = new Set(['__pycache__', 'node_modules', '.next', 'venv', '.git', 'experimental', 'generated', '.serena', 'scrap-docs']);
// Skip the Python test suite — the viewer showcases the pipeline, not its tests.
const EXCLUDE_FILE = (name) => /^test_/.test(name) || name.endsWith('.test.ts');
const EXT_LANG = new Map([
  ['.ts', 'typescript'], ['.tsx', 'tsx'], ['.js', 'javascript'], ['.jsx', 'jsx'],
  ['.py', 'python'], ['.json', 'json'], ['.css', 'css'], ['.md', 'markdown'],
  ['.yml', 'yaml'], ['.yaml', 'yaml'], ['.sh', 'bash'], ['.sql', 'sql'],
]);
const MAX_BYTES = 80_000;

function walk(dir, label, files) {
  let entries;
  try {
    entries = fs.readdirSync(dir, { withFileTypes: true });
  } catch {
    return [];
  }
  const dirs = [];
  const fileNodes = [];
  for (const e of entries) {
    if (e.name.startsWith('.')) continue;
    const abs = path.join(dir, e.name);
    const relPath = `${label}/${e.name}`;
    if (e.isDirectory()) {
      if (EXCLUDE_DIRS.has(e.name)) continue;
      const children = walk(abs, relPath, files);
      if (children.length) dirs.push({ type: 'dir', name: e.name, path: relPath, children });
    } else if (e.isFile()) {
      if (EXCLUDE_FILE(e.name)) continue;
      const lang = EXT_LANG.get(path.extname(e.name));
      if (!lang) continue;
      try {
        if (fs.statSync(abs).size > MAX_BYTES) continue;
        files[relPath] = { content: fs.readFileSync(abs, 'utf8'), lang };
        fileNodes.push({ type: 'file', name: e.name, path: relPath, lang });
      } catch {
        /* skip unreadable */
      }
    }
  }
  dirs.sort((a, b) => a.name.localeCompare(b.name));
  fileNodes.sort((a, b) => a.name.localeCompare(b.name));
  return [...dirs, ...fileNodes];
}

const tree = [];
const files = {};
for (const root of ROOTS) {
  const children = walk(root.dir, root.label, files);
  if (children.length) tree.push({ type: 'dir', name: root.label, path: root.label, children });
}

const outDir = path.join(WEB_ROOT, 'src', 'generated');
fs.mkdirSync(outDir, { recursive: true });
fs.writeFileSync(path.join(outDir, 'code-snapshot.json'), JSON.stringify({ tree, files }));
console.log(`code-snapshot: ${Object.keys(files).length} files across ${tree.length} roots`);
