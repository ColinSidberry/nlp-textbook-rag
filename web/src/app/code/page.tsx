import { getCodeData } from '@/lib/code-files';
import { CodeBrowser } from '@/components/code/CodeBrowser';

// Source is read from disk at build time and baked in (see code-files.ts).
export default function CodePage() {
  return <CodeBrowser data={getCodeData()} />;
}
