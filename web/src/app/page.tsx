import { ProjectLanding } from '@/components/project/ProjectLanding';
import { projectConfig } from '@/components/project/config';

export default function Home() {
  return <ProjectLanding config={projectConfig} />;
}
