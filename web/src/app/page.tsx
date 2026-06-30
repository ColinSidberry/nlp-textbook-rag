import { ProjectLanding } from '@/components/project/ProjectLanding';
import { projectConfig } from '@/components/project/config';
import { NlpRagBody } from '@/components/project/NlpRagBody';

export default function Home() {
  return (
    <ProjectLanding config={projectConfig}>
      <NlpRagBody />
    </ProjectLanding>
  );
}
