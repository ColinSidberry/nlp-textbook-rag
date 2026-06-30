'use client';

import { useState } from 'react';

/** Brand logo (simpleicons CDN by slug, or an explicit src), with a text-chip fallback.
 *  `wide` keeps the natural aspect ratio for wordmarks (e.g. Groq) instead of a square. */
export function Logo({
  name,
  slug,
  src,
  wide,
  mono,
}: {
  name: string;
  slug?: string;
  src?: string;
  wide?: boolean;
  mono?: boolean;
}) {
  const [err, setErr] = useState(false);
  const url = src ?? (slug ? `https://cdn.simpleicons.org/${slug}/64748b` : undefined);
  return (
    <div className="flex w-20 flex-col items-center gap-1.5">
      <div className="flex h-9 items-center justify-center">
        {url && mono ? (
          // Render an arbitrarily-colored logo as a flat slate silhouette (the same
          // #64748b as the simpleicons marks) via a CSS mask, so it matches the others
          // and stays legible in both light and dark mode.
          <span
            role="img"
            aria-label={name}
            className={`bg-[#64748b] ${wide ? 'h-6 w-[4.75rem]' : 'h-7 w-7'}`}
            style={{
              maskImage: `url(${url})`, WebkitMaskImage: `url(${url})`,
              maskRepeat: 'no-repeat', WebkitMaskRepeat: 'no-repeat',
              maskPosition: 'center', WebkitMaskPosition: 'center',
              maskSize: 'contain', WebkitMaskSize: 'contain',
            }}
          />
        ) : url && !err ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={url}
            alt={name}
            onError={() => setErr(true)}
            className={`object-contain ${wide ? 'h-6 w-auto max-w-[4.5rem]' : 'h-7 w-7'}`}
          />
        ) : (
          <span className="flex h-7 w-7 items-center justify-center rounded-md bg-muted font-mono text-xs text-muted-foreground">
            {name.slice(0, 2)}
          </span>
        )}
      </div>
      <span className="text-center text-xs text-muted-foreground">{name}</span>
    </div>
  );
}
