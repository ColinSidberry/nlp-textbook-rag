/**
 * Shared three-dot loading indicator — the standard section-level loader across
 * all of Colin's project sites. Three brand dots fill in sequence.
 */
export function Dots({ label, className }: { label?: string; className?: string }) {
  return (
    <div className={`flex items-center gap-2 text-muted-foreground ${className ?? ""}`}>
      <span className="flex items-center gap-1.5">
        {[0, 1, 2].map((i) => (
          <span
            key={i}
            className="h-2 w-2 rounded-full bg-brand"
            style={{ animation: "wa-dot 1.05s ease-in-out infinite", animationDelay: `${i * 0.18}s` }}
          />
        ))}
      </span>
      {label && <span className="font-mono text-sm">{label}</span>}
      <style>{`@keyframes wa-dot { 0%, 80%, 100% { opacity: 0.25; transform: scale(0.85); } 40% { opacity: 1; transform: scale(1); } }`}</style>
    </div>
  );
}
