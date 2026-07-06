"use client";

import { useEffect } from "react";
import { useTheme } from "next-themes";

/**
 * Mirrors the resolved theme into a cookie on `.colinsidberry.com` so the
 * light/dark choice carries across sibling subdomains — the same way the auth
 * cookie does. A tiny seed script in the root layout reads this cookie back
 * before next-themes initializes, so there's no flash on cross-subdomain nav.
 */
export function ThemeSync() {
  const { resolvedTheme } = useTheme();
  useEffect(() => {
    if (resolvedTheme !== "light" && resolvedTheme !== "dark") return;
    const host = window.location.hostname;
    const domain = host.endsWith("colinsidberry.com") ? "; domain=.colinsidberry.com" : "";
    document.cookie = `theme=${resolvedTheme}; path=/; max-age=31536000; samesite=lax${domain}`;
  }, [resolvedTheme]);
  return null;
}
