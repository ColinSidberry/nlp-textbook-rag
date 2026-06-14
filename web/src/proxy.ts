/**
 * Next 16 proxy (formerly middleware) — the RAG site is PUBLIC. Only the
 * "behind-the-scenes" routes require the shared password: the source-code viewer
 * (/code) and the raw database browser (/viewer). Everything else — /, /ask,
 * /login, and the /api/* routes — is open. Auth uses the shared HMAC cookie so
 * one login carries across *.colinsidberry.com.
 */
import { NextResponse, type NextRequest } from "next/server";
import { CODE_COOKIE, verifyToken } from "@/lib/auth";

// Prefixes that require the password. Public otherwise.
const GATED_PREFIXES = ["/viewer", "/code"];

export async function proxy(request: NextRequest) {
  const { pathname } = request.nextUrl;

  const isGated = GATED_PREFIXES.some(
    (p) => pathname === p || pathname.startsWith(`${p}/`)
  );
  if (!isGated) return NextResponse.next();

  const secret = process.env.SITE_SECRET;
  if (!secret) {
    return new NextResponse("Gate not configured", { status: 503 });
  }

  if (await verifyToken(request.cookies.get(CODE_COOKIE)?.value, secret)) {
    return NextResponse.next();
  }

  const url = request.nextUrl.clone();
  url.pathname = "/login";
  url.searchParams.set("from", pathname);
  return NextResponse.redirect(url);
}

export const config = {
  // Only run the gate for the protected routes.
  matcher: ["/viewer", "/viewer/:path*", "/code", "/code/:path*"],
};
