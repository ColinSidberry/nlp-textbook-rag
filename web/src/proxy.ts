/**
 * Next 16 proxy (formerly middleware) — gates every page and API route behind
 * the shared /code password cookie. The login page and login endpoint are
 * excluded so an unauthenticated visitor can reach the form. Fails closed if
 * the gate isn't configured.
 */
import { NextResponse, type NextRequest } from "next/server";
import { CODE_COOKIE, verifyToken } from "@/lib/auth";

export async function proxy(request: NextRequest) {
  const secret = process.env.SITE_SECRET;
  if (!secret) {
    return new NextResponse("Gate not configured", { status: 503 });
  }

  if (await verifyToken(request.cookies.get(CODE_COOKIE)?.value, secret)) {
    return NextResponse.next();
  }

  const url = request.nextUrl.clone();
  url.pathname = "/login";
  url.searchParams.set("from", request.nextUrl.pathname);
  return NextResponse.redirect(url);
}

export const config = {
  matcher: ["/((?!login|api/login|_next/static|_next/image|favicon.ico).*)"],
};
