import { NextResponse } from "next/server";
import { CODE_COOKIE, signToken } from "@/lib/auth";

export const runtime = "nodejs";

function timingSafeEqual(a: string, b: string): boolean {
  if (a.length !== b.length) return false;
  let diff = 0;
  for (let i = 0; i < a.length; i++) diff |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return diff === 0;
}

export async function POST(request: Request) {
  const secret = process.env.SITE_SECRET;
  const expected = process.env.SITE_PASSWORD;
  if (!secret || !expected) {
    return NextResponse.json({ error: "Gate not configured" }, { status: 503 });
  }

  let password = "";
  try {
    const body = await request.json();
    password = typeof body?.password === "string" ? body.password : "";
  } catch {
    return NextResponse.json({ error: "Invalid request" }, { status: 400 });
  }

  if (!timingSafeEqual(password, expected)) {
    return NextResponse.json({ error: "Incorrect password" }, { status: 401 });
  }

  const response = NextResponse.json({ ok: true });
  response.cookies.set(CODE_COOKIE, await signToken(secret), {
    httpOnly: true,
    secure: process.env.NODE_ENV === "production",
    sameSite: "lax",
    path: "/",
    // Set COOKIE_DOMAIN=.colinsidberry.com in production for subdomain SSO; unset
    // locally so it works on localhost.
    domain: process.env.COOKIE_DOMAIN || undefined,
    maxAge: 60 * 60 * 24 * 30, // 30 days
  });
  return response;
}
