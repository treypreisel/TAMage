// TAMage site worker: serves the static landing page and records waitlist
// signups from the npx-button email box into Workers KV (binding: WAITLIST).
// Each email is one KV key -> {at, ua, country}; atomic, no race conditions.

const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]{2,}$/;

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    if (url.pathname === "/api/waitlist" && request.method === "POST") {
      let email = "";
      try {
        email = String((await request.json()).email || "").trim().toLowerCase();
      } catch {
        // fall through to validation failure
      }
      if (!EMAIL_RE.test(email) || email.length > 254) {
        return Response.json({ ok: false, error: "invalid email" }, { status: 400 });
      }
      await env.WAITLIST.put(email, JSON.stringify({
        at: new Date().toISOString(),
        ua: request.headers.get("user-agent") || "",
        country: request.cf?.country || "",
      }));
      return Response.json({ ok: true });
    }

    return env.ASSETS.fetch(request);
  },
};
