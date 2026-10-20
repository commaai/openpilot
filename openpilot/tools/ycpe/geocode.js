// reverse geocoding for drive cards: "Little Italy" / "San Diego, CA", cached by the browser
export const MAPBOX_TOKEN = "pk.eyJ1IjoiY29tbWFhaSIsImEiOiJjangyYXV0c20wMGU2NDluMWR4amUydGl5In0.6Vb11S6tdX6Arpj6trRE_g";
export const MAPBOX_STYLE = "mapbox://styles/commaai/cjj4yzqk201c52ss60ebmow0w";

const cache = new Map();
const MAX_IN_FLIGHT = 4;
let inFlight = 0;
const waiting = [];

// at most a few lookups at a time, so a long drive list doesn't flood mapbox
async function limited(fn) {
  if (inFlight >= MAX_IN_FLIGHT) await new Promise((resolve) => waiting.push(resolve));
  inFlight++;
  try {
    return await fn();
  } finally {
    inFlight--;
    waiting.shift()?.();
  }
}

// cached per ~100 m, for the session and by the browser
export async function reverseLookup(lng, lat) {
  if (!lng && !lat) return null;
  const key = `${lng.toFixed(3)},${lat.toFixed(3)}`;
  if (!cache.has(key)) {
    cache.set(key, limited(async () => {
      const url = `https://api.mapbox.com/geocoding/v5/mapbox.places/${lng},${lat}.json?` + new URLSearchParams({access_token: MAPBOX_TOKEN, limit: 1});
      const resp = await fetch(url, {cache: "force-cache"}).catch(() => null);
      if (!resp || !resp.ok) return null;
      const {features} = await resp.json();
      const f = features[0];
      if (!f || !f.context) return null;
      const ctx = Object.fromEntries(f.context.map((c) => [c.id.split(".")[0], c]));
      const place = (ctx.neighborhood || ctx.locality || ctx.place || {}).text || "";
      const region = ctx.region ? (ctx.region.short_code || "").replace(/^[A-Z]{2}-/, "") || ctx.region.text : "";
      const city = ctx.place && ctx.place.text !== place ? ctx.place.text : ctx.district ? ctx.district.text : "";
      const details = [city, region].filter(Boolean).join(", ");
      return {place, details};
    }));
  }
  return cache.get(key);
}
