export const HEVC = "hev1.1.6.L150.B0";  // the device's encoder: Main profile, level 5

// Annex B elementary streams (the raw HEVC camera files) split into access units as bytes arrive

// [slice, first slice of a picture, keyframe, starts a new access unit when it follows a slice]
function nal(d, n) {
  const t = (d[n] >> 1) & 63;
  const slice = t < 32;
  return [slice, slice && (d[n + 2] & 0x80) !== 0, t >= 16 && t <= 21, (t >= 32 && t <= 35) || t === 39];
}

export class AnnexB {
  aus = [];        // [{off, len, key}]
  header = 0;      // bytes before the first slice: the parameter sets a mid-file keyframe needs
  #pos = 0;
  #start = 0;
  #slice = false;
  #key = false;

  // scans d[0, length) from where the last call stopped; done closes the final access unit
  scan(d, length, done) {
    let i = this.#pos;
    for (; i + 5 < length; i++) {
      if (d[i + 2] > 1) { i += 2; continue; }
      if (d[i] || d[i + 1] || d[i + 2] !== 1) continue;
      const n = i + 3, start = i && !d[i - 1] ? i - 1 : i;
      const [slice, first, key, boundary] = nal(d, n);
      if (this.#slice && (first || boundary)) this.#close(start);
      if (slice) {
        if (!this.aus.length && !this.#slice) this.header = start;
        this.#slice = true;
        this.#key ||= key;
      }
      i += 2;
    }
    this.#pos = i;
    if (done && this.#slice) this.#close(length);
  }

  #close(end) {
    this.aus.push({off: this.#start, len: end - this.#start, key: this.#key});
    this.#start = end;
    this.#slice = this.#key = false;
  }
}
