export const MAX_CHARS = 40_000;
export const MIN_WORDS = 80;
const TARGET_WORDS = 110;
const BOUNDARY_REACH = 10;

export function countWords(text) {
  return (text.match(/[\p{L}\p{M}]+(?:[-’'][\p{L}\p{M}]+)*/gu) ?? []).length;
}

export function partition(text) {
  // Keep offsets into the source so hard wraps, dialogue and scene breaks survive.
  const terms = [...text.matchAll(/\S+/gu)];
  const size = Math.ceil(terms.length / Math.max(1, Math.round(terms.length / TARGET_WORDS)));
  const parts = [];
  let start = 0;
  while (start + size + BOUNDARY_REACH < terms.length) {
    const target = start + size;
    let end = target;
    let best = -Infinity;
    for (let i = target - BOUNDARY_REACH; i <= target + BOUNDARY_REACH; i += 1) {
      const previous = terms[i - 1];
      const gap = text.slice(previous.index + previous[0].length, terms[i].index);
      const hasLineBreak = /[\r\n]/u.test(gap);
      const endsSentence = /[.!?…][»”"')\]]*$/u.test(previous[0]);
      const score = (hasLineBreak ? 40 : endsSentence ? 20 : 0) - Math.abs(i - target);
      if (score > best) {
        best = score;
        end = i;
      }
    }
    parts.push(text.slice(terms[start].index, terms[end].index).trim());
    start = end;
  }
  if (terms.length) parts.push(text.slice(terms[start].index).trim());
  return parts;
}

export function subdivide(text) {
  const midpoint = Math.floor(text.length / 2);
  const left = text.lastIndexOf(" ", midpoint);
  const right = text.indexOf(" ", midpoint);
  const cut = [left, right].filter(i => i > midpoint / 2 && i < midpoint * 1.5)
    .sort((a, b) => Math.abs(a - midpoint) - Math.abs(b - midpoint))[0];
  if (cut !== undefined) return [text.slice(0, cut).trim(), text.slice(cut).trim()];
  const points = Array.from(text);
  const middle = Math.floor(points.length / 2);
  return [points.slice(0, middle).join(""), points.slice(middle).join("")];
}
