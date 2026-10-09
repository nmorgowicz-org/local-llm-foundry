// Single source of truth for how an inference backend (loader) + file format is
// presented. Adding a new loader/format (omlx/MLX, strata/GGUF, ninfer/ninfer…)
// is one registry entry; every surface (preset cards, wizard bar, dashboard)
// renders from the same descriptor.
//   engine: display name   format: file format   glyph: 1-char mark
//   hue: 0-360 accent hue (drives --engine-hue)
const ENGINES = {
    llama_cpp: { engine: 'llama.cpp', format: 'GGUF', glyph: 'L', hue: 330 },
    rapid_mlx: { engine: 'Rapid-MLX', format: 'MLX', glyph: 'R', hue: 265 },
    // omlx:    { engine: 'oMLX',      format: 'MLX',  glyph: 'O', hue: 200 },
    // strata:  { engine: 'Strata',    format: 'GGUF', glyph: 'S', hue: 150 },
};

/** Resolve a backend id to a descriptor. Unknown ids get a neutral fallback. */
export function engineDescriptor(backendId) {
    const id = String(backendId || 'llama_cpp');
    const known = ENGINES[id];
    if (known) return { id, ...known };
    const label = id.replace(/[_-]+/g, ' ').trim() || 'unknown';
    return { id, engine: label, format: '', glyph: label.charAt(0).toUpperCase() || '?', hue: null };
}

/** Inline style that exposes the accent hue to CSS (empty for neutral engines). */
export function engineHueStyle(descriptor) {
    return descriptor.hue == null ? '' : `--engine-hue:${Number(descriptor.hue) || 0}`;
}

/** Small "[R] Rapid-MLX · MLX" tag. Values are escaped by the caller-provided fn. */
export function renderEngineTag(descriptor, escape) {
    const text = descriptor.format
        ? `${escape(descriptor.engine)} <span class="engine-tag-sep">·</span> ${escape(descriptor.format)}`
        : escape(descriptor.engine);
    return `<div class="engine-tag" title="${escape(descriptor.engine)}${descriptor.format ? ' · ' + escape(descriptor.format) : ''}">`
        + `<span class="engine-tag-glyph">${escape(descriptor.glyph)}</span>`
        + `<span class="engine-tag-text">${text}</span></div>`;
}
