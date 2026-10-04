// Sanitized innerHTML replacement.
//
// `setHtml(el, html)` is the one sanctioned way to render an HTML string into an
// element: the string passes through DOMPurify and the resulting fragment replaces
// the element's children — no `innerHTML =` assignment anywhere. If DOMPurify is
// unavailable (should not happen; dompurify-init loads first) the fragment is built
// from the raw string, matching the pre-helper behavior.
export function setHtml(el, html) {
    if (!el) return;
    if (typeof window.DOMPurify !== 'undefined') {
        el.replaceChildren(window.DOMPurify.sanitize(html, { RETURN_DOM_FRAGMENT: true }));
        return;
    }
    const doc = new DOMParser().parseFromString(html, 'text/html');
    el.replaceChildren(...doc.body.childNodes);
}
