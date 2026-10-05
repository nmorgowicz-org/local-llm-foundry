// Sanitized innerHTML replacement.
//
// `setHtml(el, html)` is the sanctioned way to render an HTML string into an
// element: the string passes through DOMPurify and the resulting fragment replaces
// the element's children — no `innerHTML =` assignment anywhere. If DOMPurify is
// unavailable the markup is rendered as inert text rather than parsed, so a missing
// sanitizer can never turn into an injection path.
//
// `target="_blank"` is allowed through (release-notes links open in a new tab) but
// every such anchor is forced to `rel="noopener noreferrer"`; any other `target`
// value is dropped.
function hardenLinks(fragment) {
    for (const a of fragment.querySelectorAll('a[target]')) {
        if (a.getAttribute('target') === '_blank') {
            a.setAttribute('rel', 'noopener noreferrer');
        } else {
            a.removeAttribute('target');
        }
    }
    return fragment;
}

export function setHtml(el, html) {
    if (!el) return;
    const purify = window.DOMPurify;
    if (purify && typeof purify.sanitize === 'function') {
        const fragment = purify.sanitize(String(html ?? ''), {
            RETURN_DOM_FRAGMENT: true,
            ADD_ATTR: ['target'],
        });
        el.replaceChildren(hardenLinks(fragment));
        return;
    }
    el.textContent = String(html ?? '');
}
