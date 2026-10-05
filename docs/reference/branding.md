# Local LLM Foundry brand usage

The source-of-truth mark is [`assets/brand/token-ingot.svg`](../../assets/brand/token-ingot.svg).
All PNG, ICO, ICNS, tray, PWA, package, and social files are generated
derivatives. Do not hand-edit derivatives or create a competing mark.

Use the Token Ingot at its supplied aspect ratio with clear space equal to at
least one quarter of the mark's width. Use the dark/light variants for their
matching surfaces, the monochrome variants for one-color printing, and the
small-size derivatives at 16–32 px. The macOS tray variant remains a template
image so the operating system controls its foreground color.

The menu-bar source is
[`assets/brand/token-ingot-macos-template.svg`](../../assets/brand/token-ingot-macos-template.svg),
a small-size adaptation of the master with transparent layer seams, forge
facets, and an ingot cutout. Keep these gaps: simply recoloring the master
merges its overlapping layers into a solid silhouette. macOS uses the
44×44 raster at a 22-point footprint for Retina detail; Windows and Linux
retain the full-color tray mark. Gradients and cast shadows are not used in
the template because macOS derives its appearance from alpha, not RGB.

Regenerate only the menu-bar derivatives with
`node scripts/generate-brand-derivatives.mjs --tray-only`; the full generator
also preserves this hand-tuned source.

Brand tokens are Foundry teal (`--brand-foundry-teal`), deep teal
(`--brand-foundry-teal-deep`), ingot copper (`--brand-ingot-copper`), and forge
charcoal (`--brand-forge-charcoal`). Status colors remain semantic and must not
be replaced with brand colors. Preserve contrast, reduced-motion behavior, and
light-theme overrides.

When adding a new surface, update the identity registry and use the generated
asset registration. Run the asset validator, release build, and the relevant
screenshot scenario. Do not use the old Llama Monitor mark in current UI,
package metadata, or release assets; historical receipts and changelog entries
must remain truthful and are not bulk-rewritten.
