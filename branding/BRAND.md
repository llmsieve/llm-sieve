# Sieve — Brand

Short reference for using the Sieve mark consistently.

## Colour

| Role           | Value      | Notes                                          |
| -------------- | ---------- | ---------------------------------------------- |
| Brand accent   | `#0D9488`  | The output dot. The only colour in the system. |
| Foreground     | `#0A0A0A`  | Near-black. Wordmark and icon strokes.         |
| Foreground alt | `#FAFAFA`  | Near-white. On dark backgrounds.               |
| Background     | `#FAFAFA`  | Off-white for marketing surfaces.              |
| Muted text     | `#525252`  | Taglines and secondary copy.                   |

The accent colour (`#0D9488`, Tailwind `teal-600`) is reserved. Don't use it for buttons, links, or UI state — it belongs to the mark.

## Typography

- **Wordmark**: Inter, weight 500, letter-spacing -2
- **UI / body**: Inter, weight 400
- **Code / URLs on brand surfaces**: monospace (system stack: `ui-monospace, SFMono-Regular, Menlo, Consolas`)

Sentence case everywhere. No all-caps branding.

## The mark

Two icon variants exist:

| File                        | Use                                        |
| --------------------------- | ------------------------------------------ |
| `sieve-icon.svg`            | Primary — anywhere the icon renders ≥32px  |
| `sieve-icon-favicon.svg`    | Simplified — favicon.ico and ≤16px only    |

Never use the simplified variant above 16px; never use the primary below 32px. They are not interchangeable.

## Clear space

Minimum padding around the lockup equals the height of the icon's funnel (roughly the cap-height of the wordmark). Don't crowd the mark with other UI.

## Minimum sizes

| Surface     | Minimum width          |
| ----------- | ---------------------- |
| Lockup      | 120px                  |
| Icon (primary) | 32px                |
| Icon (favicon) | 16px                |

## On dark vs light backgrounds

Use the `currentColor` master (`sieve-lockup.svg`) wherever CSS can set the colour. For surfaces that can't (PyPI, email, embedded images), use:

- `sieve-lockup-light.svg` on light backgrounds
- `sieve-lockup-dark.svg` on dark backgrounds

The teal output dot remains `#0D9488` in all cases — don't invert it.

## Don'ts

- Don't recolour the wordmark to teal. The dot is the accent; the wordmark is the substrate.
- Don't add a drop shadow, glow, gradient, or outline to any part of the mark.
- Don't rotate, skew, or stretch the lockup. Use the provided horizontal lockup only.
- Don't place the mark on a photo, gradient, or textured background without a solid plate.
- Don't recreate the wordmark in a different typeface.
- Don't use the mark to endorse something Sieve isn't affiliated with.

## File inventory

```
logo/
├── sieve-icon.svg              Primary icon (currentColor + teal dot)
├── sieve-icon-favicon.svg      Simplified mark for ≤16px use only
├── sieve-wordmark.svg          Wordmark only
├── sieve-lockup.svg            Horizontal lockup (currentColor)
├── sieve-lockup-light.svg      For light backgrounds (hardcoded)
├── sieve-lockup-dark.svg       For dark backgrounds (hardcoded)
├── social-card.svg             OG card with corner-dot signature
├── social-card-clean.svg       OG card without corner-dot
├── manifest.webmanifest        PWA manifest
├── README-snippet.md           Drop-in README and <head> code
├── png/                        Raster exports for non-SVG surfaces
└── favicon/                    Site favicon bundle
```

## Licence

The Sieve wordmark, icon, and lockup are trademarks of the project. The source files in this directory are released alongside the Sieve codebase, but the marks themselves may not be used to imply endorsement by the project. See LICENSE for the software licence.
