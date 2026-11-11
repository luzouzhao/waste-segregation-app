# Design (Streamlit)

## 1. Theme
Use `.streamlit/config.toml` for a consistent theme instead of inline CSS hacks wherever possible.

```toml
[theme]
base = "light"
primaryColor = "#2E7D32"          # recycling green — primary actions/buttons
backgroundColor = "#FAFAF7"       # off-white background
secondaryBackgroundColor = "#E8F0E3"  # soft sage for cards/sidebar
textColor = "#1B1B1B"
font = "sans serif"
```

## 2. Color Palette
| Purpose | Color | Hex |
|---|---|---|
| Primary (brand/action) | Recycling Green | `#2E7D32` |
| Secondary/accent | Earth Amber | `#C77B23` |
| Background | Off-white | `#FAFAF7` |
| Surface/cards | Soft Sage | `#E8F0E3` |
| Text (primary) | Charcoal | `#1B1B1B` |
| Text (muted) | Grey | `#6B6B6B` |
| Success (recyclable) | `#2E7D32` |
| Warning (hazardous) | `#C62828` |
| Info (needs check) | `#F9A825` |

Category tag colors (used as small badges in the UI):
- Organic → `#6D4C29` (brown)
- Plastic → `#1976D2` (blue)
- Paper → `#8D6E63` (tan)
- Glass → `#00897B` (teal)
- Metal → `#78909C` (steel grey)
- E-waste → `#5E35B1` (purple)
- Hazardous → `#C62828` (red)

## 3. Typography
- Primary font: system sans-serif (Streamlit default) or "Inter"/"Source Sans Pro" if a custom font is added via CSS injection.
- Headings: bold, sentence case, no all-caps.
- Sizes (approx, via `st.markdown` custom CSS if needed): H1 32px, H2 24px, H3 18px, body 16px, caption 13px.
- Keep default Streamlit heading components (`st.title`, `st.header`, `st.subheader`) rather than hand-rolled HTML headings, for accessibility and consistency.

## 4. Layout Principles
- Sidebar: navigation + global controls (city/pincode input for centre lookup, dataset/model info).
- Main area: single clear focal action per page (e.g., upload page = one uploader + one result card).
- Result card pattern: category badge (colored per table above) + confidence bar + recyclability icon + short guidance text, in a `st.container(border=True)`.
- Use `st.columns` for side-by-side comparisons (e.g., ML model metrics table), not custom flexbox CSS.
- Icons: simple line icons for each waste category, stored in `assets/icons/`, referenced via `st.image` next to badges.

## 5. Tone of UI Copy
- Plain, instructional, non-alarming — especially for hazardous waste guidance (informative, not scary).
- Always show a short disclaimer under hazardous/e-waste guidance: "Please confirm with your local municipal disposal rules."
