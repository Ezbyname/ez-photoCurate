# E-z Photo Organizer — Future Features

## With Current Libraries (face_recognition, DeepFace, PIL, OpenCV)

- [ ] **Emotion/Expression Detection** — Use DeepFace emotion analysis to auto-tag "best smiles" and filter out photos where people look unhappy
- [ ] **Auto-Group by Scene** — Cluster similar photos using image vectors (same event/location/burst). Let user pick the best from each group
- [ ] **Face Timeline** — Visual timeline of a person growing up, auto-sorted by estimated age. Highlight gaps in the presentation
- [ ] **Photo Quality Scoring** — Rate sharpness (Laplacian variance), brightness, contrast, face centering. Auto-suggest best shot from similar photos
- [ ] **Red-Eye Detection** — Detect red eyes with OpenCV. Flag or auto-fix

## With Small New Libraries

- [ ] **Background Removal** (`rembg`) — Clean cutouts of people for collages or presentation slides
- [ ] **Image Upscaling** (`Real-ESRGAN` / `waifu2x`) — Enhance old/small photos to higher resolution. Especially useful for baby photos
- [ ] **OCR on Photos** (`pytesseract`) — Read text from photos (birthday cards, signs, dates on back of prints). Help with date detection
- [ ] **GPS/Location Map** (`folium` + EXIF GPS) — World map of where photos were taken. Group by location
- [ ] **Auto-Captioning** (`BLIP` / `clip-interrogator`) — AI-generated descriptions for each photo. Useful for search and accessibility

## Presentation-Specific

- [ ] **Slideshow Preview** — Built-in slideshow player with transitions. Preview before exporting
- [ ] **Auto-Layout Collages** — Generate collage pages per age bracket automatically
- [ ] **Music Sync** — Upload a song and auto-time photo transitions to the beat
- [ ] **Before/After Comparison** — Side-by-side of the same person at different ages
