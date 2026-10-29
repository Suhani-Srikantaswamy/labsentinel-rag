"""
Creates a realistic lab safety incident video for testing LabSentinel-RAG.

Scenes (each clearly shows a different type of violation):
  00-12s  SAFE     — Researcher at bench, full PPE, normal work
  12-24s  VIOLATION — No lab coat or gloves while pipetting chemicals
  24-36s  VIOLATION — Fume hood sash fully raised, open solvent flask
  36-48s  VIOLATION — Unlabelled container left open on bench, no goggles
  48-60s  VIOLATION — Centrifuge loaded unbalanced (5 of 6 positions)
  60-72s  VIOLATION — Chemical spill, researcher walks away without cleanup
  72-84s  SAFE     — Correct emergency response: coat on, spill kit used
  84-96s  VIOLATION — Eating/drinking at lab bench

Run:  python scripts/create_sample_video.py
Output: data/videos/lab_incident_sample.mp4
"""

import cv2
import numpy as np
from pathlib import Path

OUT = Path("data/videos/lab_incident_sample.mp4")
OUT.parent.mkdir(parents=True, exist_ok=True)

W, H, FPS = 1280, 720, 15
DURATION  = 96
writer    = cv2.VideoWriter(str(OUT), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))

# ── Colour palette ─────────────────────────────────────────────────────────────
C = dict(
    bg_safe   = (245, 250, 245),
    bg_warn   = (250, 248, 225),
    bg_danger = (250, 230, 230),
    bench     = (190, 170, 145),
    coat      = (210, 215, 235),
    shirt     = (210, 160, 120),
    skin      = (205, 175, 155),
    goggle    = (60,  60,  160),
    glove     = (60,  60,  180),
    flask_g   = (80,  170, 80),
    flask_b   = (80,  130, 200),
    hood_bg   = (200, 200, 220),
    spill     = (40,  170, 230),
    centrifuge= (160, 160, 175),
    label_ok  = (20,  130, 20),
    label_bad = (180, 20,  20),
    white     = (255, 255, 255),
    black     = (10,  10,  10),
    red       = (0,   0,   210),
    yellow    = (0,   210, 230),
    green     = (0,   160, 30),
    orange    = (0,   140, 255),
)


def blank(bg):
    f = np.ones((H, W, 3), np.uint8)
    f[:] = bg
    return f


def bench_rect(f):
    cv2.rectangle(f, (80, 430), (1200, 530), C["bench"], -1)
    cv2.rectangle(f, (80, 430), (1200, 530), (130, 110, 90), 2)


def person(f, x, coat=True, goggles=True, gloves=True):
    """Draw a stick-figure researcher at x-position."""
    # Body
    body_col = C["coat"] if coat else C["shirt"]
    cv2.rectangle(f, (x - 40, 240), (x + 40, 430), body_col, -1)
    # Head
    cv2.circle(f, (x, 215), 38, C["skin"], -1)
    cv2.circle(f, (x, 215), 38, (160, 140, 120), 2)
    # Goggles
    if goggles:
        cv2.ellipse(f, (x - 16, 208), (16, 10), 0, 0, 360, C["goggle"], 3)
        cv2.ellipse(f, (x + 16, 208), (16, 10), 0, 0, 360, C["goggle"], 3)
        cv2.line(f, (x - 32, 208), (x + 32, 208), C["goggle"], 2)
    else:
        # Cross over eyes = no goggles
        cv2.line(f, (x - 30, 200), (x + 30, 215), C["red"], 3)
        cv2.line(f, (x + 30, 200), (x - 30, 215), C["red"], 3)
    # Gloves
    if gloves:
        cv2.ellipse(f, (x - 48, 380), (18, 12), 0, 0, 360, C["glove"], -1)
        cv2.ellipse(f, (x + 48, 380), (18, 12), 0, 0, 360, C["glove"], -1)
    else:
        cv2.ellipse(f, (x - 48, 380), (18, 12), 0, 0, 360, C["skin"], -1)
        cv2.ellipse(f, (x + 48, 380), (18, 12), 0, 0, 360, C["skin"], -1)
        cv2.line(f, (x - 60, 372), (x - 38, 392), C["red"], 2)
        cv2.line(f, (x - 38, 372), (x - 60, 392), C["red"], 2)


def text_box(f, txt, pos, fg, scale=0.7, bg=None):
    x, y = pos
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), bl = cv2.getTextSize(txt, font, scale, 2)
    if bg is not None:
        cv2.rectangle(f, (x - 4, y - th - 4), (x + tw + 4, y + bl + 4), bg, -1)
    cv2.putText(f, txt, (x, y), font, scale, fg, 2, cv2.LINE_AA)


def status_bar(f, violations: list[str], ok_items: list[str]):
    """Bottom status bar."""
    cv2.rectangle(f, (0, H - 80), (W, H), (30, 30, 30), -1)
    for i, ok in enumerate(ok_items):
        text_box(f, f"✓ {ok}", (20 + i * 220, H - 52), C["green"], 0.58)
    for i, viol in enumerate(violations):
        text_box(f, f"✗ {viol}", (20 + i * 320, H - 22), C["red"], 0.60)


def timestamp_overlay(f, ts, scene_name):
    cv2.rectangle(f, (0, 0), (W, 36), (20, 20, 20), -1)
    text_box(f, f"  t = {ts:05.1f}s", (4, 26), (200, 255, 200), 0.70)
    text_box(f, scene_name, (W // 2 - 200, 26), (255, 200, 100), 0.70)
    text_box(f, "LabSentinel-RAG Sample", (W - 320, 26), (160, 160, 160), 0.60)


# ── Scene renderers ────────────────────────────────────────────────────────────

def scene_safe(f, t, fi):
    """Normal safe lab work — full PPE."""
    f[:] = C["bg_safe"]
    bench_rect(f)
    person(f, 500, coat=True, goggles=True, gloves=True)
    # Flask on bench
    cv2.rectangle(f, (650, 360), (700, 430), C["flask_g"], -1)
    cv2.putText(f, "NaCl", (652, 415), cv2.FONT_HERSHEY_SIMPLEX, 0.45, C["white"], 1)
    # Label on flask
    cv2.rectangle(f, (655, 365), (695, 385), C["white"], -1)
    cv2.putText(f, "NaCl", (657, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.38, C["black"], 1)
    # Notebook
    cv2.rectangle(f, (780, 400), (900, 428), (240, 230, 210), -1)
    text_box(f, "Lab Notebook", (785, 423), C["black"], 0.45)
    status_bar(f, [], ["Lab coat", "Goggles", "Gloves", "Labelled containers"])
    text_box(f, "COMPLIANT: All safety rules followed", (340, 590), C["green"], 0.95, bg=C["bg_safe"])


def scene_no_ppe(f, t, fi):
    """Researcher handling chemical with NO lab coat, gloves, or goggles."""
    f[:] = C["bg_warn"]
    bench_rect(f)
    person(f, 500, coat=False, goggles=False, gloves=False)
    # Acid bottle
    cv2.rectangle(f, (630, 340), (680, 430), (80, 80, 200), -1)
    cv2.putText(f, "HCl", (638, 395), cv2.FONT_HERSHEY_SIMPLEX, 0.6, C["white"], 2)
    cv2.putText(f, "Conc.", (632, 418), cv2.FONT_HERSHEY_SIMPLEX, 0.38, C["white"], 1)
    # Pouring action line
    cv2.line(f, (680, 370), (740, 410), (80, 80, 200), 4)
    # Beaker receiving
    cv2.rectangle(f, (740, 390), (800, 430), (170, 210, 250), -1)
    blink = (fi // 6) % 2 == 0
    if blink:
        cv2.rectangle(f, (100, 45), (900, 90), (0, 0, 180), -1)
        text_box(f, "WARNING: No lab coat, goggles or gloves during acid handling!", (110, 80), C["white"], 0.75)
    status_bar(f, ["No lab coat", "No goggles", "No gloves"], ["Chemicals present"])


def scene_fume_hood(f, t, fi):
    """Fume hood sash fully raised during solvent work."""
    f[:] = C["bg_warn"]
    bench_rect(f)
    # Hood cabinet
    cv2.rectangle(f, (250, 120), (850, 430), C["hood_bg"], -1)
    cv2.rectangle(f, (250, 120), (850, 430), (140, 140, 160), 2)
    cv2.putText(f, "FUME HOOD", (440, 175), cv2.FONT_HERSHEY_SIMPLEX, 1.1, C["black"], 2)
    # Sash at TOP (wrong) — should be at the line
    cv2.rectangle(f, (258, 125), (842, 150), (100, 100, 220), -1)
    cv2.putText(f, "SASH POSITION (fully raised — UNSAFE)", (270, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.52, C["white"], 2)
    # Safe height line
    cv2.line(f, (250, 320), (850, 320), C["red"], 3)
    cv2.putText(f, "<-- MAX SAFE SASH HEIGHT", (600, 315), cv2.FONT_HERSHEY_SIMPLEX, 0.5, C["red"], 1)
    # Open solvent flask inside
    cv2.rectangle(f, (480, 330), (530, 425), C["flask_b"], -1)
    cv2.putText(f, "EtOH", (482, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.5, C["white"], 1)
    # Vapour puffs drifting OUT of hood
    for i in range(4):
        cx = 420 + i * 60 + int((t % 2) * 25)
        cv2.circle(f, (cx, 200 - i * 18), 14 + i * 4, (215, 215, 240), -1)
    person(f, 970, coat=False, goggles=False, gloves=False)
    blink = (fi // 6) % 2 == 0
    if blink:
        text_box(f, "DANGER: Vapour escaping from raised sash!", (270, 100), C["red"], 0.75, bg=(255, 220, 220))
    status_bar(f, ["Sash raised", "Vapour escaping", "No PPE"], ["Fume hood in use"])


def scene_unlabelled(f, t, fi):
    """Unlabelled container open on bench, researcher without goggles."""
    f[:] = C["bg_warn"]
    bench_rect(f)
    person(f, 500, coat=True, goggles=False, gloves=True)
    # Unlabelled open bottle
    cv2.rectangle(f, (650, 350), (700, 430), (140, 200, 140), -1)
    # No label (blank)
    cv2.rectangle(f, (655, 360), (695, 395), (200, 200, 200), -1)
    cv2.putText(f, "???", (658, 383), cv2.FONT_HERSHEY_SIMPLEX, 0.6, C["red"], 2)
    # Open lid (no cap)
    cv2.putText(f, "(OPEN)", (642, 348), cv2.FONT_HERSHEY_SIMPLEX, 0.45, C["red"], 1)
    blink = (fi // 7) % 2 == 0
    if blink:
        text_box(f, "VIOLATION: Unlabelled open container — unknown chemical!", (180, 95), C["red"], 0.75, bg=(255, 220, 220))
    status_bar(f, ["No goggles", "Unlabelled container", "Container open/uncapped"], ["Lab coat", "Gloves"])


def scene_centrifuge(f, t, fi):
    """Centrifuge — 5 of 6 positions loaded (unbalanced), no balance tube."""
    f[:] = C["bg_warn"]
    bench_rect(f)
    # Machine body
    cv2.rectangle(f, (400, 260), (760, 430), C["centrifuge"], -1)
    cv2.rectangle(f, (400, 260), (760, 430), (110, 110, 125), 2)
    cv2.putText(f, "CENTRIFUGE", (450, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.85, C["black"], 2)
    # Rotor view (top)
    cv2.ellipse(f, (580, 295), (100, 28), 0, 0, 360, (130, 130, 145), -1)
    # Draw 6 tube slots; 5 filled, slot 5 (300°) empty
    angles = [0, 60, 120, 180, 240, 300]
    for idx, ang in enumerate(angles):
        r = np.radians(ang)
        tx = int(580 + 65 * np.cos(r))
        ty = int(295 + 18 * np.sin(r))
        if idx < 5:
            cv2.circle(f, (tx, ty), 9, (60, 180, 80), -1)  # loaded
        else:
            cv2.circle(f, (tx, ty), 9, C["red"], 2)         # empty — needs balance tube
            cv2.putText(f, "EMPTY", (tx - 22, ty + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, C["red"], 1)
    person(f, 880, coat=True, goggles=True, gloves=False)
    blink = (fi // 6) % 2 == 0
    if blink:
        text_box(f, "VIOLATION: Unbalanced rotor (5/6 loaded, no balance tube)!", (200, 100), C["red"], 0.72, bg=(255, 220, 220))
    status_bar(f, ["No balance tube", "No gloves"], ["Lab coat", "Goggles", "Machine off"])


def scene_spill(f, t, fi):
    """Chemical spill on bench — researcher walks away without response."""
    f[:] = C["bg_danger"]
    bench_rect(f)
    # Knocked over bottle
    angle = 75
    cx, cy = 600, 430
    cv2.ellipse(f, (cx, cy), (25, 10), angle, 0, 360, (80, 160, 80), -1)
    # Spreading spill
    spread = min(int((t % 12) * 18), 180)
    cv2.ellipse(f, (cx + 30, 438), (spread, 22), 0, 0, 360, C["spill"], -1)
    cv2.putText(f, "ACID SPILL", (cx + 20, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.6, C["red"], 2)
    # Spill kit visible but untouched (far side of bench)
    cv2.rectangle(f, (1000, 390), (1100, 430), (220, 50, 50), -1)
    cv2.putText(f, "SPILL KIT", (1004, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.45, C["white"], 1)
    # Researcher walking AWAY (on left side, moving left)
    rx = max(80, 420 - int((t % 12) * 25))
    person(f, rx, coat=True, goggles=False, gloves=False)
    # Arrow pointing away
    cv2.arrowedLine(f, (rx, 320), (rx - 80, 320), C["red"], 3, tipLength=0.3)
    blink = (fi // 5) % 2 == 0
    if blink:
        text_box(f, "VIOLATION: Chemical spill not contained! Researcher not responding!", (140, 100), C["red"], 0.72, bg=(255, 200, 200))
    status_bar(f, ["Spill uncontained", "No PPE", "No spill kit used"], ["Spill kit present"])


def scene_correct_response(f, t, fi):
    """Correct emergency response — PPE on, using spill kit."""
    f[:] = C["bg_safe"]
    bench_rect(f)
    # Small contained spill with absorbent pads
    cv2.ellipse(f, (620, 438), (70, 18), 0, 0, 360, (180, 220, 200), -1)
    cv2.putText(f, "CONTAINED", (565, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.55, C["green"], 2)
    # Absorbent pads on spill
    cv2.rectangle(f, (555, 425), (690, 445), (230, 240, 225), -1)
    cv2.putText(f, "Absorbent pad", (558, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.4, C["black"], 1)
    person(f, 850, coat=True, goggles=True, gloves=True)
    # Spill kit open and in use
    cv2.rectangle(f, (950, 380), (1060, 430), (200, 40, 40), -1)
    cv2.putText(f, "SPILL KIT", (956, 415), cv2.FONT_HERSHEY_SIMPLEX, 0.45, C["white"], 1)
    cv2.putText(f, "(IN USE)", (956, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 200, 0), 1)
    status_bar(f, [], ["Lab coat", "Goggles", "Gloves", "Spill contained", "Spill kit used"])
    text_box(f, "COMPLIANT: Correct spill response procedure followed", (230, 590), C["green"], 0.90, bg=C["bg_safe"])


def scene_eating(f, t, fi):
    """Eating and drinking at the lab bench."""
    f[:] = C["bg_warn"]
    bench_rect(f)
    person(f, 500, coat=True, goggles=False, gloves=False)
    # Coffee cup on bench
    cv2.rectangle(f, (660, 395), (700, 430), (180, 130, 80), -1)
    cv2.ellipse(f, (680, 395), (20, 8), 0, 0, 360, (150, 100, 60), -1)
    cv2.putText(f, "COFFEE", (645, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.4, C["black"], 1)
    # Steam from cup
    for i in range(3):
        cx = 672 + i * 8 + int((t % 2) * 5)
        cv2.line(f, (cx, 390), (cx - 3, 370), (200, 200, 200), 2)
    # Sandwich
    cv2.rectangle(f, (720, 400), (800, 430), (220, 190, 130), -1)
    cv2.putText(f, "SANDWICH", (722, 422), cv2.FONT_HERSHEY_SIMPLEX, 0.38, C["black"], 1)
    # Chemical bottle nearby
    cv2.rectangle(f, (840, 360), (880, 430), (80, 160, 80), -1)
    cv2.putText(f, "H2SO4", (836, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.38, C["white"], 1)
    blink = (fi // 6) % 2 == 0
    if blink:
        cv2.rectangle(f, (100, 45), (970, 90), (0, 0, 180), -1)
        text_box(f, "VIOLATION: Eating and drinking in the laboratory is PROHIBITED!", (110, 80), C["white"], 0.72)
    status_bar(f, ["Eating in lab", "Drinking in lab", "No goggles", "No gloves"], ["Lab coat"])


# ── Scene schedule ──────────────────────────────────────────────────────────────
SCHEDULE = [
    (0,  12, "SAFE — Full PPE, normal work",               scene_safe),
    (12, 24, "VIOLATION — No PPE during acid handling",    scene_no_ppe),
    (24, 36, "VIOLATION — Fume hood sash raised",          scene_fume_hood),
    (36, 48, "VIOLATION — Unlabelled open container",      scene_unlabelled),
    (48, 60, "VIOLATION — Centrifuge imbalance",           scene_centrifuge),
    (60, 72, "VIOLATION — Chemical spill, no response",    scene_spill),
    (72, 84, "SAFE — Correct spill response",              scene_correct_response),
    (84, 96, "VIOLATION — Eating/drinking in laboratory",  scene_eating),
]

print(f"Generating {DURATION}s lab incident sample video …")
total_frames = DURATION * FPS

for fi in range(total_frames):
    t_sec = fi / FPS
    scene_fn   = SCHEDULE[0][3]
    scene_name = SCHEDULE[0][2]
    for start, end, name, fn in SCHEDULE:
        if start <= t_sec < end:
            scene_fn   = fn
            scene_name = name
            break

    frame = blank(C["bg_safe"])
    scene_fn(frame, t_sec, fi)
    timestamp_overlay(frame, t_sec, scene_name)
    writer.write(frame)

    if fi % (FPS * 12) == 0:
        print(f"  {t_sec:4.0f}s  {scene_name}")

writer.release()
size_mb = OUT.stat().st_size / 1024 / 1024
print(f"\n✓ Saved: {OUT}  ({size_mb:.1f} MB)")
print(f"\nThis video contains 6 safety violations across 8 scenes.")
print(f"Upload it with chemistry_lab_manual.txt or biology_lab_manual.txt to test.")
