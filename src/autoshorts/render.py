"""Composite a built clip (clipbuild.py) into the final vertical (9:16)
shorts look - generalized from an earlier broadcast-specific tool.

Differences from that earlier version:
  - operates on an already-built clip file (clipbuild.build_clip's output),
    not a freshly-downloaded YouTube segment - no pad_sec/offset dance is
    needed since MediaSource.extract_clip already returns exact ranges.
  - caption timing is computed across possibly several concatenated ranges
    (see _captions_for_ranges), not a single [start_sec, end_sec] span.
  - logos and the bottom band image are always worker-uploaded at render
    time (step 8 of the review flow), never bundled defaults - so there is
    no text-fallback for them, only "show the uploaded image, or nothing".
  - split into two stages, matching the review flow: render_draft() (crop +
    captions only, plain center crop for speed) so the worker can proofread
    captions before anything is finalized, and render_final() (adds the
    face-detected crop, title banner, logos, guest label, bottom image).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .analyze import ClipRange
from .transcript import Segment

# same blank-.env-line pitfall as config.py's MODEL - `or`, not the
# two-arg .get form, so a blank AUTOSHORTS_DISPLAY_FONT= line falls back
# to this default instead of resolving to an empty path.
DISPLAY_FONT = os.environ.get("AUTOSHORTS_DISPLAY_FONT") or "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"

CANVAS_W = 1080
CANVAS_H = 1920

# extra zoom-in applied on top of the minimum scale needed to fill the
# canvas (see build_filter_complex's scale stage and find_speaker_crop).
# Without this, a speaker positioned close to either edge of the original
# frame can't always be centered - centering them would require the crop
# window to extend past the edge of the source frame, which isn't possible,
# so the crop clamps to the frame boundary instead and the speaker ends up
# off-center. Scaling the source further before cropping gives every
# position more room (in pixels) between it and the frame edge, so the
# clamp is far less likely to bite. The tradeoff is a bit more digital
# upscaling (softer image) and a good chance of cropping out whatever sits
# in the corners of the original frame - both accepted in exchange for
# reliably keeping the speaker centered.
FACE_CROP_ZOOM = 1.2

# Two-camera (side-by-side) sources have a white seam down the middle where
# the two camera boxes meet. If the crop window reaches left of that seam it
# shows up as a stray white line on the short's left edge. The window is
# therefore kept entirely inside the speaker's half of the frame, which
# needs that half (scaled) to be at least CANVAS_W wide plus this margin -
# two_camera_zoom() raises the zoom as far as that takes. The margin also
# covers the seam's own thickness and compression fuzz around it.
SEAM_MARGIN_PX = 14


def two_camera_zoom(src_w: float, src_h: float, canvas_w: int, canvas_h: int) -> float:
    """Zoom (on top of the minimum fill-the-canvas scale) at which one half
    of a side-by-side source is wide enough to fill the canvas by itself,
    never less than FACE_CROP_ZOOM."""
    base = max(canvas_w / src_w, canvas_h / src_h)
    needed_scale = 2 * (canvas_w + SEAM_MARGIN_PX) / src_w
    return max(FACE_CROP_ZOOM, needed_scale / base)

# YouTube's Shorts shelf/grid thumbnail tiles crop a vertical video down
# from the top instead of just scaling it - text sitting close to the top
# edge of the canvas gets its top sliced off in those tiles even though it
# looks fine in the full player. TITLE_TOP_MARGIN pushes the title text down
# far enough to clear that crop.
TITLE_BAND_H = 440
TITLE_TOP_MARGIN = 70
TITLE_LINE1_Y = 100 + TITLE_TOP_MARGIN
TITLE_LINE2_Y = 226 + TITLE_TOP_MARGIN
TITLE_FONTSIZE = 88
ACCENT_COLOR = "gold"

LOGO_IMAGE_HEIGHT = 210
# real logo aspect ratios vary widely - some supplied logos are very wide,
# short wordmarks (as low as ~1:6.5 height:width), which too narrow a
# width cap crushes down to an illegible sliver. 260 is a middle ground,
# checked by eye against the actual logos in use: legible without
# dominating the frame (380 was tried first and looked oversized).
LOGO_IMAGE_MAX_WIDTH = 260
LOGO_MARGIN_X = 20
LOGO_MARGIN_Y = 6

# the bottom band image - the worker-uploaded one from step 8 of the
# manual review flow, or (완전 자동 모드) the fixed logo from default_assets/,
# sized deliberately larger than the two top-corner logos since it's
# meant to stand out as the single most prominent brand mark on the clip,
# not a small badge - the bottommost item stacked in the bottom info band,
# flush against the very bottom edge of the canvas.
BOTTOM_IMAGE_BAND_H = 260
BOTTOM_IMAGE_HEIGHT = 220
BOTTOM_IMAGE_MAX_WIDTH = 550
BOTTOM_IMAGE_BAND_OPACITY = 0.75
BOTTOM_IMAGE_BAND_TOP = CANVAS_H - BOTTOM_IMAGE_BAND_H

# captions sit directly above the bottom image band, still inside the same
# solid-black bottom info band (not a separate overlay floating over video).
CAPTION_FONTSIZE = 62
CAPTION_BAND_H = 200
CAPTION_BAND_GAP_ABOVE_BOTTOM_IMAGE = 40
CAPTION_BAND_TOP = BOTTOM_IMAGE_BAND_TOP - CAPTION_BAND_GAP_ABOVE_BOTTOM_IMAGE - CAPTION_BAND_H
CAPTION_TEXT_PADDING_TOP = 25
CAPTION_LINE_HEIGHT = round(CAPTION_FONTSIZE * 1.25)
CAPTION_MAX_CHARS_PER_LINE = 14
# hard ceiling for one caption line. Hangul glyphs are ~CAPTION_FONTSIZE px
# wide, so 16 of them (~1000px) still fit inside the CANVAS_W canvas; a longer
# line would run off the screen edges.
CAPTION_HARD_MAX_CHARS_PER_LINE = 16
CAPTION_MAX_LINES = 2
CAPTION_BOX_OPACITY = 0.75

# optional name plate ("이름 직책 / 소속") identifying the on-screen guest,
# shown for the whole clip at the top of the bottom info band, directly
# above the caption.
GUEST_LABEL_FONTSIZE = 38
GUEST_LABEL_BAND_H = 66
GUEST_LABEL_GAP_ABOVE_CAPTION = 8
GUEST_LABEL_BAND_TOP = CAPTION_BAND_TOP - GUEST_LABEL_GAP_ABOVE_CAPTION - GUEST_LABEL_BAND_H
GUEST_LABEL_BOX_OPACITY = 0.75

# the bottom info band - guest name plate + caption + bottom image stacked
# together - is padded into the canvas the same way the top title band is
# (see the `pad` stage in build_filter_complex), instead of floating as a
# translucent overlay on top of the video: that would leave slivers of raw
# video peeking through above/below/between the three elements.
# GUEST_LABEL_BAND_TOP is the topmost of the three stacked elements, so it
# doubles as this band's top edge.
BOTTOM_BAND_H = CANVAS_H - GUEST_LABEL_BAND_TOP


def format_shoot_date(date_str: str | None) -> str:
    """"20261127" or "2026-11-27" -> "11월 27일" ("" if unparseable/empty)."""
    digits = re.sub(r"\D", "", date_str or "")
    if len(digits) != 8:
        return ""
    month, day = int(digits[4:6]), int(digits[6:8])
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return ""
    return f"{month}월 {day}일"


_DESC_DATE_RE = re.compile(
    r"촬영\s*(?:날짜|일자|일)?\s*[:：]?\s*(?:\d{4}\s*[년.\-/]\s*)?"
    r"(\d{1,2})\s*(?:월|[./-])\s*(\d{1,2})\s*일?"
)


def extract_shoot_date_from_description(description: str | None) -> str:
    """Find "촬영 날짜: 10월 27일" (a convention some videos use in their
    description) and return it as "10월 27일"; "" if there isn't one."""
    m = _DESC_DATE_RE.search(description or "")
    if not m:
        return ""
    month, day = int(m.group(1)), int(m.group(2))
    return f"{month}월 {day}일" if 1 <= month <= 12 and 1 <= day <= 31 else ""


def with_shoot_date(guest_label: str, shoot_date: str, kind: str = "촬영") -> str:
    """Append "[촬영 : 11월 27일]" (or "[방송 : ...]") to the guest label line."""
    if not shoot_date:
        return guest_label
    tag = f"[{kind} : {shoot_date}]"
    return f"{guest_label} {tag}" if guest_label else tag


@dataclass
class RenderAssets:
    thumbnail_text: str = ""
    guest_label: str = ""
    shoot_date: str = ""  # e.g. "11월 27일"; shown after the guest label
    date_kind: str = "촬영"  # prefix shown before the date, e.g. "촬영"
    logo_top_left: Path | None = None
    logo_top_right: Path | None = None
    banner_bottom: Path | None = None


def escape_drawtext(text: str, keep_newlines: bool = False) -> str:
    """Escape a string for safe use inside an ffmpeg drawtext filter argument.

    Every drawtext call is built with expansion=none (see build_filter_complex),
    which turns off drawtext's own %{...} expansion mini-language entirely -
    that's what a literal '%' in real captions needs to survive intact;
    backslash-escaping it (the naive fix) does not work, since expansion
    runs as a separate parsing pass after this value is already extracted.
    """
    text = text.replace("\\", "\\\\")
    text = text.replace(":", "\\:")
    text = text.replace("'", "’")  # avoid unbalanced quotes inside the filter
    if not keep_newlines:
        text = text.replace("\n", " ")
    return text


def _title_lines(thumbnail_text: str) -> tuple[str, str | None]:
    """Split thumbnail_text into (line1, line2). line2 is None if not given."""
    parts = thumbnail_text.split("\n", 1)
    line1 = parts[0].strip()
    line2 = parts[1].strip() if len(parts) > 1 else None
    return line1, (line2 or None)


def _greedy_lines(text: str, max_chars: int) -> list[str]:
    """Word-wrap on spaces so no line exceeds max_chars; a single word longer
    than max_chars is split mid-word rather than left to overflow."""
    lines: list[str] = []
    cur = ""
    for word in text.split():
        while len(word) > max_chars:
            if cur:
                lines.append(cur)
                cur = ""
            lines.append(word[:max_chars])
            word = word[max_chars:]
        if not cur:
            cur = word
        elif len(cur) + 1 + len(word) <= max_chars:
            cur += " " + word
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def _wrap_caption(text: str, max_chars: int = CAPTION_MAX_CHARS_PER_LINE) -> str:
    """Wrap caption text, preferring 2 balanced lines broken near the midpoint
    on a space. No line ever exceeds CAPTION_HARD_MAX_CHARS_PER_LINE; if the
    text can't fit in 2 such lines the result has more than 2 lines (callers
    that draw it should split it first with _caption_chunks)."""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    mid = len(text) // 2
    break_at = None
    for offset in range(0, mid + 1):
        for candidate in (mid - offset, mid + offset):
            if 0 < candidate < len(text) and text[candidate] == " ":
                break_at = candidate
                break
        if break_at is not None:
            break
    if break_at is not None:
        first, second = text[:break_at], text[break_at + 1:]
        if max(len(first), len(second)) <= CAPTION_HARD_MAX_CHARS_PER_LINE:
            return first + "\n" + second
    return "\n".join(_greedy_lines(text, CAPTION_HARD_MAX_CHARS_PER_LINE))


def _caption_chunks(start: float, end: float, text: str) -> list[tuple[float, float, str]]:
    """Split one cue into consecutive cues of at most CAPTION_MAX_LINES lines
    each (the caption band only fits that many), sharing the cue's time span
    in proportion to text length. Short cues come back unchanged."""
    lines = _wrap_caption(text).split("\n")
    if len(lines) <= CAPTION_MAX_LINES:
        return [(start, end, "\n".join(lines))]
    groups = [lines[i:i + CAPTION_MAX_LINES] for i in range(0, len(lines), CAPTION_MAX_LINES)]
    weights = [max(1, sum(len(l) for l in g)) for g in groups]
    total = sum(weights)
    chunks: list[tuple[float, float, str]] = []
    t = start
    for g, w in zip(groups, weights):
        t_end = end if g is groups[-1] else t + (end - start) * w / total
        chunks.append((t, t_end, "\n".join(g)))
        t = t_end
    return chunks


_SPEAKER_MARKER_RE = re.compile(r"\s*>>\s*")


def _strip_speaker_marker(text: str) -> str:
    """Drop the ">>" speaker-change marker analyze.py's guest-centered rule
    relies on in the transcript text - it belongs in the analysis prompt,
    not in what a viewer actually reads on-screen."""
    return _SPEAKER_MARKER_RE.sub(" ", text).strip()


def captions_for_ranges(
    segments: list[Segment], ranges: list[ClipRange]
) -> list[tuple[float, float, str]]:
    """Transcript segments overlapping any of `ranges`, with times mapped
    onto the final concatenated clip's own timeline: each range contributes
    its own overlapping segments (relative to that range's start), offset
    by the total duration of the ranges stitched in before it.

    A segment only counts as "in" a range if at least half of the segment's
    own duration falls inside it, not on any overlap at all. Transcript
    segments routinely overlap their neighbors by a second or two (rolling
    captions, whisper VAD) - ranges themselves come from the worker
    deselecting a segment in the review UI, which splits one selection into
    two ranges with that segment excluded. Without the majority-overlap
    requirement, the excluded segment's own overlap with the *adjacent*
    range's edge (a side effect of the neighbor overlap, not anything the
    worker chose) would still be enough to sneak its caption back in, even
    though its video was correctly cut out - a caption flashing over
    footage that no longer matches it.
    """
    out: list[tuple[float, float, str]] = []
    clip_offset = 0.0
    for r in ranges:
        for seg in segments:
            overlap_start = max(seg.start_sec, r.start_sec)
            overlap_end = min(seg.end_sec, r.end_sec)
            overlap = overlap_end - overlap_start
            if overlap <= 0:
                continue
            seg_duration = seg.end_sec - seg.start_sec
            if seg_duration > 0 and overlap < seg_duration / 2:
                continue
            rel_start = max(0.0, seg.start_sec - r.start_sec)
            rel_end = min(r.end_sec - r.start_sec, seg.end_sec - r.start_sec)
            if rel_end <= rel_start:
                continue
            text = _strip_speaker_marker(seg.text)
            if not text:
                continue
            out.append((clip_offset + rel_start, clip_offset + rel_end, text))
        clip_offset += r.end_sec - r.start_sec
    return _clip_overlapping_cues(out)


def _clip_overlapping_cues(
    cues: list[tuple[float, float, str]]
) -> list[tuple[float, float, str]]:
    """Clip each cue's end to the next cue's start so consecutive captions
    never both satisfy build_filter_complex's enable window on the same
    frame. Source transcripts - YouTube's rolling auto-captions especially -
    commonly have consecutive cues whose time ranges genuinely overlap by a
    second or two (rolling captions are *designed* to overlap: a new line
    appears before the old one finishes scrolling off), and without this
    clip that overlap burns in as the old caption still visible, dissolved
    together with the new one, instead of a clean cut between them."""
    result: list[tuple[float, float, str]] = []
    for i, (start, end, text) in enumerate(cues):
        if i + 1 < len(cues):
            next_start = cues[i + 1][0]
            if next_start < end:
                end = next_start
        if end > start:
            result.append((start, end, text))
    return result


def _face_crop_offset(
    faces: list[tuple[float, float, float, float]],
    scale: float,
    canvas_w: int,
    canvas_h: int,
    scaled_w: float,
    scaled_h: float,
    src_w: float,
    single_camera: bool = False,
) -> tuple[int, int] | None:
    """Given detected face boxes (x, y, w, h) in original-frame pixel coords,
    return the (x, y) crop offset - in scaled-frame pixel coords - that keeps
    the most prominent face(s) inside a canvas_w x canvas_h crop window,
    biased to leave headroom above the face rather than dead-centering it.
    None if no faces were given.

    A static multi-camera composite (e.g. host and guest, each in their own
    camera box side by side) has a face on each side of every sampled frame.
    Averaging all of them just re-centers the crop on the seam between the
    two cameras - exactly where you don't want it. Instead, cluster faces by
    which half of the source frame they're in (split at src_w/2) and go with
    whichever side has faces at all, preferring the right if both do (most
    interview-style formats seat the person actually being quoted on that
    side - see analyze.py's _GUEST_CENTERED_RULE, which this crop bias is
    built to match).

    single_camera skips that left/right clustering entirely: a true
    single-camera shot (one continuous lens) has no seam to
    avoid, so every detected face is real content in the same frame and a
    plain weighted average across all of them is the right default (unlike
    the two-camera composite case, blending both sides here would wrongly
    bias toward whichever studio camera convention a given source
    doesn't actually use).
    """
    if not faces:
        return None
    on_right = False
    if single_camera:
        dominant = faces
    else:
        midpoint = src_w / 2
        left = [f for f in faces if f[0] + f[2] / 2 < midpoint]
        right = [f for f in faces if f[0] + f[2] / 2 >= midpoint]
        dominant = right if right else left
        on_right = bool(right)

    total_area = 0.0
    weighted_x = 0.0
    weighted_y = 0.0
    for x, y, w, h in dominant:
        area = w * h
        weighted_x += (x + w / 2) * area
        weighted_y += (y + h * 0.35) * area  # roughly eye-level, not box center
        total_area += area
    face_x = (weighted_x / total_area) * scale
    face_y = (weighted_y / total_area) * scale

    max_x_off = max(0.0, scaled_w - canvas_w)
    max_y_off = max(0.0, scaled_h - canvas_h)
    x_off = min(max(0.0, face_x - canvas_w / 2), max_x_off)
    if not single_camera:
        x_off = _keep_off_seam(x_off, on_right, scaled_w, canvas_w, max_x_off)
    y_off = min(max(0.0, face_y - canvas_h / 3), max_y_off)
    return round(x_off), round(y_off)


def _keep_off_seam(
    x_off: float, on_right: bool, scaled_w: float, canvas_w: int, max_x_off: float,
) -> float:
    """Slide the crop window so it stays inside the speaker's half of a
    two-camera frame (the seam sits at scaled_w / 2) - see SEAM_MARGIN_PX."""
    seam = scaled_w / 2
    if on_right:
        return min(max(x_off, seam + SEAM_MARGIN_PX), max_x_off)
    return max(min(x_off, seam - SEAM_MARGIN_PX - canvas_w), 0.0)


# tried in order; side-lit/angled shots are often missed by the default
# frontal cascade alone
_FACE_CASCADES = ("haarcascade_frontalface_default.xml", "haarcascade_frontalface_alt2.xml")
# the profile cascade only catches faces looking roughly one direction (its
# training set is mirrored-left) - real footage has the speaker facing
# either way depending on which side of frame they're on, so it's run
# again on a horizontally flipped copy in _detect_faces to catch both.
_PROFILE_CASCADE = "haarcascade_profileface.xml"


def _run_cascade(cascade: object, gray, minNeighbors: int = 3) -> list[tuple[float, float, float, float]]:
    found = cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=minNeighbors, minSize=(30, 30))
    return [(float(x), float(y), float(w), float(h)) for x, y, w, h in found]


def _detect_faces(image_path: Path) -> list[tuple[float, float, float, float]]:
    """Detect faces in a single image, returning (x, y, w, h) boxes in pixel
    coords. Empty on any failure (no opencv installed, a build/version that
    dropped CascadeClassifier, unreadable image, etc.) - callers treat that
    the same as "no faces found".

    Haar cascades are a weak, dated detector - frontal-only and easily
    thrown off by a slight turn of the head, which is common in interview
    footage where the speaker favors one camera. Frontal + profile (run
    both normal and mirrored, since the profile cascade only catches one
    facing direction) covers meaningfully more real-world angles than the
    frontal cascades alone did.
    """
    try:
        import cv2
    except ImportError as e:
        print(f"[render] opencv를 불러올 수 없어 얼굴 인식을 건너뜁니다: {e}")
        return []
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return []
        gray = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        img_w = gray.shape[1]
        boxes: list[tuple[float, float, float, float]] = []

        for cascade_name in _FACE_CASCADES:
            cascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascade_name)
            boxes.extend(_run_cascade(cascade, gray))

        profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + _PROFILE_CASCADE)
        boxes.extend(_run_cascade(profile_cascade, gray))
        flipped = cv2.flip(gray, 1)
        for x, y, w, h in _run_cascade(profile_cascade, flipped):
            boxes.append((img_w - x - w, y, w, h))

        return boxes
    except (AttributeError, cv2.error) as e:
        print(f"[render] 얼굴 인식 실패, 가운데 크롭을 사용합니다: {e}")
        return []


def _probe_dimensions(video_path: Path) -> tuple[int, int] | None:
    try:
        out = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=width,height", "-of", "csv=p=0",
                str(video_path),
            ],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        w, h = (int(v) for v in out.split(","))
        return w, h
    except (subprocess.CalledProcessError, ValueError, FileNotFoundError):
        return None


_CENTER_CROP = ("(in_w-out_w)/2", "(in_h-out_h)/2")

# horizontal target (as a fraction of source width), used only as a last
# resort when face detection finds nothing at all to work with. Most
# interview-style formats seat the person being quoted on the right side of
# a side-by-side two-camera shot (matching analyze.py's _GUEST_CENTERED_RULE
# bias), so aiming the crop window there is a better blind guess than a
# literal center crop - a plain center crop on a two-person side-by-side
# shot reliably cuts both people's faces in half instead of keeping either
# one whole.
_RIGHT_BIAS_FRACTION = 0.72


def _blind_right_biased_offset(
    scale: float, canvas_w: int, canvas_h: int, scaled_w: float, scaled_h: float, src_w: float,
) -> tuple[int, int]:
    max_x_off = max(0.0, scaled_w - canvas_w)
    max_y_off = max(0.0, scaled_h - canvas_h)
    target_x = src_w * _RIGHT_BIAS_FRACTION * scale
    x_off = min(max(0.0, target_x - canvas_w / 2), max_x_off)
    x_off = _keep_off_seam(x_off, True, scaled_w, canvas_w, max_x_off)
    y_off = max_y_off / 2  # no vertical signal available - just center it
    return round(x_off), round(y_off)


def find_speaker_crop(
    clip_path: Path, canvas_w: int, canvas_h: int, single_camera: bool = False,
) -> tuple[str, str]:
    """Best-effort: sample a few frames from clip_path, detect faces, and
    return (crop_x, crop_y) ffmpeg crop-filter position values that keep the
    most prominent speaker in frame instead of a blind center crop. Falls
    back to a right-biased crop (see _blind_right_biased_offset) when
    detection runs but finds no faces at all, or to ffmpeg's own centered
    crop when detection can't run in the first place (opencv not installed,
    ffprobe/ffmpeg failure, etc.) - never raises.

    single_camera (true for sources that are a single
    continuous shot, not a two-camera side-by-side studio composite) disables the left/right-camera clustering in
    _face_crop_offset and, when no face is found at all, falls back to a
    literal center crop instead of _blind_right_biased_offset's two-camera
    guess - that guess assumes a studio layout this kind of source doesn't
    have, and would otherwise push a real single subject off-frame.
    """
    try:
        import cv2  # noqa: F401
    except ImportError as e:
        print(f"[render] opencv 미설치/로드 실패로 가운데 크롭을 사용합니다: {e}")
        return _CENTER_CROP

    dims = _probe_dimensions(clip_path)
    if dims is None:
        print("[render] ffprobe 실패로 가운데 크롭을 사용합니다.")
        return _CENTER_CROP
    src_w, src_h = dims
    # FACE_CROP_ZOOM here must exactly match the extra zoom build_filter_complex
    # applies to its own scale stage - otherwise the offsets computed here
    # would reference a different scaled frame than the one ffmpeg actually
    # produces, and the crop would land on the wrong region entirely.
    zoom = FACE_CROP_ZOOM if single_camera else two_camera_zoom(src_w, src_h, canvas_w, canvas_h)
    scale = max(canvas_w / src_w, canvas_h / src_h) * zoom
    scaled_w, scaled_h = src_w * scale, src_h * scale

    frames_dir = clip_path.parent / "face_sample_frames"
    frames_dir.mkdir(exist_ok=True)
    try:
        # 1 fps (was 1/3): clips here are short (a couple minutes at most),
        # so this is still cheap, and more sample frames meaningfully
        # improves the odds of catching at least one where a Haar cascade
        # actually fires - the single biggest lever available without
        # switching to a stronger (and asset-bundling) detector.
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(clip_path), "-vf", "fps=1",
                str(frames_dir / "f_%03d.jpg"),
            ],
            check=True, capture_output=True,
        )
        all_faces: list[tuple[float, float, float, float]] = []
        for frame_path in sorted(frames_dir.glob("f_*.jpg")):
            all_faces.extend(_detect_faces(frame_path))
    except subprocess.CalledProcessError as e:
        print(f"[render] 프레임 추출 실패로 가운데 크롭을 사용합니다: {e}")
        return _CENTER_CROP
    finally:
        shutil.rmtree(frames_dir, ignore_errors=True)

    offset = _face_crop_offset(
        all_faces, scale, canvas_w, canvas_h, scaled_w, scaled_h, src_w, single_camera=single_camera,
    )
    if offset is None:
        if single_camera:
            max_x_off = max(0.0, scaled_w - canvas_w)
            max_y_off = max(0.0, scaled_h - canvas_h)
            offset = (round(max_x_off / 2), round(max_y_off / 2))
            print(f"[render] 샘플 프레임에서 얼굴을 찾지 못해 가운데 크롭을 사용합니다: x={offset[0]}, y={offset[1]}")
        else:
            offset = _blind_right_biased_offset(scale, canvas_w, canvas_h, scaled_w, scaled_h, src_w)
            print(f"[render] 샘플 프레임에서 얼굴을 찾지 못해 우측 인물을 기준으로 크롭합니다: x={offset[0]}, y={offset[1]}")
        return str(offset[0]), str(offset[1])
    print(f"[render] 얼굴 {len(all_faces)}개 인식, 크롭 위치 조정: x={offset[0]}, y={offset[1]}")
    return str(offset[0]), str(offset[1])


def _logo_content_bbox(image_path: Path) -> tuple[int, int, int, int] | None:
    """Return the (left, top, right, bottom) pixel box of an image's
    non-transparent content, or None if it can't be determined.

    Uploaded logo art is often exported on a larger canvas than the visible
    mark itself, padded with transparent pixels. Left uncropped, that
    padding scales along with the logo and reappears as visible empty space
    once overlaid - fixing it requires cropping to the actual content
    before scaling, not just tightening the overlay margin.
    """
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        with Image.open(image_path) as img:
            alpha = img.convert("RGBA").split()[-1]
            return alpha.getbbox()
    except Exception:
        return None


def _logo_scaled_size(bbox: tuple[int, int, int, int] | None) -> tuple[float, float]:
    """The actual on-canvas (width, height) a top-corner logo renders at
    after force_original_aspect_ratio=decrease fits it into the
    LOGO_IMAGE_MAX_WIDTH x LOGO_IMAGE_HEIGHT box - used to size the backing
    box in build_filter_complex to the logo's own footprint rather than the
    (often much larger, for a thin wordmark) full box. Falls back to the
    full box size when bbox is unavailable (PIL missing)."""
    if bbox is None:
        return (LOGO_IMAGE_MAX_WIDTH, LOGO_IMAGE_HEIGHT)
    content_w, content_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    scale = min(LOGO_IMAGE_MAX_WIDTH / content_w, LOGO_IMAGE_HEIGHT / content_h)
    return content_w * scale, content_h * scale


def _logo_overlay_y(bbox: tuple[int, int, int, int] | None) -> int:
    """Vertical overlay offset for a top-corner logo, centered within the
    shared LOGO_IMAGE_HEIGHT box instead of just top-aligned - two logos of
    different aspect ratios scaled to fit the same box
    (force_original_aspect_ratio=decrease) end up at different actual
    heights, so top-aligning both would put their centers out of line.
    Falls back to the plain top-aligned offset when bbox is unavailable
    (PIL missing), since the scaled height can't be computed without it.
    """
    base_y = TITLE_BAND_H + LOGO_MARGIN_Y
    _scaled_w, scaled_h = _logo_scaled_size(bbox)
    return base_y + round((LOGO_IMAGE_HEIGHT - scaled_h) / 2)


def _bottom_image_overlay_y(bbox: tuple[int, int, int, int] | None) -> int:
    """Vertical overlay offset that centers the bottom band image within
    the BOTTOM_IMAGE_BAND_H band at BOTTOM_IMAGE_BAND_TOP, analogous to
    _logo_overlay_y for the top-corner logos."""
    base_y = BOTTOM_IMAGE_BAND_TOP
    if bbox is None:
        return base_y
    content_w, content_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    scale = min(BOTTOM_IMAGE_MAX_WIDTH / content_w, BOTTOM_IMAGE_HEIGHT / content_h)
    scaled_h = content_h * scale
    return base_y + round((BOTTOM_IMAGE_BAND_H - scaled_h) / 2)


def build_filter_complex(
    thumbnail_text: str = "",
    captions: list[tuple[float, float, str]] | None = None,
    display_font_path: str = DISPLAY_FONT,
    logo_top_left: Path | None = None,
    logo_top_right: Path | None = None,
    banner_bottom: Path | None = None,
    guest_label_text: str | None = None,
    crop_x: str = _CENTER_CROP[0],
    crop_y: str = _CENTER_CROP[1],
    zoom: float = FACE_CROP_ZOOM,
) -> tuple[str, str, str, list[Path]]:
    """Return (filter_complex, video_map_label, audio_map_label, extra_inputs).

    crop_x/crop_y are ffmpeg crop-filter position expressions (or plain
    pixel offsets), in the coordinate space of the scaled frame. Defaults
    to a plain center crop; the caller overrides them with a face-detected
    offset when possible so a speaker isn't cropped out by a blind center
    crop.

    guest_label_text, if given, is a single pre-formatted line (e.g. "김은비
    변호사 / 손해보험협회") shown for the whole clip just above the caption
    band, identifying who's on screen.

    extra_inputs is the ordered list of image files (logo_top_left,
    logo_top_right, banner_bottom, in that order, whichever are given) the
    caller must also pass as ffmpeg "-i" arguments right after the main
    video input - the filter graph references them as [1:v], [2:v], etc.
    """
    captions = captions or []
    # the video sits between the top title band and the bottom info band
    # (guest name/caption/bottom image, see BOTTOM_BAND_H) - both padded in
    # as solid black rather than overlaid on top of the video, so it's
    # cropped to fill only the space actually left between them
    video_h = CANVAS_H - TITLE_BAND_H - BOTTOM_BAND_H
    extra_inputs: list[Path] = []

    # scale to FACE_CROP_ZOOM's inflated target, not the plain canvas size -
    # see find_speaker_crop, which computes crop_x/crop_y against this same
    # inflated scale so the two stay in sync
    zoom_w = round(CANVAS_W * zoom)
    zoom_h = round(video_h * zoom)
    stages = [
        f"[0:v]scale={zoom_w}:{zoom_h}:force_original_aspect_ratio=increase:flags=lanczos,"
        f"crop={CANVAS_W}:{video_h}:{crop_x}:{crop_y}[cropped]",
        # video sits between the title band and the bottom info band;
        # padding to the full canvas leaves both as black automatically
        f"[cropped]pad={CANVAS_W}:{CANVAS_H}:0:{TITLE_BAND_H}:color=black[padded]",
    ]

    line1, line2 = _title_lines(thumbnail_text)
    last = "padded"
    stages.append(
        f"[{last}]drawtext=fontfile={display_font_path}:expansion=none:text='{escape_drawtext(line1)}':"
        f"fontcolor=white:fontsize={TITLE_FONTSIZE}:"
        f"x=(w-text_w)/2:y={TITLE_LINE1_Y}[t1]"
    )
    last = "t1"
    if line2:
        stages.append(
            f"[{last}]drawtext=fontfile={display_font_path}:expansion=none:text='{escape_drawtext(line2)}':"
            f"fontcolor={ACCENT_COLOR}:fontsize={TITLE_FONTSIZE}:"
            f"x=(w-text_w)/2:y={TITLE_LINE2_Y}[t2]"
        )
        last = "t2"

    if logo_top_left is not None and logo_top_left.is_file():
        extra_inputs.append(logo_top_left)
        idx = len(extra_inputs)
        bbox = _logo_content_bbox(logo_top_left)
        crop_stage = (
            f"crop={bbox[2] - bbox[0]}:{bbox[3] - bbox[1]}:{bbox[0]}:{bbox[1]}," if bbox else ""
        )
        stages.append(
            f"[{idx}:v]{crop_stage}scale=w={LOGO_IMAGE_MAX_WIDTH}:h={LOGO_IMAGE_HEIGHT}:"
            f"force_original_aspect_ratio=decrease[lgimg{idx}]"
        )
        stages.append(
            f"[{last}][lgimg{idx}]overlay=x={LOGO_MARGIN_X}:y={_logo_overlay_y(bbox)}[lg1]"
        )
        last = "lg1"

    if logo_top_right is not None and logo_top_right.is_file():
        extra_inputs.append(logo_top_right)
        idx = len(extra_inputs)
        bbox = _logo_content_bbox(logo_top_right)
        crop_stage = (
            f"crop={bbox[2] - bbox[0]}:{bbox[3] - bbox[1]}:{bbox[0]}:{bbox[1]}," if bbox else ""
        )
        stages.append(
            f"[{idx}:v]{crop_stage}scale=w={LOGO_IMAGE_MAX_WIDTH}:h={LOGO_IMAGE_HEIGHT}:"
            f"force_original_aspect_ratio=decrease[lgimg{idx}]"
        )
        stages.append(
            f"[{last}][lgimg{idx}]overlay=x=main_w-overlay_w-{LOGO_MARGIN_X}:"
            f"y={_logo_overlay_y(bbox)}[lg2]"
        )
        last = "lg2"

    if banner_bottom is not None and banner_bottom.is_file():
        stages.append(
            f"[{last}]drawbox=x=0:y={BOTTOM_IMAGE_BAND_TOP}:w={CANVAS_W}:h={BOTTOM_IMAGE_BAND_H}:"
            f"color=black@{BOTTOM_IMAGE_BAND_OPACITY}:t=fill[botbox]"
        )
        last = "botbox"
        extra_inputs.append(banner_bottom)
        idx = len(extra_inputs)
        bbox = _logo_content_bbox(banner_bottom)
        crop_stage = (
            f"crop={bbox[2] - bbox[0]}:{bbox[3] - bbox[1]}:{bbox[0]}:{bbox[1]}," if bbox else ""
        )
        stages.append(
            f"[{idx}:v]{crop_stage}scale=w={BOTTOM_IMAGE_MAX_WIDTH}:h={BOTTOM_IMAGE_HEIGHT}:"
            f"force_original_aspect_ratio=decrease[botimg{idx}]"
        )
        stages.append(
            f"[{last}][botimg{idx}]overlay=x=(main_w-overlay_w)/2:y={_bottom_image_overlay_y(bbox)}[botimg]"
        )
        last = "botimg"

    if guest_label_text:
        stages.append(
            f"[{last}]drawbox=x=0:y={GUEST_LABEL_BAND_TOP}:w={CANVAS_W}:h={GUEST_LABEL_BAND_H}:"
            f"color=black@{GUEST_LABEL_BOX_OPACITY}:t=fill[guestbox]"
        )
        stages.append(
            f"[guestbox]drawtext=fontfile={display_font_path}:expansion=none:"
            f"text='{escape_drawtext(guest_label_text)}':"
            f"fontcolor={ACCENT_COLOR}:fontsize={GUEST_LABEL_FONTSIZE}:"
            f"x=(w-text_w)/2:y={GUEST_LABEL_BAND_TOP}+({GUEST_LABEL_BAND_H}-text_h)/2[guestlabel]"
        )
        last = "guestlabel"

    # over-long cues become several consecutive on-screen captions so no text
    # runs past the screen edge or the caption band
    captions = [c for cue in captions for c in _caption_chunks(*cue)]
    for i, (cue_start, cue_end, text) in enumerate(captions):
        # between(t,start,end) is inclusive on *both* ends, so two cues that
        # share an exact touching boundary (cue1.end == cue2.start, the
        # normal case for consecutive transcript segments) would both
        # evaluate true for the single frame landing exactly on that
        # instant - a one-frame double-exposure at every caption change.
        # gte(t,start)*lt(t,end) makes the window half-open ([start, end))
        # so consecutive cues never both draw on the same frame.
        enable = f"enable='gte(t,{cue_start})*lt(t,{cue_end})'"
        cap_lines = text.split("\n", 1)
        cap_line1 = cap_lines[0]
        cap_line2 = cap_lines[1] if len(cap_lines) > 1 else None

        band_label = f"capband{i}"
        stages.append(
            f"[{last}]drawbox=x=0:y={CAPTION_BAND_TOP}:w={CANVAS_W}:h={CAPTION_BAND_H}:"
            f"color=black@{CAPTION_BOX_OPACITY}:t=fill:{enable}[{band_label}]"
        )
        last = band_label

        line1_y = CAPTION_BAND_TOP + CAPTION_TEXT_PADDING_TOP
        label = f"cap{i}a"
        stages.append(
            f"[{last}]drawtext=fontfile={display_font_path}:expansion=none:text='{escape_drawtext(cap_line1)}':"
            f"fontcolor=white:fontsize={CAPTION_FONTSIZE}:borderw=3:bordercolor=black:"
            f"x=(w-text_w)/2:y={line1_y}:{enable}[{label}]"
        )
        last = label

        if cap_line2:
            line2_y = line1_y + CAPTION_LINE_HEIGHT
            label2 = f"cap{i}b"
            stages.append(
                f"[{last}]drawtext=fontfile={display_font_path}:expansion=none:text='{escape_drawtext(cap_line2)}':"
                f"fontcolor=white:fontsize={CAPTION_FONTSIZE}:borderw=3:bordercolor=black:"
                f"x=(w-text_w)/2:y={line2_y}:{enable}[{label2}]"
            )
            last = label2

    stages.append(f"[{last}]null[vout]")
    stages.append("[0:a]anull[aout]")

    return ";".join(stages), "[vout]", "[aout]", extra_inputs


def _run_ffmpeg(
    clip_path: Path,
    work_dir: Path,
    out_path: Path,
    filter_complex: str,
    v_label: str,
    a_label: str,
    extra_inputs: list[Path],
    crf: int,
    preset: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-i", str(clip_path)]
    for image_path in extra_inputs:
        cmd += ["-i", str(image_path)]
    cmd += [
        "-filter_complex", filter_complex,
        "-map", v_label,
        "-map", a_label,
        "-c:v", "libx264",
        "-preset", preset,
        "-crf", str(crf),
        "-c:a", "aac",
        "-b:a", "160k",
        "-movflags", "+faststart",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, cwd=work_dir)


def _local_font(display_font_path: str, work_dir: Path) -> str:
    """ffmpeg's filter option syntax uses ':' as a key=value separator, so a
    raw font path containing one (e.g. a Windows drive letter like C:\\...)
    needs escaping - and that escaping is finicky and inconsistent across
    ffmpeg versions. Sidestep it entirely: copy the font next to the clip
    and run ffmpeg with that as its cwd, so the filter can reference it by
    a bare filename that never contains a colon."""
    local_path = work_dir / f"display_font{Path(display_font_path).suffix}"
    shutil.copyfile(display_font_path, local_path)
    return local_path.name


def render_draft(
    clip_path: Path,
    captions: list[tuple[float, float, str]],
    out_path: Path,
    work_dir: Path | None = None,
) -> Path:
    """Fast preview: crop + burned-in captions only, no title banner, logos,
    or guest label - just enough for the worker to proofread captions
    against picture (step 7). Uses a plain center crop (skipping the
    face-detection frame sampling) and a faster encode preset, since
    accurate framing doesn't matter yet at this stage.

    `captions` is the (start, end, text) cue list for the built clip's own
    timeline - get the initial version from captions_for_ranges(segments,
    ranges), shown to the worker for proofreading; pass it back in
    (with corrected text, same timings) to render_final so the fix carries
    through to the final render.
    """
    work_dir = work_dir or out_path.parent
    work_dir.mkdir(parents=True, exist_ok=True)

    local_display_font = _local_font(DISPLAY_FONT, work_dir)

    filter_complex, v_label, a_label, extra_inputs = build_filter_complex(
        captions=captions, display_font_path=local_display_font,
    )
    _run_ffmpeg(
        clip_path, work_dir, out_path, filter_complex, v_label, a_label, extra_inputs,
        crf=23, preset="veryfast",
    )
    return out_path


def render_final(
    clip_path: Path,
    captions: list[tuple[float, float, str]],
    assets: RenderAssets,
    out_path: Path,
    work_dir: Path | None = None,
    single_camera: bool = False,
) -> Path:
    """Full composite (step 9): face-detected crop, title banner, uploaded
    logos/bottom image, guest label, and captions - pass in the same
    (start, end, text) cues shown at the draft stage, with the worker's
    typo corrections applied to the text (see render_draft).

    single_camera - see find_speaker_crop - defaults to False (a two-camera
    side-by-side studio composite; 범용 서버는 항상 True를 넘긴다); pass True for a genuine single-camera source.
    """
    work_dir = work_dir or out_path.parent
    work_dir.mkdir(parents=True, exist_ok=True)

    local_display_font = _local_font(DISPLAY_FONT, work_dir)

    video_h = CANVAS_H - TITLE_BAND_H - BOTTOM_BAND_H
    crop_x, crop_y = find_speaker_crop(clip_path, CANVAS_W, video_h, single_camera=single_camera)
    zoom = FACE_CROP_ZOOM
    dims = _probe_dimensions(clip_path)
    if dims is not None and not single_camera:
        zoom = two_camera_zoom(dims[0], dims[1], CANVAS_W, video_h)

    filter_complex, v_label, a_label, extra_inputs = build_filter_complex(
        thumbnail_text=assets.thumbnail_text,
        captions=captions,
        display_font_path=local_display_font,
        logo_top_left=assets.logo_top_left,
        logo_top_right=assets.logo_top_right,
        banner_bottom=assets.banner_bottom,
        guest_label_text=with_shoot_date(assets.guest_label, assets.shoot_date, assets.date_kind) or None,
        crop_x=crop_x, crop_y=crop_y, zoom=zoom,
    )
    _run_ffmpeg(
        clip_path, work_dir, out_path, filter_complex, v_label, a_label, extra_inputs,
        crf=18, preset="slow",
    )
    return out_path
