from autoshorts.analyze import ClipRange
from autoshorts.render import (
    build_filter_complex,
    captions_for_ranges,
    escape_drawtext,
    _blind_right_biased_offset,
    _clip_overlapping_cues,
    _face_crop_offset,
    _title_lines,
    _caption_chunks,
    _wrap_caption,
)
from autoshorts.transcript import Segment


def test_escape_drawtext_escapes_colon_and_backslash():
    assert escape_drawtext("20%p 폭락") == "20%p 폭락"
    assert escape_drawtext("C:\\path") == "C\\:\\\\path"


def test_escape_drawtext_replaces_newlines_by_default():
    assert escape_drawtext("한 줄\n두 줄") == "한 줄 두 줄"


def test_escape_drawtext_can_keep_newlines():
    assert escape_drawtext("한 줄\n두 줄", keep_newlines=True) == "한 줄\n두 줄"


def test_title_lines_splits_on_first_newline():
    assert _title_lines("로봇택시\n취객은 누가 깨울까?") == ("로봇택시", "취객은 누가 깨울까?")
    assert _title_lines("한 줄만") == ("한 줄만", None)


def test_wrap_caption_leaves_short_text_alone():
    assert _wrap_caption("짧은 자막") == "짧은 자막"


def test_wrap_caption_breaks_long_text_near_midpoint_on_space():
    text = "이것은 상당히 긴 자막 문장이라 두 줄로 나눠야 한다"
    wrapped = _wrap_caption(text, max_chars=14)
    assert "\n" in wrapped
    line1, line2 = wrapped.split("\n")
    assert line1 + " " + line2 == text


def test_captions_for_ranges_offsets_each_range_by_prior_durations():
    segments = [
        Segment(0.0, 3.0, "a"),
        Segment(3.0, 5.0, "b"),
        Segment(6.0, 9.0, "c"),
    ]
    ranges = [ClipRange(1.0, 4.0), ClipRange(6.0, 8.0)]
    cues = captions_for_ranges(segments, ranges)
    # range 1 [1,4): "a" -> (0,2), "b" -> (2,3); range 2 [6,8) starts at
    # clip_offset=3.0 (4.0-1.0): "c" -> (3,5)
    assert cues == [(0.0, 2.0, "a"), (2.0, 3.0, "b"), (3.0, 5.0, "c")]


def test_captions_for_ranges_drops_segments_outside_every_range():
    segments = [Segment(0.0, 5.0, "in"), Segment(20.0, 25.0, "out")]
    cues = captions_for_ranges(segments, [ClipRange(0.0, 5.0)])
    assert cues == [(0.0, 5.0, "in")]


def test_build_filter_complex_includes_title_and_caption_text():
    filter_complex, v_label, a_label, extra_inputs = build_filter_complex(
        thumbnail_text="제목1\n제목2",
        captions=[(0.5, 2.0, "자막 문장")],
    )
    assert "제목1" in filter_complex
    assert "제목2" in filter_complex
    assert "자막 문장" in filter_complex
    assert v_label == "[vout]"
    assert a_label == "[aout]"
    assert extra_inputs == []


def test_build_filter_complex_skips_decorations_when_omitted():
    filter_complex, _, _, extra_inputs = build_filter_complex(captions=[])
    assert "drawbox" not in filter_complex  # no guest label / bottom image band
    assert extra_inputs == []


def test_clip_overlapping_cues_trims_end_to_next_start():
    # YouTube rolling auto-captions routinely overlap like this - without
    # clipping, both cues would satisfy their enable() window at once and
    # burn in as a dissolve/overlap instead of a clean cut.
    cues = [(0.0, 3.0, "a"), (2.0, 5.0, "b"), (4.5, 8.0, "c")]
    assert _clip_overlapping_cues(cues) == [(0.0, 2.0, "a"), (2.0, 4.5, "b"), (4.5, 8.0, "c")]


def test_clip_overlapping_cues_drops_fully_swallowed_cue():
    # a cue entirely inside the next one's overlap window would clip to a
    # zero/negative duration - drop it rather than emit an invalid cue.
    cues = [(0.0, 5.0, "a"), (1.0, 2.0, "b"), (1.0, 6.0, "c")]
    assert _clip_overlapping_cues(cues) == [(0.0, 1.0, "a"), (1.0, 6.0, "c")]


def test_captions_for_ranges_excludes_deselected_segment_despite_overlap():
    # a worker deselecting the middle segment of a group splits it into two
    # ranges around the deselected one - its own overlap with either
    # range's edge (from segments overlapping their neighbors) must not be
    # enough to sneak its caption into either range's video.
    segments = [
        Segment(0.0, 5.0, "keep-a"),
        Segment(4.0, 9.0, "deselected"),
        Segment(8.0, 12.0, "keep-b"),
    ]
    ranges = [ClipRange(0.0, 5.0), ClipRange(8.0, 12.0)]
    cues = captions_for_ranges(segments, ranges)
    texts = [text for _, _, text in cues]
    assert texts == ["keep-a", "keep-b"]


def test_captions_for_ranges_clips_overlapping_transcript_segments():
    segments = [
        Segment(0.0, 3.0, "a"),
        Segment(2.0, 5.0, "b"),
    ]
    cues = captions_for_ranges(segments, [ClipRange(0.0, 5.0)])
    assert cues == [(0.0, 2.0, "a"), (2.0, 5.0, "b")]


def test_face_crop_offset_prefers_right_side_faces():
    faces = [(10.0, 10.0, 50.0, 50.0), (200.0, 10.0, 50.0, 50.0)]
    offset = _face_crop_offset(
        faces, scale=1.0, canvas_w=100, canvas_h=100, scaled_w=300.0, scaled_h=300.0, src_w=300.0,
    )
    assert offset is not None
    # right face center is x=225, so the crop window (100 wide) should be
    # pulled toward that side rather than sitting at the frame's midpoint
    assert offset[0] > 100  # would be ~100 for a plain center crop


def test_blind_right_biased_offset_is_right_of_center():
    x_off, _ = _blind_right_biased_offset(
        scale=1.0, canvas_w=100, canvas_h=100, scaled_w=300.0, scaled_h=300.0, src_w=300.0,
    )
    center_x_off = (300.0 - 100) / 2  # what a plain center crop would use
    assert x_off > center_x_off


def test_face_crop_offset_single_camera_averages_both_sides_instead_of_preferring_right():
    # same two faces as test_face_crop_offset_prefers_right_side_faces, but a
    # single continuous camera shot has no studio-camera seam
    # to dodge, so both faces are real subjects in one frame and should be
    # averaged, not collapsed onto whichever one happens to sit on the right.
    faces = [(10.0, 10.0, 50.0, 50.0), (200.0, 10.0, 50.0, 50.0)]
    offset = _face_crop_offset(
        faces, scale=1.0, canvas_w=100, canvas_h=100, scaled_w=300.0, scaled_h=300.0, src_w=300.0,
        single_camera=True,
    )
    assert offset is not None
    # midpoint of both face centers (35 and 225) is 130, close to frame
    # center (150) - not pulled toward the right face alone like the
    # two-camera default would (offset[0] > 100 there).
    assert offset[0] < 100


def test_two_camera_zoom_makes_one_half_fill_the_canvas():
    from autoshorts.render import SEAM_MARGIN_PX, two_camera_zoom

    zoom = two_camera_zoom(1920, 1080, 1080, 906)
    scale = max(1080 / 1920, 906 / 1080) * zoom
    assert 1920 * scale / 2 >= 1080 + SEAM_MARGIN_PX - 1


def test_face_crop_offset_never_shows_the_seam_for_right_side_speaker():
    from autoshorts.render import SEAM_MARGIN_PX, two_camera_zoom

    zoom = two_camera_zoom(1920, 1080, 1080, 906)
    scale = max(1080 / 1920, 906 / 1080) * zoom
    scaled_w, scaled_h = 1920 * scale, 1080 * scale
    # speaker near the middle of the right half would otherwise pull the
    # window left across the seam
    faces = [(1100.0, 300.0, 150.0, 150.0)]
    x_off, _ = _face_crop_offset(faces, scale, 1080, 906, scaled_w, scaled_h, 1920)
    assert x_off >= scaled_w / 2 + SEAM_MARGIN_PX - 1


def test_wrap_caption_never_exceeds_hard_line_limit():
    from autoshorts.render import CAPTION_HARD_MAX_CHARS_PER_LINE as LIMIT
    # the real-world overflow case: halves of 11 + 9 chars are fine, but a
    # sentence whose midpoint split leaves a 20-char line must be re-wrapped
    text = "휴먼로이드를 만들겠다라고 얘기를 하고 있고요. 다만 상반기에 실제로 출하가"
    for line in _wrap_caption(text).split("\n"):
        assert len(line) <= LIMIT
    # no spaces at all: hard split instead of one giant line
    for line in _wrap_caption("가" * 40).split("\n"):
        assert len(line) <= LIMIT


def test_caption_chunks_splits_long_cue_into_two_line_pieces():
    text = "휴먼로이드를 만들겠다라고 얘기를 하고 있고요. 다만 상반기에 실제로 출하가 시작될 예정이라고 합니다"
    chunks = _caption_chunks(10.0, 20.0, text)
    assert len(chunks) >= 2
    assert chunks[0][0] == 10.0 and chunks[-1][1] == 20.0
    for (_, e1, _), (s2, _, _) in zip(chunks, chunks[1:]):
        assert e1 == s2  # contiguous, no gap/overlap
    for _, _, t in chunks:
        assert len(t.split("\n")) <= 2


def test_caption_chunks_leaves_short_cue_alone():
    assert _caption_chunks(1.0, 2.0, "짧은 자막") == [(1.0, 2.0, "짧은 자막")]
