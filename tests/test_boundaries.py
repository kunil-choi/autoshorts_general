from types import SimpleNamespace

from autoshorts import boundaries
from autoshorts.analyze import Candidate, ClipRange, refine_boundaries
from autoshorts.config import ClipLengthPreset
from autoshorts.transcript import Segment

SILENCE_LOG = """
[silencedetect @ 0x1] silence_start: 1.2
[silencedetect @ 0x1] silence_end: 1.9 | silence_duration: 0.7
[silencedetect @ 0x1] silence_start: 20.5
"""


def test_parse_silencedetect_closes_trailing_silence_at_total():
    assert boundaries.parse_silencedetect(SILENCE_LOG, 25.0) == [(1.2, 1.9), (20.5, 25.0)]


def test_choose_start_cuts_just_before_speech_resumes():
    # intended start 2.0 is mid-speech, pause 1.2-1.9 just before it
    assert abs(boundaries.choose_start([(1.2, 1.9)], 2.0) - 1.75) < 1e-9


def test_choose_start_unchanged_when_no_pause_nearby():
    assert boundaries.choose_start([(10.0, 11.0)], 2.0) == 2.0


def test_choose_end_waits_for_the_next_pause_after_the_intended_end():
    # speaker is still talking at 30.0; the pause begins at 31.0
    assert abs(boundaries.choose_end([(31.0, 32.0)], 30.0) - 31.35) < 1e-9


def test_choose_end_limits_tail_to_pause_length():
    assert boundaries.choose_end([(31.0, 31.1)], 30.0) == 31.1


def test_choose_end_falls_back_to_earlier_pause_then_unchanged():
    assert abs(boundaries.choose_end([(29.4, 29.8)], 30.0) - 29.75) < 1e-9
    assert boundaries.choose_end([], 30.0) == 30.0


def _segs():
    return [Segment(float(i * 5), float(i * 5 + 5), f"줄 {i}") for i in range(20)]


def _candidate(start, end):
    return Candidate(ranges=[ClipRange(start, end)], title="t", summary="s", thumbnail_text="a\nb", reason="r")


class _Client:
    def __init__(self, start_line, end_line):
        self.messages = self
        self.start_line, self.end_line = start_line, end_line
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        tu = SimpleNamespace(type="tool_use", input={"start_line": self.start_line, "end_line": self.end_line})
        return SimpleNamespace(content=[tu])


_PRESET = ClipLengthPreset(key="t", label="t", min_sec=10, max_sec=60)


def test_refine_extends_to_sentence_boundaries():
    # candidate covers lines 4..9 (20s-50s); model says lines 3..10
    c = _candidate(20.0, 50.0)
    out = refine_boundaries([c], _segs(), _PRESET, client=_Client(3, 10))
    assert (out[0].ranges[0].start_sec, out[0].ranges[0].end_sec) == (15.0, 55.0 )


def test_refine_keeps_original_when_too_long_or_out_of_window():
    c = _candidate(20.0, 50.0)
    # lines 0..19 would be 100s > max 60
    assert refine_boundaries([c], _segs(), _PRESET, client=_Client(0, 19))[0].ranges == c.ranges
    # outside the context window
    assert refine_boundaries([c], _segs(), _PRESET, client=_Client(19, 19))[0].ranges == c.ranges


def test_refine_keeps_original_on_api_error():
    class Boom:
        messages = None

        def __init__(self):
            self.messages = self

        def create(self, **kw):
            raise RuntimeError("api down")

    c = _candidate(20.0, 50.0)
    assert refine_boundaries([c], _segs(), _PRESET, client=Boom())[0].ranges == c.ranges


# ---- panelist-turn alignment -------------------------------------------

def _panel_segs():
    """0-1 앵커 | 2-9 패널(한 발언, 줄마다 5초) | 10 앵커 | 11-12 패널"""
    roles = ["앵커", "앵커"] + ["패널"] * 8 + ["앵커"] + ["패널"] * 2
    return [
        Segment(float(i * 5), float(i * 5 + 5), f"줄 {i}다", speaker=r) for i, r in enumerate(roles)
    ]


def test_align_extends_end_to_finish_the_panelists_turn():
    from autoshorts.analyze import _align_to_panel_turns
    # Claude ended on line 6 while the panelist keeps talking until line 9
    first, last = _align_to_panel_turns(2, 6, _panel_segs(), _PRESET)
    assert (first, last) == (2, 9)


def test_align_trims_trailing_anchor_lines():
    from autoshorts.analyze import _align_to_panel_turns
    first, last = _align_to_panel_turns(2, 10, _panel_segs(), _PRESET)
    assert last == 9


def test_align_moves_start_back_to_the_turn_start():
    from autoshorts.analyze import _align_to_panel_turns
    first, last = _align_to_panel_turns(5, 9, _panel_segs(), _PRESET)
    assert (first, last) == (2, 9)


def test_align_starting_on_the_anchor_question_is_allowed():
    from autoshorts.analyze import _align_to_panel_turns
    assert _align_to_panel_turns(1, 9, _panel_segs(), _PRESET) == (1, 9)


def test_align_respects_max_length():
    from autoshorts.analyze import _align_to_panel_turns
    short = ClipLengthPreset(key="s", label="s", min_sec=10, max_sec=30)
    # whole turn (2..9 = 40s) can't fit in 30s: end is cut at a sentence end that fits
    first, last = _align_to_panel_turns(2, 5, _panel_segs(), short)
    assert _panel_segs()[last].end_sec - _panel_segs()[first].start_sec <= 30
    # start can't move back without exceeding the limit -> unchanged
    assert _align_to_panel_turns(5, 9, _panel_segs(), short)[0] == 5


def test_align_does_nothing_without_speaker_labels():
    from autoshorts.analyze import _align_to_panel_turns
    assert _align_to_panel_turns(4, 9, _segs(), _PRESET) == (4, 9)


def test_refine_applies_panel_alignment_after_claude_answer():
    segs = _panel_segs()
    out = refine_boundaries([_candidate(25.0, 50.0)], segs, _PRESET, client=_Client(5, 8))
    # Claude picked 5..8 (mid-turn both sides) -> aligned to the whole 2..9 turn
    assert (out[0].ranges[0].start_sec, out[0].ranges[0].end_sec) == (10.0, 50.0)


def test_format_for_prompt_tags_speakers():
    from autoshorts.transcript import format_for_prompt
    out = format_for_prompt([Segment(0, 5, "질문", speaker="앵커"), Segment(5, 9, "답", speaker="패널"), Segment(9, 12, "무라벨")])
    assert out.splitlines() == ["[00:00-00:05] [앵커] 질문", "[00:05-00:09] [패널] 답", "[00:09-00:12] 무라벨"]
