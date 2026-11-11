from types import SimpleNamespace

from autoshorts.analyze import (
    Candidate,
    ClipRange,
    _closest_boundary,
    _keyword_philosophy,
    _hms_to_sec,
    _sec_to_hms,
    _snap_to_segment_boundaries,
    propose_candidates,
)
from autoshorts.config import ClipLengthPreset
from autoshorts.transcript import Segment


def test_hms_to_sec_and_back():
    assert _hms_to_sec("01:02:03") == 3723.0
    assert _hms_to_sec("02:03") == 123.0
    assert _sec_to_hms(3723) == "01:02:03"
    assert _sec_to_hms(123) == "02:03"


def test_closest_boundary_snaps_within_tolerance():
    boundaries = [0.0, 5.0, 10.0]
    assert _closest_boundary(5.4, boundaries) == 5.0
    # far outside tolerance (1.5s) - left as-is
    assert _closest_boundary(7.0, boundaries) == 7.0


def test_closest_boundary_empty_list_is_noop():
    assert _closest_boundary(5.0, []) == 5.0


def test_snap_to_segment_boundaries_corrects_all_ranges_in_a_candidate():
    segments = [Segment(0, 5, "a"), Segment(5, 10, "b"), Segment(10, 15, "c")]
    candidates = [
        Candidate(
            ranges=[ClipRange(0.2, 4.9), ClipRange(10.3, 14.8)],
            title="t", summary="s", thumbnail_text="x", reason="r", tier="hook",
        )
    ]
    snapped = _snap_to_segment_boundaries(candidates, segments)
    assert snapped[0].ranges[0] == ClipRange(0.0, 5.0)
    assert snapped[0].ranges[1] == ClipRange(10.0, 15.0)


def test_keyword_philosophy_embeds_keyword_and_count():
    text = _keyword_philosophy("금리 인상 관련 발언", count=3)
    assert "금리 인상 관련 발언" in text
    assert "3개보다 적게" in text


class _RecordingClient:
    """Stands in for anthropic.Anthropic: records each messages.create call
    and answers with one valid candidate via the forced tool."""

    def __init__(self):
        self.calls = []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        tool_use = SimpleNamespace(type="tool_use", input={"candidates": [{
            "start_hms": "00:00", "end_hms": "00:20", "title": "t", "summary": "s",
            "thumbnail_text": "a\nb", "reason": "r",
        }]})
        return SimpleNamespace(content=[tool_use])


def _segments():
    return [Segment(float(i * 10), float(i * 10 + 10), f"문장 {i}") for i in range(6)]


def _preset():
    return ClipLengthPreset(key="test", label="test", min_sec=15, max_sec=45)


def test_tier_calls_share_a_cached_transcript_prefix():
    client = _RecordingClient()
    propose_candidates(_segments(), _preset(), ["hook", "substantive"], video_title="제목", client=client)
    hook, substantive = client.calls
    assert hook["tools"] == substantive["tools"]
    assert hook["system"][0] == substantive["system"][0]
    assert hook["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert "문장 5" in hook["system"][0]["text"] and "제목" in hook["system"][0]["text"]
    assert hook["system"][1]["text"] != substantive["system"][1]["text"]
    assert "문장 5" not in hook["messages"][0]["content"]


def test_single_tier_run_skips_the_cache_write():
    client = _RecordingClient()
    propose_candidates(_segments(), _preset(), ["hook"], client=client)
    assert "cache_control" not in client.calls[0]["system"][0]



def test_keyword_tier_only_runs_with_a_keyword_and_gets_its_own_count():
    client = _RecordingClient()
    propose_candidates(_segments(), _preset(), ["hook", "keyword"], client=client)  # no keyword -> skipped
    assert len(client.calls) == 1

    client = _RecordingClient()
    propose_candidates(
        _segments(), _preset(), ["hook", "substantive", "keyword"], keyword="신제품 가격",
        counts={"hook": 1, "substantive": 1, "keyword": 4}, client=client,
    )
    assert len(client.calls) == 3
    keyword_call = client.calls[2]
    assert "신제품 가격" in keyword_call["system"][1]["text"]
    assert "정확히 4개" in keyword_call["system"][1]["text"]
    assert keyword_call["tools"][0]["input_schema"]["properties"]["candidates"]["maxItems"] == 5
    assert keyword_call["tools"] == client.calls[0]["tools"]  # 개수가 달라도 캐시 접두부 동일


def test_general_speaker_rule_is_used_when_segments_have_no_speaker_labels():
    client = _RecordingClient()
    propose_candidates(_segments(), _preset(), ["hook"], client=client)
    text = client.calls[0]["system"][1]["text"]
    assert "화자 구분이 없다" in text and "[앵커]" not in text
