from pathlib import Path

from autoshorts.transcript import (
    Segment,
    _clean_caption_text,
    excerpt_for_range,
    format_for_prompt,
    parse_vtt,
    segments_in_range,
)

FIXTURE = Path(__file__).parent / "fixtures" / "rolling_captions.ko.vtt"


def test_parse_vtt_collapses_rolling_duplicates():
    segments = parse_vtt(FIXTURE)
    texts = [s.text for s in segments]
    assert texts == ["안녕하세요", "오늘은", "경제 이야기를", "해보겠습니다"]


def test_parse_vtt_timestamps():
    segments = parse_vtt(FIXTURE)
    assert segments[0].start_sec == 0.0
    assert segments[0].end_sec == 2.0
    assert segments[-1].end_sec == 9.0


def test_format_for_prompt():
    segments = parse_vtt(FIXTURE)
    text = format_for_prompt(segments)
    assert "[00:00-00:02] 안녕하세요" in text
    assert "[00:06-00:09] 해보겠습니다" in text


def test_clean_caption_text_decodes_html_entities():
    assert _clean_caption_text("&gt;&gt; 어 계속해서") == ">> 어 계속해서"


def test_clean_caption_text_strips_tags_and_entities_together():
    assert _clean_caption_text("<b>&gt;&gt;</b> 네.") == ">> 네."


def test_excerpt_for_range_concatenates_overlapping_segments():
    segments = [
        Segment(start_sec=0.0, end_sec=5.0, text="클립 밖"),
        Segment(start_sec=12.0, end_sec=15.0, text="클립 안 첫마디"),
        Segment(start_sec=15.0, end_sec=18.0, text="클립 안 둘째마디"),
        Segment(start_sec=40.0, end_sec=45.0, text="클립 밖2"),
    ]
    excerpt = excerpt_for_range(segments, start_sec=10.0, end_sec=30.0)
    assert excerpt == "클립 안 첫마디 클립 안 둘째마디"


def test_segments_in_range_returns_full_segment_objects():
    segments = [
        Segment(start_sec=0.0, end_sec=5.0, text="클립 밖"),
        Segment(start_sec=12.0, end_sec=15.0, text="클립 안 첫마디"),
        Segment(start_sec=15.0, end_sec=18.0, text="클립 안 둘째마디"),
        Segment(start_sec=40.0, end_sec=45.0, text="클립 밖2"),
    ]
    out = segments_in_range(segments, start_sec=10.0, end_sec=30.0)
    assert out == [segments[1], segments[2]]
