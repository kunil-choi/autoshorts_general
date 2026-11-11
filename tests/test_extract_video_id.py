import pytest

from autoshorts.sources.youtube_source import _extract_video_id


@pytest.mark.parametrize(
    "url_or_id",
    [
        "dQw4w9WgXcQ",
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=42s",
        "https://youtu.be/dQw4w9WgXcQ",
        "https://www.youtube.com/shorts/dQw4w9WgXcQ",
        "https://www.youtube.com/embed/dQw4w9WgXcQ",
    ],
)
def test_extract_video_id_recognizes_all_url_forms(url_or_id):
    assert _extract_video_id(url_or_id) == "dQw4w9WgXcQ"


def test_extract_video_id_rejects_unrecognized_input():
    with pytest.raises(ValueError):
        _extract_video_id("not a youtube url")
