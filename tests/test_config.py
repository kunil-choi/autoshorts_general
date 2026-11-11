import pytest

from autoshorts.config import (
    CUSTOM_MAX_SEC, CUSTOM_MIN_SEC, DEFAULT_CLIP_LENGTH_PRESET, get_clip_length_preset, make_custom_key,
)


def test_presets_still_resolve():
    assert get_clip_length_preset("30sec").max_sec == 30
    assert get_clip_length_preset("3min").max_sec == 180
    assert get_clip_length_preset(DEFAULT_CLIP_LENGTH_PRESET).max_sec == 60


def test_custom_length_key_gives_a_preset_with_that_max():
    p = get_clip_length_preset(make_custom_key(90))
    assert (p.key, p.max_sec, p.min_sec) == ("custom:90", 90, 10)
    assert get_clip_length_preset(make_custom_key(CUSTOM_MIN_SEC)).max_sec == CUSTOM_MIN_SEC


@pytest.mark.parametrize("key", ["custom:5", f"custom:{CUSTOM_MAX_SEC + 1}", "custom:abc", "custom:", "9min", ""])
def test_bad_length_keys_are_rejected(key):
    with pytest.raises(ValueError):
        get_clip_length_preset(key)
