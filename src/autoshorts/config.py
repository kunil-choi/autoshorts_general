"""App-wide settings.

범용 버전: 특정 방송/프로그램에 묶이지 않는다. 입력은 유튜브 링크 또는
영상 파일, 후보 선정 기준은 훅 / 핵심 정보 / 사용자 키워드 3가지,
길이는 프리셋(30초~3분) 또는 사용자 지정(초 단위)이다.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ClipLengthPreset:
    key: str
    label: str
    min_sec: int
    max_sec: int


CLIP_LENGTH_PRESETS: tuple[ClipLengthPreset, ...] = (
    ClipLengthPreset("30sec", "30초 이내", 10, 30),
    ClipLengthPreset("45sec", "45초 이내", 10, 45),
    ClipLengthPreset("1min", "1분 이내", 10, 60),
    ClipLengthPreset("2min", "2분 이내", 10, 120),
    ClipLengthPreset("3min", "3분 이내", 10, 180),
)

DEFAULT_CLIP_LENGTH_PRESET = "1min"

# 사용자 지정 길이: 키는 "custom:<초>" 형태("custom:90" = 90초 이내).
# 유튜브 쇼츠는 최대 3분이라 상한을 180초로 두고, 너무 짧으면 후보 선정이
# 의미가 없어서 하한은 15초로 둔다.
CUSTOM_KEY_PREFIX = "custom:"
CUSTOM_MIN_SEC = 15
CUSTOM_MAX_SEC = 180


def make_custom_key(seconds: int) -> str:
    return f"{CUSTOM_KEY_PREFIX}{int(seconds)}"


def get_clip_length_preset(key: str) -> ClipLengthPreset:
    """프리셋 키 또는 "custom:<초>" 키를 길이 설정으로 바꾼다."""
    key = (key or "").strip()
    for preset in CLIP_LENGTH_PRESETS:
        if preset.key == key:
            return preset
    if key.startswith(CUSTOM_KEY_PREFIX):
        try:
            seconds = int(key[len(CUSTOM_KEY_PREFIX):])
        except ValueError:
            raise ValueError(f"사용자 지정 길이가 숫자가 아닙니다: '{key}'") from None
        if not (CUSTOM_MIN_SEC <= seconds <= CUSTOM_MAX_SEC):
            raise ValueError(
                f"사용자 지정 길이는 {CUSTOM_MIN_SEC}~{CUSTOM_MAX_SEC}초 사이여야 합니다 (입력: {seconds}초)."
            )
        return ClipLengthPreset(key, f"{seconds}초 이내", min(10, seconds), seconds)
    raise ValueError(f"unknown clip length preset '{key}'")


# 후보 선정 기준(티어)별 기본 개수. hook/substantive는 항상 고를 수 있고,
# keyword는 사용자가 키워드를 적었을 때만 동작한다.
DEFAULT_TIER_COUNTS = {"hook": 3, "substantive": 2, "keyword": 2}
MAX_TIER_COUNT = 5


# .env ships AUTOSHORTS_MODEL= blank as a template - os.environ.get's
# default only kicks in when the key is *absent*, not when python-dotenv
# has set it to an empty string, so `or` (not the two-arg .get form) is
# what actually makes a blank .env line fall back to this default.
MODEL = os.environ.get("AUTOSHORTS_MODEL") or "claude-sonnet-5"
