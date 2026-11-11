"""Ask Claude to propose shorts candidates from a timestamped transcript.

범용 버전: 세 가지 선정 기준(티어)을 고를 수 있다 - "hook"(훅 위주),
"substantive"(핵심 정보 위주), "keyword"(사용자가 적은 키워드에 맞는 구간,
키워드를 적었을 때만 동작). Each candidate covers a single contiguous
[start_sec, end_sec] range at proposal time (wrapped in a one-item
``ranges`` list); stitching several non-adjacent ranges into one clip is a
worker action at step 6 (see clipbuild.py), not something the AI proposes.
"""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, replace

import anthropic

from .config import DEFAULT_TIER_COUNTS, MAX_TIER_COUNT, MODEL, ClipLengthPreset
from .transcript import Segment, format_for_prompt

SYSTEM_PROMPT_TEMPLATE = """\
너는 영상 콘텐츠에서 유튜브 쇼츠로 잘라 올리기 좋은 구간을 찾아내는 편집 보조야.

지금 요청하는 건 아래 조건에 맞는 구간 정확히 {count}개다.

좋은 후보의 기준:
- 길이는 반드시 {max_sec}초를 넘기면 안 된다 ({min_sec}~{max_sec}초를 목표로 한다). {max_sec}초
  안에서 문장이 자연스럽게 시작되고 끝나는 지점을 찾아라: 새로운 문장/화제가 막 시작하는 지점에서
  시작하고, 그 발언이 결론까지 자연스럽게 끝나는 지점에서 끝내라. 문장 중간에서 시작하거나
  끊기면 안 된다. 내용이 길어서 {max_sec}초 안에 못 담는다면 그 구간은 포기하고, 더 짧고 그
  자체로 완결된 다른 구간을 찾아라. 그 구간만 봐도 무슨 얘기인지 완전히 이해가 되어야 한다
  (앞뒤 맥락 설명 없이도 독립적으로 말이 될 것).
- 시작 지점과 끝 지점 모두 그 구간의 핵심 주제와 직접 관련된 발언이어야 한다. 시작을 이전
  화제의 꼬리(마무리 멘트, "아무튼", "그래서" 같은 전환어만 있는 줄)에서 잡거나, 끝을 다음
  화제로 넘어가는 도입부(다른 주제를 막 꺼내는 줄)에서 끊으면 안 된다 — 주제와 무관한 앞뒤
  내용이 섞이면 그 줄은 버리고, 핵심 주제에 해당하는 발언만으로 시작/끝 경계를 다시 잡아라.
- 시작/끝 시각은 반드시 주어진 대본의 타임스탬프 구간 경계와 일치시켜라 (타임스탬프 구간 중간 지점을
  임의로 잘라 쓰지 말 것).
{speaker_rule}
{selection_philosophy}
- 같은 주제를 반복하는 후보끼리는 피하고, 서로 다른 순간 {count}개를 고른다.
- thumbnail_text는 영상 상단 제목 배너에 들어갈 문구로, 정확히 두 줄을 줄바꿈(\n) 하나로 구분해서 써라.
  1번째 줄: 주제를 압축한 짧은 키워드/명사구 (5~10자, 흰색으로 표시됨)
  2번째 줄: 궁금증을 유발하는 질문형 또는 임팩트 있는 훅 문장 (8~14자, 강조색으로 표시됨)
  두 줄 다 자극적이되 실제 발언 내용과 어긋나지 않아야 한다. 예: "로봇택시\n취객은 누가 깨울까?"

주어지는 대본은 "[시작-끝] 텍스트" 형식의 타임스탬프 붙은 줄들이다. 이걸 그대로 참고해서 시작/끝 시각을 고를 것.
정확히 {count}개의 후보를 제시해라.
"""

# 화자 구분(앵커/패널) 대본용 규칙 - 인터뷰형 영상 전용. assumes an
# interview-style source (fixed host + guest) where the render crop stays
# on the guest's side of frame for the whole clip. Works as-is for that
# format; a source with a different visual format (e.g. no fixed
# host/guest) may need this rule adjusted or dropped per workspace.
_GUEST_CENTERED_RULE = """\
- 대본의 각 줄 앞에는 화자 역할이 [앵커](진행자: 짧게 질문하거나 맞장구치는 쪽) 또는
  [패널](출연자: 실제로 분석·설명·의견을 길게 이야기하는 쪽)로 표시되어 있다. 표시가 없는
  줄은 내용으로 판단해라. 렌더링 화면은 클립 내내 출연자 얼굴로 고정되고 진행자가 말할 때도
  화면은 바뀌지 않은 채 목소리만 나오므로, 이를 고려해서 아래 세 가지 패턴 중 하나로
  구간을 골라라 (섞어서 판단해도 된다):
  (1) 출연자 혼자 끊김 없이 말하는 구간([앵커] 줄이 전혀 없음). 길고 완결된 발언이 있다면
      이 패턴을 우선한다.
  (2) 출연자의 발언만으로는 무슨 내용인지 이해가 안 되는 경우, 그 발언을 이끌어낸 진행자의
      바로 앞 질문부터 포함한다 — 구간의 시작을 그 진행자 질문이 시작되는 지점으로 잡는다
      (단, 그 질문이 길면 포함하지 않는다는 훅 티어의 도입 제한이 우선한다).
  (3) 출연자가 길게 이야기하는 도중에 진행자가 짧게 맞장구를 치거나 중간 질문을 던지는
      정도라면 그 부분을 잘라내지 말고 그대로 포함한다.
  패널의 발언 도중에 시작하거나 끝내지 마라: 구간은 패널이 한 번에 이어서 말하는 발언
  (연속된 [패널] 줄)의 처음에서 시작해 그 발언이 끝나는 줄에서 끝나야 하고, [앵커] 줄로
  끝나서는 안 된다. 그 발언이 너무 길어 시간 안에 못 담으면 다른 구간을 골라라.
  다만 어느 패턴이든 구간 전체 발화 시간의 대부분은 출연자여야 한다 — 진행자가 여러
  문장에 걸쳐 길게 말하는 인터뷰 도입부나, 진행자 발언이 구간의 절반 이상을 차지하는
  구간은 고르지 마라.\
"""

# 화자 구분(앵커/패널)이 없는 일반 영상용 규칙. 인터뷰/방송이 아니라 강의, 브이로그,
# 발표, 대담 등 무엇이든 올 수 있으므로 "한 사람이 이어서 말하는 완결된 발언"을 기준으로 한다.
_GENERAL_SPEAKER_RULE = """\
- 대본에는 화자 구분이 없다. 영상 종류(강의, 발표, 대담, 브이로그 등)를 가정하지 말고 내용으로
  판단해라. 여러 사람이 번갈아 말하는 영상이라면 질문과 답변이 한 구간 안에서 짝이 맞게
  (질문만 있고 답이 잘리거나, 답만 있고 질문이 없어 이해가 안 되는 일이 없게) 구간을 잡아라.
  한 사람이 길게 이어서 말하는 발언은 그 발언이 시작하는 줄에서 시작해 끝나는 줄에서 끝내고,
  발언 도중에 시작하거나 끝내지 마라.\
"""

_HOOK_VIRALITY_PHILOSOPHY = """\
- 훅(궁금증을 유발하거나 임팩트 있는 발언)은 구간 맨 앞에 있어야 한다. 영상 상단 제목
  배너(thumbnail_text)가 시청자를 어느 정도 붙잡아주긴 하지만, 실제 쇼츠 시청자의 절반
  이상은 처음 몇 초 안에 이탈한다는 게 알려져 있으므로("훅"이 늦을수록 이탈 위험이 커진다), 구간의 첫 줄 자체가
  훅이거나, 늦어도 시작 후 5초 안에는 훅이 나와야 한다. 훅이 5초 뒤에 나오는 구간은 고르지
  않고, 같은 조건이면 훅이 더 앞쪽에 있는 구간을 우선한다.
- 구간 앞에 긴 도입(인사, 배경 설명, 길게 이어지는 진행자 질문)을 붙이지 마라. 진행자의 질문은
  한 문장으로 짧을 때만 포함하고(대략 5초 이내), 질문이 길면 포함하지 말고 출연자의 결론/핵심
  발언이 시작되는 줄에서 바로 시작해라. 이렇게 잡은 구간이 그 발언만으로 이해가 어렵다면 그
  구간은 포기하고 다른 구간을 골라라.
- 시청자의 시선을 끌 만한 요소를 우선한다: 구체적인 수치, 시청자 개인의 삶(돈·건강·일상)에
  직접 영향을 준다고 느껴지는 내용, 의견 충돌이나 예상을 뒤엎는 반전, 실용적인 결론. 이런
  요소가 있는 구간을 단정적이거나 논쟁적이거나 의외성 있는 발언, 웃긴 순간과 함께 우선한다.\
"""

_SUBSTANCE_PHILOSOPHY = """\
- 이 티어는 조회수를 노리는 "훅" 위주로 고르지 않는다. 강력한 훅(궁금증 유발, 반전, 웃긴
  순간)이 없어도 괜찮다 - 대신 영상 전체 내용을 통틀어 봤을 때 실질적으로 중요하다고
  판단되는 발언, 그 영상의 핵심 주제·논지와 직접 관련된 명확한 멘트를 우선한다. "이
  영상에서 꼭 짚고 넘어가야 할 내용"에 해당하는 구간(핵심 데이터·수치, 결론적인 판단,
  실질적인 영향에 대한 설명 등)을 골라라.
- 훅이 없다고 지루해도 된다는 뜻은 아니다 - 문맥 설명 없이 그 구간만 들어도 무슨 얘기인지,
  왜 중요한지 명확하게 전달되어야 한다. 모호하거나 앞뒤 맥락이 있어야만 이해되는 발언은
  피한다.\
"""


def _keyword_philosophy(keyword: str, count: int) -> str:
    return f"""\
- 이 티어는 자유롭게 고르는 게 아니라, 사용자가 적은 다음 키워드/주제/조건에 맞는
  구간만 찾는다: "{keyword}"
- 대본 전체에서 이 키워드와 직접 관련된 발언을 찾아라. 단어가 똑같이 나오지 않아도 같은
  뜻으로 말한 부분이면 포함한다. 키워드와 관련이 있어 보여도 그 발언이
  구간 전체 발화 시간의 대부분을 차지하지 않으면 후보에서 제외한다.
- 이 키워드에 맞는 구간이 {count}개보다 적게 나와도 괜찮다 - 억지로 채우지 말고 실제로
  키워드에 부합하는 구간만 제시해라.\
"""


TIER_LABELS = {
    "hook": "훅 위주",
    "substantive": "핵심 내용 위주",
    "keyword": "키워드",
}


@dataclass
class ClipRange:
    start_sec: float
    end_sec: float


@dataclass
class Candidate:
    ranges: list[ClipRange]
    title: str
    summary: str
    thumbnail_text: str
    reason: str
    tier: str = ""


def _sec_to_hms(sec: float) -> str:
    m, s = divmod(int(round(sec)), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _hms_to_sec(hms: str) -> float:
    parts = [float(p) for p in hms.strip().split(":")]
    while len(parts) < 3:
        parts.insert(0, 0.0)
    h, m, s = parts
    return h * 3600 + m * 60 + s


def _candidate_tool() -> dict:
    # 티어별 개수와 무관하게 같은 스키마를 써야 티어끼리 프롬프트 캐시(tools -> 대본)가
    # 공유된다. 실제 개수 제한은 시스템 프롬프트("정확히 N개")와 호출 쪽 자르기로 한다.
    count = MAX_TIER_COUNT
    return {
        "name": "propose_shorts_candidates",
        "description": f"영상에서 쇼츠로 만들기 좋은 최대 {count}개 구간을 제안한다.",
        "input_schema": {
            "type": "object",
            "properties": {
                "candidates": {
                    "type": "array",
                    "maxItems": count,
                    "items": {
                        "type": "object",
                        "properties": {
                            "start_hms": {
                                "type": "string",
                                "description": (
                                    "구간 시작 시각, MM:SS 또는 HH:MM:SS. 반드시 새 문장/화제가 "
                                    "시작하는 대본 타임스탬프 경계와 일치시킬 것."
                                ),
                            },
                            "end_hms": {
                                "type": "string",
                                "description": (
                                    "구간 끝 시각, MM:SS 또는 HH:MM:SS. 반드시 그 발언이 자연스럽게 "
                                    "끝나는 대본 타임스탬프 경계와 일치시킬 것 (문장 중간에 끊지 말 것)."
                                ),
                            },
                            "title": {"type": "string", "description": "짧은 내부 제목"},
                            "summary": {
                                "type": "string",
                                "description": "이 구간에서 실제로 어떤 이야기가 나오는지 2~3문장 요약",
                            },
                            "thumbnail_text": {
                                "type": "string",
                                "description": (
                                    "영상 상단 제목 배너 문구. 정확히 두 줄, 그 사이를 \\n 하나로 구분: "
                                    "1번째 줄은 짧은 주제 키워드(5~10자), 2번째 줄은 훅이 되는 질문/임팩트 문장"
                                    "(8~14자). 예: '로봇택시\\n취객은 누가 깨울까?'"
                                ),
                            },
                            "reason": {
                                "type": "string",
                                "description": "왜 이 구간이 쇼츠로 매력적인지",
                            },
                        },
                        "required": [
                            "start_hms",
                            "end_hms",
                            "title",
                            "summary",
                            "thumbnail_text",
                            "reason",
                        ],
                    },
                }
            },
            "required": ["candidates"],
        },
    }


def _propose_candidates_for_tier(
    client: anthropic.Anthropic,
    transcript_text: str,
    clip_length: ClipLengthPreset,
    video_title: str,
    tier: str,
    count: int,
    selection_philosophy: str,
    cache_transcript: bool = False,
    speaker_rule: str = _GENERAL_SPEAKER_RULE,
) -> list[Candidate]:
    min_sec, max_sec = clip_length.min_sec, clip_length.max_sec
    instructions = SYSTEM_PROMPT_TEMPLATE.format(
        count=count,
        min_sec=min_sec,
        max_sec=max_sec,
        speaker_rule=speaker_rule,
        selection_philosophy=selection_philosophy,
    )
    # The transcript goes first, ahead of the tier-specific instructions, so
    # every tier call for the same video starts with the same bytes (tools ->
    # transcript). With cache_transcript the first tier writes that prefix to
    # the prompt cache and later tiers read it at a tenth of the input price
    # instead of paying for the whole transcript again.
    transcript_block = {"type": "text", "text": f"영상 제목: {video_title}\n\n대본:\n{transcript_text}"}
    if cache_transcript:
        transcript_block["cache_control"] = {"type": "ephemeral"}
    message = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        system=[transcript_block, {"type": "text", "text": instructions}],
        tools=[_candidate_tool()],
        tool_choice={"type": "tool", "name": "propose_shorts_candidates"},
        messages=[
            {
                "role": "user",
                "content": "위 영상 제목과 대본에서 조건에 맞는 쇼츠 구간을 골라 제시해라.",
            }
        ],
    )

    tool_use = next(b for b in message.content if b.type == "tool_use")
    raw_candidates = tool_use.input["candidates"][:count]

    candidates: list[Candidate] = []
    for c in raw_candidates:
        # tool_choice forces this schema, but don't let one malformed entry
        # (seen in practice: a candidate coming back as something other
        # than the expected object) take down the whole tier's results -
        # skip it and keep whatever candidates did parse correctly.
        if not isinstance(c, dict):
            print(f"[analyze] skipping malformed {tier} candidate (not an object): {c!r}")
            continue
        try:
            start = _hms_to_sec(c["start_hms"])
            end = _hms_to_sec(c["end_hms"])
            if end <= start:
                continue
            candidates.append(
                Candidate(
                    ranges=[ClipRange(start_sec=start, end_sec=end)],
                    title=c["title"],
                    summary=c["summary"],
                    thumbnail_text=c["thumbnail_text"],
                    reason=c["reason"],
                    tier=tier,
                )
            )
        except (KeyError, ValueError) as e:
            print(f"[analyze] skipping malformed {tier} candidate ({e}): {c!r}")
            continue
    return candidates


# format_for_prompt() truncates every segment boundary shown to the model
# down to a whole second (int(sec), not even rounded), since that's plenty
# precise for a human/model deciding *which* segment to start or end on.
# But two adjacent segments can truncate to the *same* displayed timestamp -
# e.g. one segment ending at 1764.669s and the next starting at 1764.679s
# both show as "29:24" - so when the model's chosen "29:24" is parsed back
# via _hms_to_sec, the result (exactly 1764.0) can land a fraction of a
# second inside the earlier segment instead of exactly on the boundary the
# model actually meant. That stray fraction is enough for the rendered
# clip's audio and captions to start or end on a fragment of the wrong
# sentence. _SEGMENT_SNAP_TOLERANCE bounds how far a parsed timestamp can be
# from the nearest real segment boundary and still be considered "meant to
# be exactly there" - it comfortably covers the max <1s truncation error
# without being wide enough to snap onto some unrelated, distant segment.
_SEGMENT_SNAP_TOLERANCE = 1.5


def _closest_boundary(target: float, boundaries: list[float]) -> float:
    if not boundaries:
        return target
    closest = min(boundaries, key=lambda b: abs(b - target))
    return closest if abs(closest - target) <= _SEGMENT_SNAP_TOLERANCE else target


def _snap_to_segment_boundaries(
    candidates: list[Candidate], segments: list[Segment]
) -> list[Candidate]:
    """Correct candidate range boundaries for format_for_prompt's
    whole-second display precision (see _SEGMENT_SNAP_TOLERANCE) by
    snapping each to the nearest real segment boundary, so a clip never
    starts/ends a fraction of a second inside the wrong segment."""
    starts = [s.start_sec for s in segments]
    ends = [s.end_sec for s in segments]
    snapped = []
    for c in candidates:
        new_ranges = [
            replace(
                r,
                start_sec=_closest_boundary(r.start_sec, starts),
                end_sec=_closest_boundary(r.end_sec, ends),
            )
            for r in c.ranges
        ]
        snapped.append(replace(c, ranges=new_ranges))
    return snapped


# 자동 생성의 기본 구성: 훅성 3개 + 주제 밀착·완결성 2개 = 5개


def _avoid_text(picked: list[Candidate]) -> str:
    """Tells a later tier's call which moments earlier tiers already took."""
    if not picked:
        return ""
    lines = "\n".join(
        f"  - {_sec_to_hms(c.ranges[0].start_sec)} ~ {_sec_to_hms(c.ranges[-1].end_sec)} ({c.title})"
        for c in picked
    )
    return (
        "\n- 아래 구간은 이미 다른 기준으로 선정됐다. 이 구간과 시간이 겹치거나 같은 순간을 "
        "다루는 후보는 고르지 마라:\n" + lines
    )


def propose_candidates(
    segments: list[Segment],
    clip_length: ClipLengthPreset,
    tiers: list[str],
    video_title: str = "",
    keyword: str | None = None,
    client: anthropic.Anthropic | None = None,
    counts: dict[str, int] | None = None,
) -> list[Candidate]:
    """tiers is a subset of ("hook", "substantive", "keyword"). Each tier asks
    for `counts[tier]` candidates (기본값은 config.DEFAULT_TIER_COUNTS).
    Later tiers are told which moments earlier tiers already picked so the
    results don't overlap. "keyword" only runs when `keyword` is non-empty;
    it may return fewer candidates than asked (or none) since it's
    constrained to an actual match rather than a free pick."""
    client = client or anthropic.Anthropic()
    transcript_text = format_for_prompt(segments)
    counts = {**DEFAULT_TIER_COUNTS, **(counts or {})}
    keyword = (keyword or "").strip()
    # 화자 구분(앵커/패널)이 붙은 대본이면 기존 인터뷰 규칙, 아니면 일반 규칙
    speaker_rule = _GUEST_CENTERED_RULE if any(s.speaker for s in segments) else _GENERAL_SPEAKER_RULE

    tier_specs: list[tuple[str, int, str]] = []
    if "hook" in tiers and counts["hook"] > 0:
        tier_specs.append(("hook", counts["hook"], _HOOK_VIRALITY_PHILOSOPHY))
    if "substantive" in tiers and counts["substantive"] > 0:
        tier_specs.append(("substantive", counts["substantive"], _SUBSTANCE_PHILOSOPHY))
    if "keyword" in tiers and keyword and counts["keyword"] > 0:
        tier_specs.append(("keyword", counts["keyword"], _keyword_philosophy(keyword, counts["keyword"])))

    # A cache write costs 1.25x the input price, so it only pays off when a
    # second tier will read it back - a single-tier run stays uncached.
    cache_transcript = len(tier_specs) > 1
    candidates: list[Candidate] = []
    for tier, count, selection_philosophy in tier_specs:
        candidates.extend(
            _propose_candidates_for_tier(
                client, transcript_text, clip_length, video_title, tier, count,
                selection_philosophy + _avoid_text(candidates),
                cache_transcript=cache_transcript, speaker_rule=speaker_rule,
            )
        )
    return _snap_to_segment_boundaries(candidates, segments)


_REFINE_CONTEXT_LINES = 6
_REFINE_MIN_SEC = 10

REFINE_TOOL = {
    "name": "set_clip_boundaries",
    "description": "쇼츠 구간의 시작 줄과 끝 줄을 문장이 온전한 위치로 확정한다.",
    "input_schema": {
        "type": "object",
        "properties": {
            "start_line": {"type": "integer", "description": "구간이 시작할 줄 번호(대본 앞의 [번호])"},
            "end_line": {"type": "integer", "description": "구간이 끝날 줄 번호(포함)"},
        },
        "required": ["start_line", "end_line"],
    },
}


def _segment_index_range(c: Candidate, segments: list[Segment]) -> tuple[int, int] | None:
    start, end = c.ranges[0].start_sec, c.ranges[-1].end_sec
    idxs = [i for i, s in enumerate(segments) if s.end_sec > start + 0.1 and s.start_sec < end - 0.1]
    return (idxs[0], idxs[-1]) if idxs else None


def refine_boundaries(
    candidates: list[Candidate],
    segments: list[Segment],
    clip_length: ClipLengthPreset,
    client: anthropic.Anthropic | None = None,
) -> list[Candidate]:
    """Second pass over single-range candidates: show Claude each candidate's
    transcript lines plus a few lines on both sides and have it settle the
    first/last line so the clip starts at the beginning of a sentence and
    ends once the speaker has finished the thought. The initial proposal
    only picks cue boundaries, and caption cues are fragments, so it often
    starts mid-sentence or stops before the speaker is done.

    Best-effort: any failure, or an answer that is out of bounds / too
    long / too short, keeps that candidate's original range.
    """
    client = client or anthropic.Anthropic()
    refined: list[Candidate] = []
    for c in candidates:
        if len(c.ranges) != 1:
            refined.append(c)
            continue
        try:
            refined.append(_refine_one(client, c, segments, clip_length))
        except Exception as e:  # noqa: BLE001 - keep the original range rather than lose the candidate
            print(f"[analyze] 구간 경계 보정 실패, 원래 구간 유지: {e}")
            refined.append(c)
    return refined


def _refine_one(
    client: anthropic.Anthropic, c: Candidate, segments: list[Segment], clip_length: ClipLengthPreset,
) -> Candidate:
    span = _segment_index_range(c, segments)
    if span is None:
        return c
    first, last = span
    lo = max(0, first - _REFINE_CONTEXT_LINES)
    hi = min(len(segments) - 1, last + _REFINE_CONTEXT_LINES)
    lines = "\n".join(
        f"[{i}] ({_sec_to_hms(segments[i].start_sec)}-{_sec_to_hms(segments[i].end_sec)}) "
        + (f"[{segments[i].speaker}] " if segments[i].speaker else "")
        + segments[i].text
        for i in range(lo, hi + 1)
    )
    message = client.messages.create(
        model=MODEL,
        max_tokens=512,
        system=(
            "너는 쇼츠 편집 보조다. 아래 대본은 자막 조각이라 문장 중간에서 끊겨 있을 수 있다. "
            f"현재 후보 구간은 {first}번 줄부터 {last}번 줄까지다. 앞뒤 줄을 참고해서 구간을 다음처럼 확정해라:\n"
            "- 시작 줄은 새 문장·새 발언이 시작되는 줄이어야 한다. 앞 줄에서 이어지는 말의 중간(조사·어미·"
            "접속어로만 시작하거나, 앞 내용을 받는 말)이면 앞으로 더 가서 그 문장의 시작 줄을 잡아라.\n"
            "- 끝 줄은 화자가 문장과 생각을 끝맺는 줄이어야 한다. 말이 이어지는 중이면 뒤로 더 가서 "
            "끝맺는 줄까지 포함해라. 다음 화제를 꺼내는 줄은 포함하지 마라.\n"
            "- 줄 앞의 [앵커]는 진행자, [패널]은 주된 내용을 말하는 출연자다. 끝 줄은 패널이 그 발언을 "
            "다 마친 줄이어야 한다: 끝 줄 다음 줄도 [패널]이라면 발언이 계속되는 중이니 그 발언이 끝날 "
            "때까지 뒤로 늘려라. [앵커] 줄(질문·마무리 멘트)로 끝내지 마라. 시작 줄도 패널 발언의 "
            "중간이 아니라 그 발언의 첫 줄이어야 한다(앞 줄도 [패널]이면 발언의 첫 줄까지 앞으로 가라). "
            "단, 패널의 발언을 이끌어낸 [앵커]의 바로 앞 질문 줄에서 시작하는 것은 괜찮다.\n"
            "- 구간 전체의 핵심 내용은 유지하고, 주제와 무관한 앞뒤 줄은 넣지 마라.\n"
            f"- 전체 길이는 {clip_length.max_sec}초를 넘으면 안 된다. 줄 번호는 {lo}~{hi} 안에서만 고른다."
        ),
        tools=[REFINE_TOOL],
        tool_choice={"type": "tool", "name": "set_clip_boundaries"},
        messages=[{"role": "user", "content": f"후보 제목: {c.title}\n요약: {c.summary}\n\n대본:\n{lines}"}],
    )
    data = next(b for b in message.content if b.type == "tool_use").input
    new_first, new_last = int(data["start_line"]), int(data["end_line"])
    if not (lo <= new_first <= new_last <= hi):
        return c
    new_start, new_end = segments[new_first].start_sec, segments[new_last].end_sec
    duration = new_end - new_start
    if duration > clip_length.max_sec or duration < _REFINE_MIN_SEC:
        return c
    new_first, new_last = _align_to_panel_turns(new_first, new_last, segments, clip_length)
    new_start, new_end = segments[new_first].start_sec, segments[new_last].end_sec
    return replace(c, ranges=[ClipRange(start_sec=new_start, end_sec=new_end)])


_SENTENCE_END_CHARS = tuple(".?!…다요죠까네")


def _looks_like_sentence_end(text: str) -> bool:
    t = text.strip().rstrip("\"'”’)」")
    return bool(t) and t.endswith(_SENTENCE_END_CHARS)


def _align_to_panel_turns(
    first: int, last: int, segments: list[Segment], clip_length: ClipLengthPreset
) -> tuple[int, int]:
    """Deterministic safety net on top of Claude's boundary pick, using the
    앵커/패널 labels (analyze.label_speakers): a clip should not start or end
    in the middle of a panelist's speech.

    - end: ends on a 앵커 line -> trimmed back to the panelist's last line;
      ends while the panelist keeps talking -> extended to the end of that
      turn if it still fits max_sec, else cut at the last sentence ending
      that fits.
    - start: begins mid-turn of the panelist -> moved back to the turn's first
      line if it still fits max_sec.
    No labels, or any adjustment that would break the length limits, keeps
    the input as is.
    """
    if not any(s.speaker for s in segments):
        return first, last
    n = len(segments)
    max_sec = clip_length.max_sec

    def span(a: int, b: int) -> float:
        return segments[b].end_sec - segments[a].start_sec

    # --- end ---
    new_last = last
    if segments[new_last].speaker == "앵커":
        j = new_last
        while j > first and segments[j].speaker == "앵커":
            j -= 1
        if segments[j].speaker == "패널" and span(first, j) >= _REFINE_MIN_SEC:
            new_last = j
    if segments[new_last].speaker == "패널" and new_last + 1 < n and segments[new_last + 1].speaker == "패널":
        turn_end = new_last
        while turn_end + 1 < n and segments[turn_end + 1].speaker == "패널":
            turn_end += 1
        if span(first, turn_end) <= max_sec:
            new_last = turn_end
        else:
            fit = [
                k for k in range(new_last, turn_end + 1)
                if span(first, k) <= max_sec and _looks_like_sentence_end(segments[k].text)
            ]
            if fit:
                new_last = fit[-1]

    # --- start ---
    new_first = first
    if segments[first].speaker == "패널" and first > 0 and segments[first - 1].speaker == "패널":
        turn_start = first
        while turn_start > 0 and segments[turn_start - 1].speaker == "패널":
            turn_start -= 1
        if span(turn_start, new_last) <= max_sec:
            new_first = turn_start

    return new_first, new_last


def candidate_to_dict(c: Candidate) -> dict:
    d = asdict(c)
    d["tier_label"] = TIER_LABELS.get(c.tier, c.tier)
    return d


CAPTION_CORRECTION_TOOL = {
    "name": "correct_captions",
    "description": "제공된 자막 문장들에서 음성인식 오류로 보이는 부분만 문맥에 맞게 교정한다.",
    "input_schema": {
        "type": "object",
        "properties": {
            "corrected": {
                "type": "array",
                "description": "입력과 정확히 같은 개수·순서로 교정된 문장을 반환한다.",
                "items": {"type": "string"},
            }
        },
        "required": ["corrected"],
    },
}


def correct_caption_errors(texts: list[str], client: anthropic.Anthropic | None = None) -> list[str]:
    """Best-effort: ask Claude to fix likely speech-recognition mishearings in
    a batch of caption lines, preserving meaning/count/order. Falls back to
    the original texts on any failure (wrong count, API error, etc.) - this
    is a polish step, never worth breaking the pipeline over.
    """
    if not texts:
        return texts
    client = client or anthropic.Anthropic()
    numbered = "\n".join(f"{i}: {t}" for i, t in enumerate(texts))
    try:
        message = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=(
                "다음은 영상을 자동 음성인식(STT)으로 받아쓴 자막 문장들이다. "
                "발음이 비슷한 다른 단어로 잘못 인식되어 문맥상 말이 안 되는 부분이 "
                "있을 수 있다. 그런 오류만 문맥에 맞게 자연스러운 한국어로 고쳐라. "
                "이미 맞는 문장은 절대 건드리지 말고, 뜻이나 어투를 임의로 바꾸지 마라 "
                "(오타/오인식 교정이지 문장 재작성이 아니다). 입력과 정확히 같은 "
                "개수·순서로 반환해야 한다."
            ),
            tools=[CAPTION_CORRECTION_TOOL],
            tool_choice={"type": "tool", "name": "correct_captions"},
            messages=[{"role": "user", "content": numbered}],
        )
        tool_use = next(b for b in message.content if b.type == "tool_use")
        corrected = tool_use.input["corrected"]
        if len(corrected) != len(texts):
            return texts
        return corrected
    except Exception:
        return texts


GUEST_INFO_TOOL = {
    "name": "extract_guest_info",
    "description": "영상 제목/설명에서 그 영상에 나온 출연자(게스트)의 이름·직책·소속을 찾아낸다.",
    "input_schema": {
        "type": "object",
        "properties": {
            "found": {
                "type": "boolean",
                "description": "제목/설명에서 출연자 정보를 확인할 수 있으면 true, 근거가 없으면 false.",
            },
            "name": {"type": "string", "description": "출연자 이름"},
            "title": {"type": "string", "description": "직위/직책 (예: 변호사, 상석본부장). 없으면 빈 문자열."},
            "org": {"type": "string", "description": "소속 기관/회사. 없으면 빈 문자열."},
        },
        "required": ["found"],
    },
}


def extract_guest_info(
    video_title: str, description: str | None, client: anthropic.Anthropic | None = None
) -> str:
    """Best-effort: ask Claude to find the on-screen guest's name/title/org
    from the video title+description, formatted as a single display line
    ("이름 직책 / 소속", e.g. "김은비 변호사 / 손해보험협회") to prefill (never
    force) the guest-name field in step 8. Empty string on any failure or
    when nothing could be confidently determined (e.g. a solo-host episode,
    or a description with no guest credit).
    """
    if not description:
        return ""
    client = client or anthropic.Anthropic()
    try:
        message = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=(
                "영상의 제목과 설명을 보고, 그 영상에 초대되어 나온 출연자(게스트)의 "
                "이름, 직위/직책, 소속 기관을 찾아내라. 그 채널을 진행하는 고정 진행자"
                "(MC)는 대상이 아니다 - 외부에서 초대된 게스트만 찾는다. 제목/설명에 "
                "명확한 근거가 없으면 found=false로 답하고 다른 필드는 추측해서 채우지 마라."
            ),
            tools=[GUEST_INFO_TOOL],
            tool_choice={"type": "tool", "name": "extract_guest_info"},
            messages=[
                {"role": "user", "content": f"제목: {video_title}\n\n설명:\n{description}"}
            ],
        )
        tool_use = next(b for b in message.content if b.type == "tool_use")
        data = tool_use.input
        if not data.get("found"):
            return ""
        name = (data.get("name") or "").strip()
        if not name:
            return ""
        title = (data.get("title") or "").strip()
        org = (data.get("org") or "").strip()
        label = f"{name} {title}".strip() if title else name
        return f"{label} / {org}" if org else label
    except Exception:
        return ""


SPEAKER_LABEL_TOOL = {
    "name": "label_speaker_turns",
    "description": "대본을 화자가 바뀌는 구간(turn)별로 나누고, 각 구간이 앵커(진행자)인지 패널(출연자)인지 표시한다.",
    "input_schema": {
        "type": "object",
        "properties": {
            "turns": {
                "type": "array",
                "description": (
                    "화자가 이어지는 연속 구간들. 대본 줄 번호 기준으로, 같은 화자가 "
                    "계속 말하는 동안은 하나의 turn으로 묶는다 (매 줄마다 나누지 않는다)."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "start_idx": {"type": "integer", "description": "이 화자 구간이 시작하는 줄 번호 (0부터 시작)"},
                        "end_idx": {"type": "integer", "description": "이 화자 구간이 끝나는 줄 번호 (포함)"},
                        "role": {
                            "type": "string",
                            "enum": ["앵커", "패널"],
                            "description": "이 구간에서 말하는 사람의 역할 - 앵커(진행자, 짧게 질문/진행) 또는 패널(출연자, 실질적인 내용을 설명)",
                        },
                    },
                    "required": ["start_idx", "end_idx", "role"],
                },
            }
        },
        "required": ["turns"],
    },
}


def label_speakers(segments: list[Segment], client: anthropic.Anthropic | None = None) -> list[str]:
    """Best-effort per-segment speaker role ("앵커" or "패널"), guessed by
    Claude from conversational pattern (who asks short questions vs who
    gives substantive answers) - there is no real audio diarization here,
    so this can be wrong, especially for solo narration or ambiguous
    back-and-forth. Returns a list the same length as segments, with ""
    for any segment the model didn't cover or on any failure - never
    breaks the pipeline over this, it's a display nicety in the review UI.
    """
    if not segments:
        return []
    client = client or anthropic.Anthropic()
    numbered = "\n".join(f"{i}: {s.text}" for i, s in enumerate(segments))
    labels = [""] * len(segments)
    try:
        message = client.messages.create(
            model=MODEL,
            max_tokens=8192,
            system=(
                "다음은 방송 대본을 줄 번호와 함께 나열한 것이다. 이 방송에는 진행을 맡은 "
                "앵커(짧게 질문하거나 맞장구치며 진행하는 쪽)와, 실제로 분석·설명·의견을 "
                "길게 이야기하는 출연자(패널)가 있다. 대본 내용을 보고 각 줄이 누구의 "
                "발언인지 판단해서, 같은 화자가 이어지는 구간(turn)별로 묶어 표시해라. "
                "화자가 단 한 명뿐이라 구분이 불가능하면 turns를 빈 배열로 반환해라."
            ),
            tools=[SPEAKER_LABEL_TOOL],
            tool_choice={"type": "tool", "name": "label_speaker_turns"},
            messages=[{"role": "user", "content": numbered}],
        )
        tool_use = next(b for b in message.content if b.type == "tool_use")
        turns = tool_use.input.get("turns", [])
        for turn in turns:
            if not isinstance(turn, dict):
                continue
            role = turn.get("role")
            if role not in ("앵커", "패널"):
                continue
            try:
                start = max(0, int(turn["start_idx"]))
                end = min(len(segments) - 1, int(turn["end_idx"]))
            except (KeyError, TypeError, ValueError):
                continue
            for i in range(start, end + 1):
                labels[i] = role
        return labels
    except Exception:
        return [""] * len(segments)
