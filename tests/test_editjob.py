from datetime import datetime, timedelta, timezone

import pytest

from autoshorts import editjob
from autoshorts.analyze import Candidate, ClipRange
from autoshorts.transcript import Segment


def _segs(n=12, step=5.0):
    # each line overlaps the next by 1s, like rolling captions
    return [Segment(i * step, i * step + step + 1, f"줄 {i}", "패널") for i in range(n)]


def _cand(start, end, tier="hook"):
    return Candidate(ranges=[ClipRange(start, end)], title="제목", summary="요약",
                     thumbnail_text="첫줄\n둘째줄", reason="r", tier=tier)


def _block(segs, first=2, last=9):
    return editjob.build_blocks([_cand(segs[first].start_sec, segs[last].end_sec)], segs,
                                guest_label="g", shoot_date="d", date_kind="방송")[0]


def test_build_blocks_maps_candidate_to_segment_span():
    segs = _segs()
    b = _block(segs)
    assert (b["first"], b["last"]) == (2, 9)
    assert b["title"] == "첫줄 둘째줄" and b["thumbnail_text"] == "첫줄\n둘째줄"
    assert b["tier_label"]


def test_block_ranges_without_deletion_is_one_range():
    segs = _segs()
    ranges, bounds = editjob.block_ranges(_block(segs), segs)
    assert [(r.start_sec, r.end_sec) for r in ranges] == [(10.0, 51.0)]
    assert bounds == [(None, None)]


def test_deleting_middle_lines_splits_into_two_ranges_clamped_to_the_cut():
    segs = _segs()
    b = _block(segs)
    b["deleted"] = [5, 6]
    ranges, bounds = editjob.block_ranges(b, segs)
    assert len(ranges) == 2
    # range before the cut ends where deleted line 5 starts (25s), not at line 4's end (26s)
    assert ranges[0].end_sec == 25.0 and bounds[0][1] == 25.0
    # range after the cut starts where deleted line 6 ends (36s) - line 7 itself starts at 35s
    assert ranges[1].start_sec == 36.0 and bounds[1][0] == 36.0


def test_deleting_edge_lines_trims_the_clip():
    segs = _segs()
    b = _block(segs)
    b["deleted"] = [2, 9]
    ranges, _ = editjob.block_ranges(b, segs)
    assert len(ranges) == 1
    assert ranges[0].start_sec == 16.0  # end of deleted line 2 (10+6)
    assert ranges[0].end_sec == 45.0    # start of deleted line 9


def test_cue_segments_only_kept_lines_with_edits():
    segs = _segs()
    b = _block(segs)
    b["deleted"] = [3]
    b["texts"] = {"4": "고친 줄"}
    out = editjob.block_cue_segments(b, segs)
    assert [s.text for s in out][:3] == ["줄 2", "고친 줄", "줄 5"]
    assert "줄 3" not in [s.text for s in out]


def test_apply_block_edit_validates_and_drops_noop_text_edits():
    segs = _segs()
    b = _block(segs)
    editjob.apply_block_edit(b, segs, {
        "deleted": [4, 99], "texts": {"3": "줄 3", "5": "수정\n됨", "4": "삭제된줄"},
        "title": " 새 제목 ", "date_kind": "촬영", "last": 10,
    })
    assert b["deleted"] == [4]
    assert b["texts"] == {"5": "수정 됨"}  # unchanged text and deleted line's text dropped
    assert b["title"] == "새 제목" and b["date_kind"] == "촬영" and b["last"] == 10


def test_apply_block_edit_rejects_deleting_everything_and_bad_range():
    segs = _segs()
    b = _block(segs)
    before = dict(b)
    with pytest.raises(ValueError):
        editjob.apply_block_edit(b, segs, {"deleted": list(range(2, 10))})
    with pytest.raises(ValueError):
        editjob.apply_block_edit(b, segs, {"first": 5, "last": 2})
    with pytest.raises(ValueError):
        editjob.apply_block_edit(b, segs, {"last": 500})
    assert b["first"] == before["first"] and b["deleted"] == []


def test_render_hash_changes_with_text_cut_and_metadata_but_not_title():
    segs = _segs()
    b = _block(segs)
    h0, c0 = editjob.render_hash(b, segs), editjob.clip_hash(b, segs)
    b["texts"] = {"3": "수정"}
    assert editjob.render_hash(b, segs) != h0 and editjob.clip_hash(b, segs) == c0
    h1 = editjob.render_hash(b, segs)
    b["title"] = "업로드 제목만 변경"
    assert editjob.render_hash(b, segs) == h1
    b["deleted"] = [5]
    assert editjob.clip_hash(b, segs) != c0


def test_render_state_becomes_stale_after_edit():
    segs = _segs()
    b = _block(segs)
    b["render"] = {"status": "ready", "hash": editjob.render_hash(b, segs), "message": "", "version": 1}
    assert editjob.render_state(b, segs) == "ready"
    b["texts"] = {"3": "수정"}
    assert editjob.render_state(b, segs) == "stale"


def _new_job(tmp_path, clip="1min"):
    return editjob.create_job(
        tmp_path, source_type="youtube", url="u", clip_length_key=clip,
        tiers=["hook", "substantive"], tier_counts={"hook": 3, "substantive": 2}, keyword="",
    )


def test_job_persistence_and_listing(tmp_path):
    job = _new_job(tmp_path, "2min")
    editjob.save_segments(tmp_path, job["id"], _segs(3))
    assert [s.text for s in editjob.load_segments(tmp_path, job["id"])] == ["줄 0", "줄 1", "줄 2"]
    editjob.update_job(tmp_path, job["id"], lambda j: j.update(status="editing", video_title="영상"))
    assert editjob.load_job(tmp_path, job["id"])["video_title"] == "영상"
    listed = editjob.list_jobs(tmp_path)
    assert listed[0]["id"] == job["id"] and listed[0]["stage"] == "자막 편집 대기"
    assert editjob.load_job(tmp_path, "../etc") is None


def test_cleanup_only_removes_old_finished_jobs(tmp_path):
    done = _new_job(tmp_path)
    open_job = _new_job(tmp_path)
    long_ago = (datetime.now(timezone.utc) - timedelta(days=8)).isoformat()
    editjob.update_job(tmp_path, done["id"], lambda j: j.update(finished_at=long_ago))
    editjob.update_job(tmp_path, open_job["id"], lambda j: j.update(created_at=long_ago))
    assert editjob.cleanup_old(tmp_path) == [done["id"]]
    assert editjob.load_job(tmp_path, open_job["id"]) is not None


def test_mark_finished_when_all_enabled_blocks_uploaded():
    segs = _segs()
    job = {"blocks": [_block(segs), _block(segs)], "finished_at": None}
    job["blocks"][0]["upload"]["status"] = "done"
    editjob.mark_finished_if_done(job)
    assert job["finished_at"] is None
    job["blocks"][1]["enabled"] = False
    editjob.mark_finished_if_done(job)
    assert job["finished_at"]


def test_title_joins_both_thumbnail_lines():
    assert editjob.title_from_thumbnail("첫째 줄\n둘째 줄") == "첫째 줄 둘째 줄"
    assert editjob.title_from_thumbnail("한 줄만") == "한 줄만"
    assert editjob.title_from_thumbnail("", "대체") == "대체"
