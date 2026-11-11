"""End-to-end test of the staged edit flow through the real FastAPI app, with
the slow/external parts (Claude analysis, ffmpeg cutting/rendering, YouTube
upload) replaced by fakes."""
import os
import time
from types import SimpleNamespace

import pytest

os.environ.setdefault("ANTHROPIC_API_KEY", "test")

from fastapi.testclient import TestClient  # noqa: E402

from autoshorts import editjob  # noqa: E402
from autoshorts.analyze import Candidate, ClipRange  # noqa: E402
from autoshorts.transcript import Segment  # noqa: E402
from webui import server  # noqa: E402


def _segments():
    roles = ["앵커"] * 2 + ["패널"] * 10
    return [Segment(i * 5.0, i * 5.0 + 5.0, f"줄 {i}", r) for i, r in enumerate(roles)]


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "WORK_DIR", tmp_path)
    calls = {"cuts": [], "renders": [], "uploads": [], "prepare": []}

    async def fake_prepare(work_dir, source_type, url, upload_path, clip_length_key, tiers, keyword,
                           tier_counts, on_status):
        on_status("분석 중...")
        calls["prepare"].append((source_type, url, upload_path, clip_length_key, tiers, keyword, tier_counts))
        segs = _segments()
        cands = [
            Candidate(ranges=[ClipRange(10.0, 45.0)], title="훅1", summary="요약1",
                      thumbnail_text="첫줄\n둘째줄", reason="r", tier="hook"),
            Candidate(ranges=[ClipRange(20.0, 55.0)], title="주제1", summary="요약2",
                      thumbnail_text="A\nB", reason="r", tier="substantive"),
        ]
        return {"source": SimpleNamespace(id="vid"), "video_title": "원본 영상", "segments": segs,
                "transcript_source": "captions", "guest_label_guess": "홍길동 교수 / 대학",
                "shoot_date": "11월 4일", "date_kind": "방송", "candidates": cands}

    def fake_cut(source, ranges, out_path, work_dir=None, bounds=None):
        calls["cuts"].append(([(r.start_sec, r.end_sec) for r in ranges], bounds))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"clip")
        return out_path, list(ranges)

    def fake_render(clip_path, cues, assets, out_path, single_camera=False):
        calls["renders"].append(([c[2] for c in cues], assets))
        out_path.write_bytes(b"final")
        return out_path

    def fake_upload(path, title, description, secret, token, privacy):
        calls["uploads"].append((title, description, privacy))
        return SimpleNamespace(video_id="abc123", url="https://youtu.be/abc123")

    monkeypatch.setattr(server, "_prepare_source_and_candidates", fake_prepare)
    monkeypatch.setattr(server.clipbuild, "build_clip_snapped", fake_cut)
    monkeypatch.setattr(server.render, "render_final", fake_render)
    monkeypatch.setattr(server.youtube_upload, "upload_short", fake_upload)
    with TestClient(server.app) as client:
        yield client, calls


def _wait(client, job_id, cond, timeout=10):
    end = time.time() + timeout
    while time.time() < end:
        st = client.get(f"/edit-jobs/{job_id}/status").json()
        if cond(st):
            return st
        time.sleep(0.05)
    raise AssertionError(f"timeout: {st}")


def _start(client, **extra):
    r = client.post("/edit-jobs", data={"url": "https://youtu.be/abcdefghijk", "clip_length": "2min", **extra})
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]
    st = _wait(client, job_id, lambda s: s["status"] != "analyzing")
    assert st["status"] == "editing"
    return job_id


def test_analysis_creates_blocks_with_prefilled_info(env):
    client, _ = env
    job_id = _start(client)
    data = client.get(f"/edit-jobs/{job_id}/data").json()
    assert [b["tier"] for b in data["blocks"]] == ["hook", "substantive"]
    b0 = data["blocks"][0]
    assert (b0["first"], b0["last"]) == (2, 8)
    assert b0["guest_label"] == "홍길동 교수 / 대학" and b0["shoot_date"] == "11월 4일"
    assert b0["title"] == "첫줄 둘째줄" and data["source_type"] == "youtube" and data["source_url"] == ""
    assert len(data["segments"]) == 12
    # shows up in the summary list; the old 검토함/수동 pages are gone
    assert client.get("/auto/summary").json()["edit_jobs"][0]["id"] == job_id
    assert client.get("/library").status_code == 404 and client.get("/manual").status_code == 404
    assert client.get("/").history or client.get("/", follow_redirects=False).status_code in (302, 307)


def test_edit_render_with_deleted_line_cuts_footage_and_uses_edited_caption(env):
    client, calls = env
    job_id = _start(client)
    payload = {"deleted": [5], "texts": {"4": "고친 자막"}, "thumbnail_text": "새\n문구"}
    assert client.post(f"/edit-jobs/{job_id}/blocks/0/render", json=payload).json() == {"started": True}
    st = _wait(client, job_id, lambda s: s["blocks"][0]["render_state"] in ("ready", "error"))
    assert st["blocks"][0]["render_state"] == "ready", st["blocks"][0]["render"]
    ranges, bounds = calls["cuts"][0]
    assert len(ranges) == 2 and ranges[0][1] == 25.0 and ranges[1][0] == 30.0  # line 5 (25-30s) cut out
    assert bounds == [(None, 25.0), (30.0, None)]
    texts, assets = calls["renders"][0]
    assert "고친 자막" in texts and "줄 5" not in texts and assets.thumbnail_text == "새\n문구"
    assert client.get(f"/edit-jobs/{job_id}/blocks/0/final.mp4").content == b"final"

    # text-only edit: stale until re-rendered, and re-render reuses the cut footage
    client.post(f"/edit-jobs/{job_id}/blocks/0/save", json={"texts": {"4": "또 고침"}, "deleted": [5]})
    assert client.get(f"/edit-jobs/{job_id}/status").json()["blocks"][0]["render_state"] == "stale"
    client.post(f"/edit-jobs/{job_id}/blocks/0/render", json={})
    _wait(client, job_id, lambda s: s["blocks"][0]["render_state"] == "ready")
    assert len(calls["cuts"]) == 1 and len(calls["renders"]) == 2


def test_upload_requires_fresh_render_then_goes_private(env):
    client, calls = env
    job_id = _start(client)
    assert client.post(f"/edit-jobs/{job_id}/blocks/0/upload").status_code == 409  # not rendered yet
    client.post(f"/edit-jobs/{job_id}/blocks/0/render", json={"title": "내 제목"})
    _wait(client, job_id, lambda s: s["blocks"][0]["render_state"] == "ready")
    assert client.post(f"/edit-jobs/{job_id}/blocks/0/upload").json() == {"started": True}
    st = _wait(client, job_id, lambda s: s["blocks"][0]["upload"]["status"] in ("done", "error"))
    up = st["blocks"][0]
    assert up["upload"]["status"] == "done"
    assert up["studio_url"] == "https://studio.youtube.com/video/abc123/edit"
    title, description, privacy = calls["uploads"][0]
    assert title == "내 제목" and privacy == "private" and "11월 4일" in description
    # an uploaded block is locked, and one block done != job finished
    assert client.post(f"/edit-jobs/{job_id}/blocks/0/save", json={"deleted": [3]}).status_code == 409
    assert st["stage"] == "일부 업로드됨"


def test_invalid_edits_are_rejected_with_message(env):
    client, _ = env
    job_id = _start(client)
    r = client.post(f"/edit-jobs/{job_id}/blocks/0/save", json={"deleted": list(range(2, 9))})
    assert r.status_code == 400 and "한 줄" in r.json()["detail"]
    assert client.post(f"/edit-jobs/{job_id}/blocks/9/save", json={}).status_code == 404


def test_delete_job_and_recover_interrupted(env, tmp_path):
    client, _ = env
    job_id = _start(client)
    editjob.update_job(tmp_path, job_id, lambda j: j["blocks"][0]["upload"].update(status="uploading"))
    editjob.recover_interrupted(tmp_path)
    assert editjob.load_job(tmp_path, job_id)["blocks"][0]["upload"]["status"] == "error"
    assert client.post(f"/edit-jobs/{job_id}/delete").json() == {"ok": True}
    assert client.get(f"/edit-jobs/{job_id}/data").status_code == 404


def test_full_auto_requires_a_link_or_a_file(env):
    client, _ = env
    assert client.post("/auto-jobs", data={"source_type": "url", "url": ""}).status_code == 400
    assert client.post("/auto-jobs", data={"source_type": "file"}).status_code == 400


def test_options_are_validated_and_passed_to_analysis(env):
    client, calls = env
    job_id = _start(client, clip_length="custom", custom_seconds="90", tiers="hook,keyword",
                    keyword=" 금리 ", count_hook="1", count_keyword="4")
    assert calls["prepare"][0][3:] == ("custom:90", ["hook", "keyword"], "금리",
                                       {"hook": 1, "substantive": 2, "keyword": 4})
    summary = client.get("/auto/summary").json()
    assert summary["edit_jobs"][0]["id"] == job_id

    def bad(**extra):
        data = {"url": "https://youtu.be/abcdefghijk", **extra}
        return client.post("/edit-jobs", data=data).status_code
    assert bad(clip_length="custom", custom_seconds="500") == 400      # over the 3-minute shorts limit
    assert bad(clip_length="custom", custom_seconds="abc") == 400
    assert bad(tiers="keyword", keyword="") == 400                      # keyword tier needs a keyword
    assert bad(tiers="nonsense") == 400                                 # nothing valid selected
    assert bad(count_hook="9") == 400
    assert client.post("/edit-jobs", data={"url": "https://example.com/x"}).status_code == 400


def test_file_source_is_saved_with_the_job_and_served_for_the_original_span_player(env, tmp_path):
    client, calls = env
    r = client.post("/edit-jobs", data={"source_type": "file", "clip_length": "1min"},
                    files={"file": ("../evil/내 영상.mp4", b"RAWVIDEO", "video/mp4")})
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]
    _wait(client, job_id, lambda s: s["status"] != "analyzing")
    source_type, url, upload_path, *_ = calls["prepare"][0]
    assert source_type == "file" and url == ""
    # only the bare file name is kept, inside this job's own folder
    assert upload_path.name == "내 영상.mp4" and tmp_path in upload_path.parents
    data = client.get(f"/edit-jobs/{job_id}/data").json()
    assert data["source_type"] == "file" and data["video_id"] == ""
    assert data["source_url"] == f"/edit-jobs/{job_id}/source"
    assert client.get(data["source_url"]).content == b"RAWVIDEO"
    assert client.get(data["source_url"], headers={"Range": "bytes=0-2"}).content == b"RAW"
    # a YouTube job has no original file to serve
    yt = _start(client)
    assert client.get(f"/edit-jobs/{yt}/source").status_code == 404


# ---- the real shared analysis step (only the slow external calls are faked) ----

@pytest.fixture()
def real_prepare(tmp_path, monkeypatch):
    """Runs server._prepare_source_and_candidates for real; fakes only the
    YouTube source, transcript, and Claude calls."""
    monkeypatch.setattr(server, "WORK_DIR", tmp_path)
    seen = {}

    class FakeSource:
        id = "vid"
        title = "원본 영상"
        def __init__(self, url):
            self.url = url

    def fake_transcript(source, work_dir):
        return _segments(), "captions"

    def fake_propose(segments, clip_length, tiers, title, keyword, counts=None):
        seen["counts"] = counts
        seen["tiers"] = tiers
        return [Candidate(ranges=[ClipRange(10.0, 45.0)], title="훅1", summary="요약",
                          thumbnail_text="첫줄\n둘째줄", reason="r", tier="hook")]

    monkeypatch.setattr(server, "YouTubeSource", FakeSource)
    monkeypatch.setattr(server, "get_transcript_for_source", fake_transcript)
    monkeypatch.setattr(server.analyze, "propose_candidates", fake_propose)
    monkeypatch.setattr(server.analyze, "refine_boundaries", lambda cands, segs, clip: cands)
    return seen


def test_real_analysis_step_runs_end_to_end_with_tier_counts(real_prepare):
    with TestClient(server.app) as client:
        job_id = _start(client)
        data = client.get(f"/edit-jobs/{job_id}/data").json()
    assert real_prepare["counts"] == {"hook": 3, "substantive": 2, "keyword": 2}
    assert real_prepare["tiers"] == ["hook", "substantive"]
    assert data["video_title"] == "원본 영상" and data["blocks"][0]["guest_label"] == ""


def test_full_auto_mode_runs_through_to_private_upload(real_prepare, monkeypatch):
    uploads, renders = [], []
    monkeypatch.setattr(server.clipbuild, "build_clip_snapped",
                        lambda source, ranges, out: (out.write_bytes(b"c") or out, list(ranges)))
    monkeypatch.setattr(server.analyze, "correct_caption_errors", lambda texts: texts)
    monkeypatch.setattr(server.render, "render_final",
                        lambda clip, cues, assets, out, single_camera=False: (renders.append(single_camera) or out.write_bytes(b"f") or out))
    monkeypatch.setattr(server.youtube_upload, "upload_short",
                        lambda path, title, desc, s, t, privacy: uploads.append((title, privacy))
                        or SimpleNamespace(video_id="v1", url="https://youtu.be/v1"))
    with TestClient(server.app) as client:
        r = client.post("/auto-jobs", data={"source_type": "url", "url": "https://youtu.be/x",
                                            "clip_length": "2min"})
        assert r.status_code == 200, r.text
        auto_id = r.json()["auto_job_id"]
        end = time.time() + 10
        while time.time() < end:
            st = client.get(f"/auto-jobs/{auto_id}").json()
            if st["status"] in ("done", "error"):
                break
            time.sleep(0.05)
    assert st["status"] == "done", st
    assert uploads == [("첫줄 둘째줄", "private")]
    assert renders == [True]  # 범용 버전은 항상 단일 카메라 기준


def test_edit_data_gives_the_youtube_video_id_for_the_original_span_player(env):
    client, _ = env
    job_id = _start(client)
    assert client.get(f"/edit-jobs/{job_id}/data").json()["video_id"] == "abcdefghijk"


def test_company_logo_upload_preview_and_delete(env, tmp_path, monkeypatch):
    import io
    from PIL import Image
    client, _ = env
    monkeypatch.setattr(server, "DEFAULT_ASSETS_DIR", tmp_path / "assets")
    buf = io.BytesIO(); Image.new("RGB", (40, 20), "red").save(buf, "JPEG")
    assert not any(s["has_image"] for s in client.get("/assets").json())
    assert client.post("/assets/logo_left", files={"file": ("l.jpg", buf.getvalue(), "image/jpeg")}).status_code == 200
    assert {s["slot"]: s["has_image"] for s in client.get("/assets").json()}["logo_left"] is True
    assert client.get("/assets/logo_left/image").status_code == 200
    assert server._find_default_asset("logo_left").suffix == ".png"
    assert client.post("/assets/logo_left", files={"file": ("x.txt", b"nope", "text/plain")}).status_code == 400
    assert client.post("/assets/evil", files={"file": ("l.jpg", buf.getvalue(), "image/jpeg")}).status_code == 404
    client.post("/assets/logo_left/delete")
    assert client.get("/assets/logo_left/image").status_code == 404
