"""Local web app for autoshorts (범용 버전).

Run with: uvicorn webui.server:app --reload --port 8787

Runs on the worker's own machine (not a shared server) so downloads and
rendering happen locally - see the project README for why. One screen,
"자동 생성" (/auto): 영상(유튜브 링크 또는 원본 파일), 후보 선정 기준(훅 /
핵심 정보 / 키워드), 길이(프리셋 또는 사용자 지정)를 고르면

- staged (✂️ 편집 시작): analysis proposes candidates; the worker edits each
  block's caption lines (deleting a line cuts that footage), watches a
  confirmation render, then uploads privately. State is persisted by
  editjob.py so jobs survive restarts.
- fully automatic (⚡ 완전 자동): the same analysis, then render + private
  upload of every candidate with no editing (AUTO_JOBS, in memory).
"""

from __future__ import annotations

import asyncio
import functools
import json
import os
import shutil
import sys
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# a PyInstaller-frozen (onedir) build has two distinct base paths, and
# conflating them was a real bug caught by test-building this spec: bundled
# read-only data (templates/, static/ - see installer/autoshorts.spec)
# lands under sys._MEIPASS (PyInstaller >= 6 puts this in an _internal/
# subfolder next to the exe), while user-facing/writable things (.env,
# work/, ffmpeg_bin/ - see installer/launcher.py) belong directly next to
# the exe itself, i.e. sys.executable's directory. __file__ points inside
# the frozen bundle's internals and isn't usable as a base path at all in
# this mode. Dev mode (plain `uvicorn webui.server:app`) is unaffected -
# it keeps using __file__ exactly as before.
if getattr(sys, "frozen", False):
    REPO_ROOT = Path(sys.executable).resolve().parent
    WEBUI_DIR = Path(sys._MEIPASS)
else:
    WEBUI_DIR = Path(__file__).resolve().parent
    REPO_ROOT = WEBUI_DIR.parent
    SRC_DIR = REPO_ROOT / "src"
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))

load_dotenv(REPO_ROOT / ".env")

from autoshorts import analyze, clipbuild, editjob, library, render, youtube_upload  # noqa: E402
from autoshorts.config import (  # noqa: E402
    CLIP_LENGTH_PRESETS, CUSTOM_MAX_SEC, CUSTOM_MIN_SEC, DEFAULT_CLIP_LENGTH_PRESET, DEFAULT_TIER_COUNTS,
    MAX_TIER_COUNT, get_clip_length_preset, make_custom_key,
)
from autoshorts.sources import UploadedFileSource, YouTubeSource  # noqa: E402
from autoshorts.sources.youtube_source import _extract_video_id  # noqa: E402
from autoshorts.transcript import get_transcript_for_source  # noqa: E402

WORK_DIR = REPO_ROOT / "work"
WORK_DIR.mkdir(exist_ok=True)

# "완전 자동 모드" (see create_auto_job below) needs images for every clip
# it produces without a worker there to upload any - default_assets/ is a
# user-maintained folder (see its README.txt) a department admin drops a
# fixed logo/banner set into once, reused for every auto-run.
DEFAULT_ASSETS_DIR = REPO_ROOT / "default_assets"

# YouTube Data API OAuth client - see youtube_upload.py. Defaults to files
# sitting next to .env so there's nothing extra to configure for the common
# case; overridable via env var for a non-standard layout.
YOUTUBE_CLIENT_SECRET_FILE = Path(
    os.environ.get("YOUTUBE_CLIENT_SECRET_FILE") or str(REPO_ROOT / "client_secret.json")
)
YOUTUBE_TOKEN_FILE = Path(os.environ.get("YOUTUBE_TOKEN_FILE") or str(REPO_ROOT / "youtube_token.json"))


def _find_default_asset(name: str) -> Path | None:
    """default_assets/<name>.(png|jpg|jpeg) - 있으면 영상에 넣고 없으면 그 자리는 비워 둔다
    (default_assets/README.txt 참고)."""
    for ext in (".png", ".jpg", ".jpeg"):
        path = DEFAULT_ASSETS_DIR / f"{name}{ext}"
        if path.is_file():
            return path
    return None


# optional: a playlist on the worker's own channel every uploaded short also
# gets added to, in addition to just sitting on the channel. Skipped if unset.
PLAYLIST_ID = os.environ.get("AUTOSHORTS_PLAYLIST_ID") or ""

app = FastAPI(title="autoshorts")
templates = Jinja2Templates(directory=str(WEBUI_DIR / "templates"))
app.mount(
    "/static", StaticFiles(directory=str(WEBUI_DIR / "static")), name="static"
)

# 완전 자동 모드 (see create_auto_job): source -> candidates -> (build,
# correct captions, render, upload) per candidate, all unattended.
AUTO_JOBS: dict[str, dict] = {}


def _auto_job_work_dir(auto_job_id: str) -> Path:
    return WORK_DIR / "auto_jobs" / auto_job_id


# ---------------------------------------------------------------------
# 화면 구성: 자동 생성(/auto, 첫 화면) 하나.
#   - 단계별 작업(✂️ 편집 시작): 후보 분석 → 자막 편집(줄 삭제 = 영상 컷)
#     → 확인 영상 → 비공개 업로드. 상태는 work/edit_jobs/에 저장된다(editjob.py).
#   - 완전 자동(⚡): 편집 없이 끝까지 만들어 비공개 업로드.
# ---------------------------------------------------------------------

STUDIO_URL = "https://studio.youtube.com/video/{video_id}/edit"


@app.get("/")
def index():
    return RedirectResponse("/auto")


@app.get("/auto", response_class=HTMLResponse)
def auto_view(request: Request):
    return templates.TemplateResponse(
        request, "auto.html",
        {
            "clip_length_presets": CLIP_LENGTH_PRESETS,
            "default_clip_length": DEFAULT_CLIP_LENGTH_PRESET,
            "tier_counts": DEFAULT_TIER_COUNTS,
            "max_tier_count": MAX_TIER_COUNT,
            "custom_min": CUSTOM_MIN_SEC,
            "custom_max": CUSTOM_MAX_SEC,
        },
    )


@app.get("/auto/summary")
def auto_summary():
    """Everything the 자동 생성 page shows: shorts made today, the staged
    edit jobs, and the recent one-click auto jobs."""
    today = datetime.now(timezone.utc).astimezone().date().isoformat()
    made_today = sum(
        1 for it in library.list_items(WORK_DIR)
        if datetime.fromisoformat(it["created_at"]).astimezone().date().isoformat() == today
    )
    recent_auto = [
        {"id": job_id, "video_title": job.get("video_title", ""),
         "status": job.get("status", ""), "message": job.get("message", "")}
        for job_id, job in reversed(list(AUTO_JOBS.items()))
    ][:8]
    return {"made_today": made_today, "edit_jobs": editjob.list_jobs(WORK_DIR), "recent_jobs": recent_auto}


# ------------------------------------------------------------- 회사 로고 (default_assets)

ASSET_SLOTS = {
    "logo_left": "좌측 상단 로고",
    "logo_right": "우측 상단 로고",
    "banner_bottom": "중앙 하단 큰 로고",
}


@app.get("/assets")
def assets_status():
    return [
        {"slot": slot, "label": label, "has_image": _find_default_asset(slot) is not None}
        for slot, label in ASSET_SLOTS.items()
    ]


@app.get("/assets/{slot}/image")
def asset_image(slot: str):
    path = _find_default_asset(slot) if slot in ASSET_SLOTS else None
    if path is None:
        raise HTTPException(404, "등록된 로고가 없습니다.")
    return FileResponse(path, headers={"Cache-Control": "no-store"})


@app.post("/assets/{slot}")
async def asset_upload(slot: str, file: UploadFile):
    """로고 이미지를 default_assets/<slot>.png로 저장한다 (이후 만드는 쇼츠부터 적용)."""
    if slot not in ASSET_SLOTS:
        raise HTTPException(404, "unknown slot")
    from PIL import Image
    try:
        img = Image.open(file.file)
        img.load()
    except Exception:  # noqa: BLE001 - not an image
        raise HTTPException(400, "이미지 파일(png/jpg)을 선택해 주세요.") from None
    DEFAULT_ASSETS_DIR.mkdir(exist_ok=True)
    _remove_asset(slot)
    img.convert("RGBA").save(DEFAULT_ASSETS_DIR / f"{slot}.png")
    return {"ok": True}


def _remove_asset(slot: str) -> None:
    for ext in (".png", ".jpg", ".jpeg"):
        (DEFAULT_ASSETS_DIR / f"{slot}{ext}").unlink(missing_ok=True)


@app.post("/assets/{slot}/delete")
def asset_delete(slot: str):
    if slot not in ASSET_SLOTS:
        raise HTTPException(404, "unknown slot")
    _remove_asset(slot)
    return {"ok": True}


# ------------------------------------------------------------- 작업 옵션

TIER_KEYS = ("hook", "substantive", "keyword")


def _parse_options(
    clip_length: str, custom_seconds: str, tiers: str, keyword: str,
    count_hook: int, count_substantive: int, count_keyword: int,
) -> tuple[str, list[str], dict[str, int], str]:
    """화면에서 넘어온 길이/선정 기준/개수/키워드를 검증해서
    (길이 키, 티어 목록, 티어별 개수, 키워드)로 돌려준다. 잘못되면 400."""
    clip_key = clip_length.strip() or DEFAULT_CLIP_LENGTH_PRESET
    if clip_key == "custom":
        try:
            clip_key = make_custom_key(int(custom_seconds))
        except ValueError:
            raise HTTPException(400, "사용자 지정 길이는 초 단위 숫자로 입력해 주세요.") from None
    try:
        get_clip_length_preset(clip_key)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e

    selected = [t for t in (x.strip() for x in tiers.split(",")) if t in TIER_KEYS]
    keyword = keyword.strip()
    if "keyword" in selected and not keyword:
        raise HTTPException(400, "키워드 기준을 쓰려면 키워드를 입력해 주세요.")
    if not selected:
        raise HTTPException(400, "후보 선정 기준(훅 / 핵심 정보 / 키워드)을 하나 이상 선택해 주세요.")
    counts = {"hook": count_hook, "substantive": count_substantive, "keyword": count_keyword}
    for tier in selected:
        if not (1 <= counts[tier] <= MAX_TIER_COUNT):
            raise HTTPException(400, f"기준별 후보 개수는 1~{MAX_TIER_COUNT}개 사이여야 합니다.")
    return clip_key, selected, counts, keyword


def _save_upload(file: UploadFile, dest_dir: Path) -> Path:
    """브라우저로 올린 원본 영상 파일을 dest_dir에 저장한다(경로 조작 방지로 이름만 사용)."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    name = Path(file.filename or "video.mp4").name or "video.mp4"
    path = dest_dir / name
    with path.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    return path


# ------------------------------------------------------------- 단계별 작업

SOURCES: dict[str, object] = {}  # job_id -> source, so video info isn't re-fetched
RENDER_LOCK = asyncio.Semaphore(1)  # one confirmation render/upload at a time (CPU-heavy)


def _job_or_404(job_id: str) -> dict:
    job = editjob.load_job(WORK_DIR, job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    return job


def _block_or_404(job: dict, index: int) -> dict:
    if not (0 <= index < len(job["blocks"])):
        raise HTTPException(404, "unknown block")
    return job["blocks"][index]


def _source_for(job: dict):
    source = SOURCES.get(job["id"])
    if source is None:
        if job.get("source_type") == "file":
            source = UploadedFileSource(Path(job["source_path"]))
        else:
            source = YouTubeSource(job["url"])
        SOURCES[job["id"]] = source
    return source


@app.post("/edit-jobs")
async def create_edit_job(
    source_type: str = Form("url"),
    url: str = Form(""),
    file: UploadFile | None = None,
    clip_length: str = Form(DEFAULT_CLIP_LENGTH_PRESET),
    custom_seconds: str = Form(""),
    tiers: str = Form("hook,substantive"),
    keyword: str = Form(""),
    count_hook: int = Form(DEFAULT_TIER_COUNTS["hook"]),
    count_substantive: int = Form(DEFAULT_TIER_COUNTS["substantive"]),
    count_keyword: int = Form(DEFAULT_TIER_COUNTS["keyword"]),
):
    if source_type not in ("url", "file"):
        raise HTTPException(400, "source_type must be 'url' or 'file'")
    clip_key, selected, counts, keyword = _parse_options(
        clip_length, custom_seconds, tiers, keyword, count_hook, count_substantive, count_keyword,
    )
    url = url.strip()
    if source_type == "url":
        if not url:
            raise HTTPException(400, "유튜브 링크를 입력해 주세요.")
        try:
            _extract_video_id(url)
        except ValueError:
            raise HTTPException(400, "유튜브 영상 링크가 아닙니다. 링크를 다시 확인해 주세요.") from None
    elif file is None or not file.filename:
        raise HTTPException(400, "원본 영상 파일을 선택해 주세요.")

    job = editjob.create_job(
        WORK_DIR, source_type="file" if source_type == "file" else "youtube", url=url,
        clip_length_key=clip_key, tiers=selected, tier_counts=counts, keyword=keyword,
    )
    if source_type == "file":
        path = _save_upload(file, editjob.job_dir(WORK_DIR, job["id"]) / "source")
        editjob.update_job(WORK_DIR, job["id"], lambda j: j.update(source_path=str(path)))
    asyncio.create_task(_run_edit_analysis(job["id"]))
    return {"job_id": job["id"]}


async def _run_edit_analysis(job_id: str) -> None:
    def set_message(msg: str) -> None:
        try:
            editjob.update_job(WORK_DIR, job_id, lambda j: j.update(message=msg))
        except KeyError:
            pass  # job was deleted meanwhile

    job = editjob.load_job(WORK_DIR, job_id)
    if job is None:
        return
    try:
        result = await _prepare_source_and_candidates(
            editjob.job_dir(WORK_DIR, job_id),
            "file" if job.get("source_type") == "file" else "url", job["url"],
            Path(job["source_path"]) if job.get("source_path") else None, job["clip_length_key"],
            job["tiers"], job["keyword"], job["tier_counts"], on_status=set_message,
        )
        SOURCES[job_id] = result["source"]
        segments = result["segments"]
        blocks = editjob.build_blocks(
            result["candidates"], segments,
            guest_label=result["guest_label_guess"], shoot_date=result["shoot_date"],
            date_kind=result["date_kind"],
        )
        if not blocks:
            raise RuntimeError("조건에 맞는 쇼츠 후보를 찾지 못했습니다.")
        editjob.save_segments(WORK_DIR, job_id, segments)
        editjob.update_job(
            WORK_DIR, job_id,
            lambda j: j.update(status="editing", message="", video_title=result["video_title"], blocks=blocks),
        )
    except Exception as e:  # noqa: BLE001 - surface any failure to the page
        traceback.print_exc()
        try:
            editjob.update_job(WORK_DIR, job_id, lambda j: j.update(status="error", message=str(e)))
        except KeyError:
            pass


@app.get("/edit-jobs/{job_id}/view", response_class=HTMLResponse)
def edit_job_view(request: Request, job_id: str):
    _job_or_404(job_id)
    return templates.TemplateResponse(request, "edit.html", {"job_id": job_id})


def _block_view(job: dict, segments: list, block: dict, clip_max_sec: int) -> dict:
    """A block plus the derived facts the page needs."""
    view = dict(block)
    if segments:
        view["render_state"] = editjob.render_state(block, segments)
        view["duration"] = round(editjob.block_duration(block, segments), 1)
    else:
        view["render_state"] = block["render"]["status"]
        view["duration"] = 0
    view["render_version"] = block["render"].get("version", 0)
    view["studio_url"] = STUDIO_URL.format(video_id=block["upload"]["video_id"]) if block["upload"].get("video_id") else ""
    view["max_sec"] = clip_max_sec
    return view


def _job_view(job: dict, include_segments: bool) -> dict:
    segments = editjob.load_segments(WORK_DIR, job["id"]) if job["status"] != "analyzing" else []
    try:
        clip_max_sec = get_clip_length_preset(job["clip_length_key"]).max_sec
    except ValueError:
        clip_max_sec = 0
    is_file = job.get("source_type") == "file"
    try:
        video_id = "" if is_file else _extract_video_id(job["url"])
    except ValueError:
        video_id = ""  # not a YouTube link: the page falls back to the confirmation render only
    out = {
        "id": job["id"], "video_title": job["video_title"], "source_type": job.get("source_type", "youtube"),
        "url": job["url"], "video_id": video_id,
        # 원본 파일 작업은 서버가 원본을 그대로 내려줘서 <video>로 구간 재생한다
        "source_url": f"/edit-jobs/{job['id']}/source" if is_file else "",
        "status": job["status"], "message": job["message"],
        "stage": editjob.stage_label(job),
        "blocks": [_block_view(job, segments, b, clip_max_sec) for b in job["blocks"]],
        "context_lines": editjob.CONTEXT_LINES,
    }
    if include_segments:
        out["segments"] = [
            {"i": i, "start": s.start_sec, "end": s.end_sec, "text": s.text, "speaker": s.speaker}
            for i, s in enumerate(segments)
        ]
    return out


@app.get("/edit-jobs/{job_id}/data")
def edit_job_data(job_id: str):
    """Full state incl. the transcript - loaded once when the page opens."""
    return _job_view(_job_or_404(job_id), include_segments=True)


@app.get("/edit-jobs/{job_id}/status")
def edit_job_status(job_id: str):
    """Lightweight state for polling (no transcript)."""
    return _job_view(_job_or_404(job_id), include_segments=False)


@app.post("/edit-jobs/{job_id}/delete")
def edit_job_delete(job_id: str):
    _job_or_404(job_id)
    SOURCES.pop(job_id, None)
    editjob.delete_job(WORK_DIR, job_id)
    return {"ok": True}


async def _read_edit_payload(request: Request) -> dict:
    try:
        payload = await request.json()
    except Exception:  # noqa: BLE001 - empty body is fine
        return {}
    return payload if isinstance(payload, dict) else {}


def _apply_edit(job_id: str, index: int, payload: dict) -> dict:
    """Validate + persist a block edit; returns the updated job."""
    job = _job_or_404(job_id)
    block = _block_or_404(job, index)
    if block["upload"]["status"] in ("uploading", "done"):
        raise HTTPException(409, "이미 업로드한 항목은 수정할 수 없습니다.")
    if block["render"]["status"] == "rendering":
        raise HTTPException(409, "확인 영상을 만드는 중에는 수정할 수 없습니다.")
    segments = editjob.load_segments(WORK_DIR, job_id)

    def mutate(j: dict) -> None:
        editjob.apply_block_edit(j["blocks"][index], segments, payload)
        editjob.mark_finished_if_done(j)

    try:
        return editjob.update_job(WORK_DIR, job_id, mutate)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@app.post("/edit-jobs/{job_id}/blocks/{index}/save")
async def edit_block_save(request: Request, job_id: str, index: int):
    job = _apply_edit(job_id, index, await _read_edit_payload(request))
    segments = editjob.load_segments(WORK_DIR, job_id)
    max_sec = get_clip_length_preset(job["clip_length_key"]).max_sec
    return {"block": _block_view(job, segments, job["blocks"][index], max_sec)}


@app.post("/edit-jobs/{job_id}/blocks/{index}/render")
async def edit_block_render(request: Request, job_id: str, index: int):
    """Saves the posted edit (if any) and starts the confirmation render."""
    payload = await _read_edit_payload(request)
    job = _apply_edit(job_id, index, payload) if payload else _job_or_404(job_id)
    block = _block_or_404(job, index)
    if not block.get("enabled", True):
        raise HTTPException(400, "사용하지 않는 항목입니다.")
    if block["upload"]["status"] in ("uploading", "done"):
        raise HTTPException(409, "이미 업로드한 항목입니다.")
    if block["render"]["status"] == "rendering":
        return {"started": False}

    def mark(j: dict) -> None:
        j["blocks"][index]["render"].update(status="rendering", message="대기 중...")

    editjob.update_job(WORK_DIR, job_id, mark)
    asyncio.create_task(_run_block_render(job_id, index))
    return {"started": True}


def _set_render(job_id: str, index: int, **fields) -> None:
    try:
        editjob.update_job(WORK_DIR, job_id, lambda j: j["blocks"][index]["render"].update(fields))
    except KeyError:
        pass


async def _run_block_render(job_id: str, index: int) -> None:
    loop = asyncio.get_event_loop()
    async with RENDER_LOCK:
        try:
            job = editjob.load_job(WORK_DIR, job_id)
            if job is None:
                return
            block = job["blocks"][index]
            segments = editjob.load_segments(WORK_DIR, job_id)
            bdir = editjob.block_dir(WORK_DIR, job_id, index)
            bdir.mkdir(parents=True, exist_ok=True)
            new_render_hash = editjob.render_hash(block, segments)
            ranges, bounds = editjob.block_ranges(block, segments)
            wanted_clip = editjob.clip_hash(block, segments)
            clip_path = bdir / "clip.mp4"
            used_path = bdir / "clip_ranges.json"

            # footage is only re-cut when the kept ranges changed; caption/text
            # edits reuse the existing clip
            if block.get("clip_hash") == wanted_clip and clip_path.is_file() and used_path.is_file():
                used_ranges = [analyze.ClipRange(a, b) for a, b in json.loads(used_path.read_text(encoding="utf-8"))]
            else:
                _set_render(job_id, index, message="구간 추출/이어붙이는 중...")
                source = _source_for(job)
                parts_dir = bdir / "_parts"
                clip_path, used_ranges = await loop.run_in_executor(
                    None, clipbuild.build_clip_snapped, source, ranges, clip_path, parts_dir, bounds,
                )
                shutil.rmtree(parts_dir, ignore_errors=True)
                used_path.write_text(
                    json.dumps([[r.start_sec, r.end_sec] for r in used_ranges]), encoding="utf-8")
                editjob.update_job(WORK_DIR, job_id, lambda j: j["blocks"][index].update(clip_hash=wanted_clip))

            _set_render(job_id, index, message="자막 입히고 최종 렌더링 중...")
            cues = render.captions_for_ranges(editjob.block_cue_segments(block, segments), used_ranges)
            assets = render.RenderAssets(
                thumbnail_text=block["thumbnail_text"],
                guest_label=block["guest_label"],
                shoot_date=block["shoot_date"],
                date_kind=block["date_kind"],
                logo_top_left=_find_default_asset("logo_left"),
                logo_top_right=_find_default_asset("logo_right"),
                banner_bottom=_find_default_asset("banner_bottom"),
            )
            # 범용 버전은 어떤 영상이 올지 모르므로 투카메라 합성 화면(방송 스튜디오 전용)을
            # 가정하지 않고 항상 단일 카메라 기준으로 얼굴을 찾아 구도를 잡는다.
            await loop.run_in_executor(
                None,
                functools.partial(
                    render.render_final, clip_path, cues, assets, bdir / "final.mp4",
                    single_camera=True,
                ),
            )
            version = block["render"].get("version", 0) + 1
            _set_render(job_id, index, status="ready", hash=new_render_hash, message="", version=version)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            _set_render(job_id, index, status="error", message=str(e))


@app.get("/edit-jobs/{job_id}/source")
def edit_job_source(job_id: str):
    """원본 영상 파일 작업의 원본 - 편집 화면의 구간 재생용 (Range 요청 지원)."""
    job = _job_or_404(job_id)
    path = Path(job.get("source_path") or "")
    if job.get("source_type") != "file" or not path.is_file():
        raise HTTPException(404, "원본 영상 파일이 없습니다.")
    return FileResponse(path)


@app.get("/edit-jobs/{job_id}/blocks/{index}/final.mp4")
def edit_block_video(job_id: str, index: int):
    job = _job_or_404(job_id)
    _block_or_404(job, index)
    path = editjob.block_dir(WORK_DIR, job_id, index) / "final.mp4"
    if not path.is_file():
        raise HTTPException(404, "확인 영상이 아직 없습니다.")
    return FileResponse(path, media_type="video/mp4", headers={"Cache-Control": "no-store"})


@app.post("/edit-jobs/{job_id}/blocks/{index}/upload")
async def edit_block_upload(job_id: str, index: int):
    job = _job_or_404(job_id)
    block = _block_or_404(job, index)
    segments = editjob.load_segments(WORK_DIR, job_id)
    if block["upload"]["status"] in ("uploading", "done"):
        raise HTTPException(409, "이미 업로드했거나 업로드 중입니다.")
    if editjob.render_state(block, segments) != "ready":
        raise HTTPException(409, "수정한 내용이 반영된 확인 영상을 먼저 만들어 주세요.")

    def mark(j: dict) -> None:
        j["blocks"][index]["upload"].update(status="uploading", message="유튜브에 비공개로 업로드 중...")

    editjob.update_job(WORK_DIR, job_id, mark)
    asyncio.create_task(_run_block_upload(job_id, index))
    return {"started": True}


def _set_upload(job_id: str, index: int, **fields) -> None:
    def mutate(j: dict) -> None:
        j["blocks"][index]["upload"].update(fields)
        editjob.mark_finished_if_done(j)
    try:
        editjob.update_job(WORK_DIR, job_id, mutate)
    except KeyError:
        pass


async def _run_block_upload(job_id: str, index: int) -> None:
    loop = asyncio.get_event_loop()
    try:
        job = editjob.load_job(WORK_DIR, job_id)
        block = job["blocks"][index]
        final_path = editjob.block_dir(WORK_DIR, job_id, index) / "final.mp4"
        title = block["title"].strip() or editjob.title_from_thumbnail(block["thumbnail_text"]) or "쇼츠"
        upload_title = title
        description = block["summary"]
        if block["shoot_date"]:
            description += f"\n[영상 {block['date_kind']} 날짜: {block['shoot_date']}]"
        result = await loop.run_in_executor(
            None, youtube_upload.upload_short, final_path, upload_title, description,
            YOUTUBE_CLIENT_SECRET_FILE, YOUTUBE_TOKEN_FILE, "private",
        )
        if PLAYLIST_ID:
            try:
                await loop.run_in_executor(
                    None, youtube_upload.add_to_playlist, result.video_id,
                    PLAYLIST_ID, YOUTUBE_CLIENT_SECRET_FILE, YOUTUBE_TOKEN_FILE,
                )
            except Exception:  # noqa: BLE001 - the upload itself already succeeded
                traceback.print_exc()
        library.add_item(WORK_DIR, library.new_item(
            source_label=job["video_title"], title=title, summary=block["summary"],
            thumbnail_text=block["thumbnail_text"], video_id=result.video_id, youtube_url=result.url,
        ))
        _set_upload(job_id, index, status="done", video_id=result.video_id, url=result.url, message="")
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        _set_upload(job_id, index, status="error", message=str(e))


async def _prepare_source_and_candidates(
    work_dir: Path,
    source_type: str,
    url: str,
    upload_path: Path | None,
    clip_length_key: str,
    tiers: list[str],
    keyword: str,
    tier_counts: dict[str, int],
    on_status,
) -> dict:
    """Shared pipeline behind the staged flow (_run_edit_analysis) and the
    fully-automatic flow (_run_auto_pipeline): source setup, transcript, and
    Claude candidate proposals. source_type은 "url"(유튜브 링크) 또는
    "file"(upload_path = 원본 영상 파일). on_status(message) is called
    before each step so the caller can surface progress through whichever
    status dict it's tracking (an edit job or AUTO_JOBS[id]). Raises on any
    failure - callers decide how to record that as an error.
    """
    loop = asyncio.get_event_loop()

    # fail fast with a clear Korean message instead of burning several
    # minutes on transcript/whisper work only to hit a cryptic English SDK
    # error ("Could not resolve authentication method...") once the
    # pipeline finally reaches a Claude call - the most common cause by far
    # is a first-run .env with ANTHROPIC_API_KEY= left blank.
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            f"ANTHROPIC_API_KEY가 설정되지 않았습니다. {REPO_ROOT / '.env'} 파일을 "
            "메모장으로 열어 ANTHROPIC_API_KEY= 뒤에 발급받은 키를 붙여넣고 저장한 "
            "뒤, 프로그램을 다시 실행해주세요."
        )

    on_status("영상 정보를 불러오는 중...")
    if source_type == "url":
        source = YouTubeSource(url)
    else:
        source = UploadedFileSource(upload_path)

    on_status("자막 추출 중 (없으면 음성인식, 몇 분 걸릴 수 있습니다)...")
    segments, transcript_source = await loop.run_in_executor(
        None, get_transcript_for_source, source, work_dir
    )

    on_status("Claude로 쇼츠 후보 분석 중...")
    clip_length = get_clip_length_preset(clip_length_key)
    candidates = await loop.run_in_executor(
        None,
        functools.partial(
            analyze.propose_candidates, segments, clip_length, tiers,
            source.title, keyword or None, counts=tier_counts,
        ),
    )
    # 2nd pass: make each candidate start at a sentence start and end once
    # the speaker has finished (best-effort, keeps the original on failure)
    on_status("쇼츠 구간의 시작/끝을 문장 단위로 다듬는 중...")
    candidates = await loop.run_in_executor(
        None, analyze.refine_boundaries, candidates, segments, clip_length,
    )

    return {
        "source": source,
        "video_title": source.title,
        "segments": segments,
        "transcript_source": transcript_source,
        # 범용 버전은 출연자/촬영일 정보를 자동으로 채우지 않는다 (편집 화면에서 원하면 직접 입력)
        "guest_label_guess": "",
        "shoot_date": "",
        "date_kind": "촬영",
        "candidates": candidates,
    }


# ---------------------------------------------------------------------
# "완전 자동 모드" (⚡): 링크나 파일을 고르면 나머지 - 후보 선정(선택한 기준과
# 개수대로), 자막 맞춤법 교정, default_assets의 고정 로고로 렌더링, 유튜브
# 비공개 업로드 - 를 편집 없이 끝까지 진행한다. 사용자가 시작해야만 돌고,
# 저절로 시작되는 것은 없다.
# ---------------------------------------------------------------------

@app.post("/auto-jobs")
async def create_auto_job(
    source_type: str = Form("url"),
    url: str = Form(""),
    file: UploadFile | None = None,
    clip_length: str = Form(DEFAULT_CLIP_LENGTH_PRESET),
    custom_seconds: str = Form(""),
    tiers: str = Form("hook,substantive"),
    keyword: str = Form(""),
    count_hook: int = Form(DEFAULT_TIER_COUNTS["hook"]),
    count_substantive: int = Form(DEFAULT_TIER_COUNTS["substantive"]),
    count_keyword: int = Form(DEFAULT_TIER_COUNTS["keyword"]),
):
    if source_type not in ("url", "file"):
        raise HTTPException(400, "source_type must be 'url' or 'file'")
    clip_key, selected, counts, keyword = _parse_options(
        clip_length, custom_seconds, tiers, keyword, count_hook, count_substantive, count_keyword,
    )
    if source_type == "url" and not url.strip():
        raise HTTPException(400, "유튜브 링크를 입력해 주세요.")
    if source_type == "file" and (file is None or not file.filename):
        raise HTTPException(400, "원본 영상 파일을 선택해 주세요.")

    auto_job_id = uuid.uuid4().hex[:12]
    work_dir = _auto_job_work_dir(auto_job_id)
    work_dir.mkdir(parents=True, exist_ok=True)
    upload_path = _save_upload(file, work_dir) if source_type == "file" else None

    AUTO_JOBS[auto_job_id] = {"status": "starting", "message": "", "video_title": "", "items": []}
    asyncio.create_task(
        _run_auto_pipeline(
            auto_job_id, source_type, url.strip(), upload_path, clip_key, selected, keyword, counts,
        )
    )
    return {"auto_job_id": auto_job_id}


async def _run_auto_pipeline(
    auto_job_id: str,
    source_type: str,
    url: str,
    upload_path: Path | None,
    clip_length_key: str,
    tiers: list[str],
    keyword: str,
    tier_counts: dict[str, int],
) -> None:
    """Runs the full unattended pipeline (candidates -> per-candidate
    build/correct/render/upload) for one source, writing live progress into
    AUTO_JOBS[auto_job_id] and a record of each successful upload into the
    library store (see library.py - only used for the "오늘 N개" count)."""
    auto_job = AUTO_JOBS[auto_job_id]
    work_dir = _auto_job_work_dir(auto_job_id)
    loop = asyncio.get_event_loop()
    try:
        auto_job.update(status="running", message="")
        result = await _prepare_source_and_candidates(
            work_dir, source_type, url, upload_path, clip_length_key,
            tiers, keyword, tier_counts, on_status=lambda msg: auto_job.update(message=msg),
        )
        source = result["source"]
        segments = result["segments"]
        guest_label_guess = result["guest_label_guess"]
        shoot_date = result["shoot_date"]
        date_kind = result["date_kind"]
        candidates = result["candidates"]
        auto_job["video_title"] = result["video_title"]
        source_label = result["video_title"]

        if not candidates:
            raise RuntimeError("조건에 맞는 쇼츠 후보를 찾지 못했습니다.")

        selected = candidates
        auto_job["items"] = [
            {"status": "pending", "title": c.title, "message": "", "youtube_url": None}
            for c in selected
        ]
        auto_job.update(status="processing", message=f"{len(selected)}개 항목 자동 처리 중...")

        for i, candidate in enumerate(selected):
            item = auto_job["items"][i]
            item_dir = work_dir / f"item_{i}"
            item_dir.mkdir(parents=True, exist_ok=True)
            try:
                item.update(status="building", message="구간 추출/이어붙이는 중...")
                # cuts are moved into the nearest audio pauses, so captions
                # must follow the ranges actually used, not the requested ones
                clip_path, used_ranges = await loop.run_in_executor(
                    None, clipbuild.build_clip_snapped, source, candidate.ranges, item_dir / "clip.mp4"
                )

                item.update(message="자막 맞춤법 교정 중...")
                cues = render.captions_for_ranges(segments, used_ranges)
                corrected_texts = await loop.run_in_executor(
                    None, analyze.correct_caption_errors, [c[2] for c in cues]
                )
                cues = [(s, e, t) for (s, e, _old), t in zip(cues, corrected_texts)]

                item.update(status="rendering", message="최종 렌더링 중...")
                assets = render.RenderAssets(
                    thumbnail_text=candidate.thumbnail_text,
                    guest_label=guest_label_guess,
                    shoot_date=shoot_date,
                    date_kind=date_kind,
                    logo_top_left=_find_default_asset("logo_left"),
                    logo_top_right=_find_default_asset("logo_right"),
                    banner_bottom=_find_default_asset("banner_bottom"),
                )
                final_path = await loop.run_in_executor(
                    None,
                    functools.partial(
                        render.render_final, clip_path, cues, assets, item_dir / "final.mp4",
                        single_camera=True,
                    ),
                )

                item.update(status="uploading", message="유튜브에 비공개로 업로드 중...")
                title_line1 = editjob.title_from_thumbnail(candidate.thumbnail_text, candidate.title)
                description = candidate.summary
                if shoot_date:
                    description += f"\n[영상 {date_kind} 날짜: {shoot_date}]"
                upload_title = title_line1
                upload_result = await loop.run_in_executor(
                    None, youtube_upload.upload_short, final_path, upload_title, description,
                    YOUTUBE_CLIENT_SECRET_FILE, YOUTUBE_TOKEN_FILE, "private",
                )

                if PLAYLIST_ID:
                    try:
                        await loop.run_in_executor(
                            None, youtube_upload.add_to_playlist, upload_result.video_id,
                            PLAYLIST_ID, YOUTUBE_CLIENT_SECRET_FILE, YOUTUBE_TOKEN_FILE,
                        )
                    except Exception:  # noqa: BLE001 - the upload itself already succeeded; don't fail the item over this
                        traceback.print_exc()

                library.add_item(WORK_DIR, library.new_item(
                    source_label=source_label, title=title_line1, summary=candidate.summary,
                    thumbnail_text=candidate.thumbnail_text, video_id=upload_result.video_id,
                    youtube_url=upload_result.url,
                ))
                item.update(status="done", message="완료", youtube_url=upload_result.url)
            except Exception as e:  # noqa: BLE001 - one item failing shouldn't stop the rest
                traceback.print_exc()
                item.update(status="error", message=str(e))

        failed = sum(1 for it in auto_job["items"] if it["status"] == "error")
        succeeded = len(auto_job["items"]) - failed
        auto_job.update(
            status="done",
            message=f"완료: {succeeded}개 업로드 성공" + (f", {failed}개 실패" if failed else ""),
        )
    except Exception as e:  # noqa: BLE001 - setup-level failure, before any items ran
        traceback.print_exc()
        auto_job.update(status="error", message=str(e))


@app.get("/auto-jobs/{auto_job_id}")
def auto_job_status(auto_job_id: str):
    auto_job = AUTO_JOBS.get(auto_job_id)
    if auto_job is None:
        raise HTTPException(404, "unknown auto job")
    return {
        "status": auto_job["status"],
        "message": auto_job["message"],
        "video_title": auto_job.get("video_title", ""),
        "items": auto_job.get("items", []),
    }


@app.get("/auto-jobs/{auto_job_id}/view", response_class=HTMLResponse)
def auto_job_view(request: Request, auto_job_id: str):
    if auto_job_id not in AUTO_JOBS:
        raise HTTPException(404, "unknown auto job")
    return templates.TemplateResponse(request, "auto_status.html", {"auto_job_id": auto_job_id})


@app.on_event("startup")
async def _tidy_up() -> None:
    # jobs left half-done by a previous run can't continue, and finished
    # jobs' big intermediate videos (원본 업로드 파일 포함) are dropped after a week
    editjob.recover_interrupted(WORK_DIR)
    editjob.cleanup_old(WORK_DIR)
    editjob.cleanup_old_dirs(WORK_DIR / "auto_jobs")
