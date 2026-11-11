"""Persistent record of every short the automatic pipeline has uploaded -
whether made through the staged flow or the "완전 자동" button - so a worker can
come back later (even after an app restart) and publish or delete each one
from webui/server.py's /library screen instead of using YouTube Studio.

Plain JSON on disk (work/library.json) rather than anything heavier: this
is a department tool producing on the order of ~10 items/day, not a
high-volume system. A single threading.Lock is enough to keep concurrent
writes (the asyncio event loop thread and any run_in_executor thread) from
corrupting the file, since this is one single-process app.
"""

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

_lock = threading.Lock()


@dataclass
class LibraryItem:
    id: str
    source_label: str  # program name (scheduled) or video title (manual)
    title: str
    summary: str
    thumbnail_text: str
    video_id: str
    youtube_url: str
    created_at: str  # ISO 8601, UTC
    status: str = "private"  # "private" | "public" | "deleted"


def new_item(
    source_label: str, title: str, summary: str, thumbnail_text: str, video_id: str, youtube_url: str,
) -> LibraryItem:
    return LibraryItem(
        id=uuid.uuid4().hex[:12],
        source_label=source_label,
        title=title,
        summary=summary,
        thumbnail_text=thumbnail_text,
        video_id=video_id,
        youtube_url=youtube_url,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def _path(work_dir: Path) -> Path:
    return work_dir / "library.json"


def _load(work_dir: Path) -> list[dict]:
    path = _path(work_dir)
    if not path.is_file():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def _save(work_dir: Path, items: list[dict]) -> None:
    path = _path(work_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")


def add_item(work_dir: Path, item: LibraryItem) -> None:
    with _lock:
        items = _load(work_dir)
        items.append(asdict(item))
        _save(work_dir, items)


def list_items(work_dir: Path) -> list[dict]:
    with _lock:
        items = _load(work_dir)
    return sorted(items, key=lambda it: it["created_at"], reverse=True)


def update_status(work_dir: Path, item_id: str, status: str) -> dict | None:
    with _lock:
        items = _load(work_dir)
        for it in items:
            if it["id"] == item_id:
                it["status"] = status
                _save(work_dir, items)
                return it
    return None


def get_item(work_dir: Path, item_id: str) -> dict | None:
    with _lock:
        items = _load(work_dir)
    for it in items:
        if it["id"] == item_id:
            return it
    return None


def mark_public_from_youtube(work_dir: Path, youtube_statuses: dict[str, str]) -> int:
    """Move still-"private" items whose real YouTube privacyStatus is
    "public" (the worker published them in Studio instead of here) to
    "public". Returns how many changed. Items YouTube didn't report on are
    left alone - absence can mean deleted, or just a different signed-in
    account, so it's never treated as a reason to change anything."""
    changed = 0
    with _lock:
        items = _load(work_dir)
        for it in items:
            if it["status"] == "private" and youtube_statuses.get(it["video_id"]) == "public":
                it["status"] = "public"
                changed += 1
        if changed:
            _save(work_dir, items)
    return changed
