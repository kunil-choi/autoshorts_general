"""Upload a rendered short to YouTube via the YouTube Data API v3, used by
the fully-automatic work path (webui/server.py's /auto-jobs) to publish straight through without a worker manually using
Studio - plus the handful of other calls webui/server.py's in-app "공개
전환" library screen needs to manage what got uploaded (checking status,
publishing, deleting, adding to a playlist) without leaving the app either.

Requires a Google Cloud OAuth client (client_secret.json) for a project
with the YouTube Data API v3 enabled, placed next to the app's .env. The
first call on a machine opens a real browser window for whoever's running
it to sign into the department's YouTube channel once (installed-app OAuth
flow - this is a local desktop tool, same as the rest of the app); after
that, a refresh token is cached on disk and reused silently.

SCOPES covers full video management (upload/update/delete/playlists), not
just upload - if youtube_token.json was created by an older build that only
ever requested the upload-only scope, delete that file once to force a
fresh consent screen covering the wider scope; a cached token can't silently
gain scopes it wasn't originally granted.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SCOPES = ["https://www.googleapis.com/auth/youtube"]


@dataclass(frozen=True)
class UploadResult:
    video_id: str
    url: str


def get_credentials(client_secret_path: Path, token_path: Path):
    """Load cached credentials, refreshing or running the one-time browser
    consent flow as needed. Raises RuntimeError with a Korean, actionable
    message if client_secret_path is missing - there's no way to proceed
    without it, and the raw google-auth error for a missing file isn't
    helpful to a non-technical worker.
    """
    from google.auth.exceptions import RefreshError
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow

    if not client_secret_path.is_file():
        raise RuntimeError(
            f"유튜브 업로드용 인증 파일을 찾을 수 없습니다: {client_secret_path}. "
            "구글 클라우드 콘솔에서 발급받은 client_secret.json 파일을 이 경로에 놓아주세요."
        )

    creds = None
    if token_path.is_file():
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
    if not creds or not creds.valid:
        refreshed = False
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
                refreshed = True
            except RefreshError:
                # 저장된 토큰이 만료·취소됐다(invalid_grant). OAuth 동의 화면이 '테스트' 상태면
                # 7일마다 이렇게 된다. 오류로 끝내지 않고 아래에서 브라우저 로그인을 다시 띄워
                # 새 토큰을 받는다 - 안 그러면 youtube_token.json을 손으로 지워야 복구된다.
                creds = None
        if not refreshed:
            flow = InstalledAppFlow.from_client_secrets_file(str(client_secret_path), SCOPES)
            creds = flow.run_local_server(port=0)
        token_path.write_text(creds.to_json())
    return creds


def _youtube_client(client_secret_path: Path, token_path: Path):
    from googleapiclient.discovery import build

    creds = get_credentials(client_secret_path, token_path)
    # static_discovery (the default when googleapiclient ships a bundled
    # discovery doc, which it does for "youtube v3") avoids an extra network
    # round-trip to Google's discovery service on every single call.
    return build("youtube", "v3", credentials=creds)


def upload_short(
    video_path: Path,
    title: str,
    description: str,
    client_secret_path: Path,
    token_path: Path,
    privacy_status: str = "private",
) -> UploadResult:
    """Upload video_path to YouTube and return its id/watch URL. Raises on
    any failure (missing credentials, API error, quota exceeded) rather
    than swallowing it - unlike analyze.py's best-effort Claude helpers,
    there's no sensible fallback for "the upload didn't happen": the
    auto-pipeline calling this needs to know so it can mark that item as
    failed instead of silently reporting success without a real upload.
    """
    from googleapiclient.http import MediaFileUpload

    youtube = _youtube_client(client_secret_path, token_path)
    body = {
        "snippet": {
            "title": title[:100],
            "description": description[:5000],
            "categoryId": "25",  # News & Politics - closest fit for broadcast clips
        },
        "status": {
            "privacyStatus": privacy_status,
            "selfDeclaredMadeForKids": False,
        },
    }
    media = MediaFileUpload(str(video_path), chunksize=-1, resumable=True, mimetype="video/mp4")
    request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)
    response = None
    while response is None:
        _status, response = request.next_chunk()
    video_id = response["id"]
    return UploadResult(video_id=video_id, url=f"https://youtu.be/{video_id}")


def set_privacy(
    video_id: str, privacy_status: str, client_secret_path: Path, token_path: Path,
) -> None:
    """Flip an already-uploaded video's privacyStatus (e.g. "private" ->
    "public") - backs the in-app library's "공개 전환" button, so a worker
    never has to go into YouTube Studio for this."""
    youtube = _youtube_client(client_secret_path, token_path)
    youtube.videos().update(
        part="status",
        body={"id": video_id, "status": {"privacyStatus": privacy_status}},
    ).execute()


def get_privacy_statuses(
    video_ids: list[str], client_secret_path: Path, token_path: Path,
) -> dict[str, str]:
    """Current privacyStatus ("private" | "unlisted" | "public") of each
    given video, looked up in one videos.list call per 50 ids (1 quota
    unit each - far cheaper than any write). Videos YouTube doesn't return
    (deleted, or not visible to the signed-in account) are simply absent
    from the result - callers must not read absence as "deleted"."""
    youtube = _youtube_client(client_secret_path, token_path)
    statuses: dict[str, str] = {}
    for i in range(0, len(video_ids), 50):
        chunk = video_ids[i:i + 50]
        response = youtube.videos().list(part="status", id=",".join(chunk), maxResults=50).execute()
        for item in response.get("items", []):
            statuses[item["id"]] = item["status"]["privacyStatus"]
    return statuses


def delete_video(video_id: str, client_secret_path: Path, token_path: Path) -> None:
    """Delete an uploaded video outright - backs the library's "삭제"
    button for shorts the worker decides not to keep.

    A video YouTube no longer has (already deleted in Studio) counts as
    successfully deleted: the end state is what the worker wanted, and
    raising would leave the item stuck in the library forever.
    """
    from googleapiclient.errors import HttpError

    youtube = _youtube_client(client_secret_path, token_path)
    try:
        youtube.videos().delete(id=video_id).execute()
    except HttpError as e:
        if e.resp.status == 404:
            return
        raise


def add_to_playlist(
    video_id: str, playlist_id: str, client_secret_path: Path, token_path: Path,
) -> None:
    """Add an already-uploaded video to a playlist on the same channel
    (e.g. a corner/series playlist) - best-effort from the caller's
    perspective (see webui/server.py), since a missing/invalid playlist id
    shouldn't block the upload itself from counting as a success."""
    youtube = _youtube_client(client_secret_path, token_path)
    youtube.playlistItems().insert(
        part="snippet",
        body={
            "snippet": {
                "playlistId": playlist_id,
                "resourceId": {"kind": "youtube#video", "videoId": video_id},
            }
        },
    ).execute()
