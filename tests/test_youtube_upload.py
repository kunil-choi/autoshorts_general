from pathlib import Path
from unittest.mock import MagicMock

import pytest
from googleapiclient.errors import HttpError

from autoshorts import youtube_upload


def _client_raising(status: int) -> MagicMock:
    resp = MagicMock(status=status, reason="x")
    client = MagicMock()
    client.videos().delete().execute.side_effect = HttpError(resp, b"{}")
    return client


def test_delete_video_treats_already_deleted_as_success(monkeypatch):
    monkeypatch.setattr(youtube_upload, "_youtube_client", lambda *a: _client_raising(404))
    youtube_upload.delete_video("gone", Path("cs.json"), Path("tok.json"))  # no raise


def test_delete_video_still_raises_other_errors(monkeypatch):
    monkeypatch.setattr(youtube_upload, "_youtube_client", lambda *a: _client_raising(403))
    with pytest.raises(HttpError):
        youtube_upload.delete_video("v", Path("cs.json"), Path("tok.json"))


def test_get_privacy_statuses_batches_by_50_and_omits_missing(monkeypatch):
    client = MagicMock()
    client.videos().list().execute.side_effect = [
        {"items": [{"id": f"v{i}", "status": {"privacyStatus": "public"}} for i in range(50)]},
        {"items": [{"id": "v50", "status": {"privacyStatus": "private"}}]},
    ]
    monkeypatch.setattr(youtube_upload, "_youtube_client", lambda *a: client)

    ids = [f"v{i}" for i in range(52)]  # v51 is not returned (e.g. deleted)
    result = youtube_upload.get_privacy_statuses(ids, Path("cs.json"), Path("tok.json"))

    assert result["v0"] == "public" and result["v50"] == "private"
    assert "v51" not in result
    assert client.videos().list().execute.call_count == 2


def _fake_creds(*, valid, expired, refresh_error=None, json_text='{"t": 1}'):
    creds = MagicMock()
    creds.valid = valid
    creds.expired = expired
    creds.refresh_token = "rt"
    creds.to_json.return_value = json_text
    if refresh_error:
        creds.refresh.side_effect = refresh_error
    return creds


def test_get_credentials_relogs_in_when_refresh_token_is_revoked(monkeypatch, tmp_path):
    """invalid_grant(토큰 만료·취소)여도 오류로 끝내지 않고 브라우저 로그인으로 새 토큰을 받는다."""
    from google.auth.exceptions import RefreshError
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow

    secret = tmp_path / "client_secret.json"
    secret.write_text("{}")
    token = tmp_path / "youtube_token.json"
    token.write_text("{}")

    stale = _fake_creds(valid=False, expired=True, refresh_error=RefreshError("invalid_grant"))
    fresh = _fake_creds(valid=True, expired=False, json_text='{"new": true}')
    flow = MagicMock()
    flow.run_local_server.return_value = fresh
    monkeypatch.setattr(Credentials, "from_authorized_user_file", lambda *a, **k: stale)
    monkeypatch.setattr(InstalledAppFlow, "from_client_secrets_file", lambda *a, **k: flow)

    result = youtube_upload.get_credentials(secret, token)

    assert result is fresh
    flow.run_local_server.assert_called_once_with(port=0)
    assert token.read_text() == '{"new": true}'


def test_get_credentials_refreshes_silently_without_login(monkeypatch, tmp_path):
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow

    secret = tmp_path / "client_secret.json"
    secret.write_text("{}")
    token = tmp_path / "youtube_token.json"
    token.write_text("{}")

    expired = _fake_creds(valid=False, expired=True, json_text='{"refreshed": true}')
    monkeypatch.setattr(Credentials, "from_authorized_user_file", lambda *a, **k: expired)
    login = MagicMock(side_effect=AssertionError("정상 갱신인데 로그인 창이 뜸"))
    monkeypatch.setattr(InstalledAppFlow, "from_client_secrets_file", login)

    result = youtube_upload.get_credentials(secret, token)

    assert result is expired
    expired.refresh.assert_called_once()
    login.assert_not_called()
    assert token.read_text() == '{"refreshed": true}'
