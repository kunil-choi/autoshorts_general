"""Entry point for the frozen (PyInstaller) build - double-clicking the
built .exe runs this. Not used in normal dev mode (`uvicorn webui.server:app`).

Responsibilities the plain dev command doesn't need to worry about:
  - put the bundled ffmpeg/ffprobe on PATH (see installer/build.bat, which
    downloads them into ffmpeg_bin/ next to the built exe)
  - default AUTOSHORTS_DISPLAY_FONT to Windows' built-in Malgun Gothic Bold
    if nothing else is set - Windows ships this on virtually every install,
    so bundling a font file (and its licensing questions) isn't necessary
  - scaffold a first-run .env from .env.example so a non-technical worker
    has an obvious file to open and paste their API key into
  - open the browser automatically, since there's no terminal-savvy user
    expected to type the localhost URL themselves
"""

from __future__ import annotations

import os
import platform
import shutil
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path

PORT = 8787


def _app_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def _setup_ffmpeg_path(app_dir: Path) -> None:
    ffmpeg_dir = app_dir / "ffmpeg_bin"
    if ffmpeg_dir.is_dir():
        os.environ["PATH"] = str(ffmpeg_dir) + os.pathsep + os.environ.get("PATH", "")


def _setup_default_font() -> None:
    if os.environ.get("AUTOSHORTS_DISPLAY_FONT"):
        return
    if platform.system() == "Windows":
        malgun = Path(r"C:\Windows\Fonts\malgunbd.ttf")
        if malgun.is_file():
            os.environ["AUTOSHORTS_DISPLAY_FONT"] = str(malgun).replace("\\", "/")


def _scaffold_env(app_dir: Path) -> None:
    env_path = app_dir / ".env"
    example_path = app_dir / ".env.example"
    if not env_path.exists() and example_path.exists():
        shutil.copyfile(example_path, env_path)
        print(f"[autoshorts] {env_path} 파일을 새로 만들었습니다.")
        print("[autoshorts] 이 파일을 열어 ANTHROPIC_API_KEY를 채운 뒤 다시 실행해주세요.")


def _scaffold_default_assets(app_dir: Path) -> None:
    # defense-in-depth alongside build.bat's own copy step - makes sure the
    # folder (and its README explaining what to drop in) exists even if a
    # worker deleted it, or ran a build that predates this feature.
    assets_dir = app_dir / "default_assets"
    assets_dir.mkdir(exist_ok=True)
    readme_path = assets_dir / "README.txt"
    if not readme_path.exists():
        readme_path.write_text(
            "이 폴더의 이미지는 만들어지는 쇼츠에 자동으로 삽입됩니다 (모두 선택 사항).\n\n"
            "화면 배치: 좌측 상단 = logo_left, 우측 상단 = logo_right, 중앙 하단(크게) = banner_bottom\n\n"
            "   - logo_left.png (또는 .jpg) : 좌측 상단 로고\n"
            "   - logo_right.png (또는 .jpg) : 우측 상단 로고\n"
            "   - banner_bottom.png (또는 .jpg) : 중앙 하단 큰 로고\n\n"
            "파일이 없으면 해당 위치는 그냥 비워둔 채로 렌더링됩니다. 나중에 파일을\n"
            "추가/교체해도 프로그램을 다시 빌드할 필요 없이 바로 다음 작업부터 적용됩니다.\n",
            encoding="utf-8",
        )


def _wait_for_server(port: int, timeout: float = 20.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.3)
    return False


def _open_browser_when_ready() -> None:
    url = f"http://127.0.0.1:{PORT}"
    if not _wait_for_server(PORT):
        print(f"[autoshorts] 서버 시작이 너무 오래 걸립니다. 브라우저 주소창에 아래 주소를 직접 입력해주세요:")
        print(f"[autoshorts] {url}")
        return
    # open_new (rather than open) asks for a genuinely new window instead of
    # possibly reusing/backgrounding an existing browser window, which is
    # what "새 창이 안 떠요" (already-open browser, no visible new window)
    # usually turns out to be - but on some Windows setups even this can
    # silently no-op, so always print a manual fallback too.
    opened = webbrowser.open_new(url)
    if not opened:
        print(f"[autoshorts] 브라우저를 자동으로 열지 못했습니다. 아래 주소를 직접 브라우저 주소창에 입력해주세요:")
        print(f"[autoshorts] {url}")


def main() -> None:
    app_dir = _app_dir()
    os.chdir(app_dir)  # so work/ (job files) lands next to the exe, not wherever it was launched from

    _setup_ffmpeg_path(app_dir)
    _setup_default_font()
    _scaffold_env(app_dir)
    _scaffold_default_assets(app_dir)

    import uvicorn

    from webui.server import app as fastapi_app  # this also loads app_dir/.env

    if not os.environ.get("ANTHROPIC_API_KEY"):
        # print this up front, at startup, rather than only after a worker
        # waits through transcript/whisper work and hits the SDK's cryptic
        # English error ("Could not resolve authentication method...") deep
        # inside a Claude call - the most common cause is simply a first-run
        # .env with ANTHROPIC_API_KEY= left blank.
        print(f"[autoshorts] 경고: ANTHROPIC_API_KEY가 설정되지 않았습니다.")
        print(f"[autoshorts] {app_dir / '.env'} 파일을 메모장으로 열어 ANTHROPIC_API_KEY= 뒤에 키를 붙여넣고 저장한 뒤 다시 실행해주세요.")

    threading.Thread(target=_open_browser_when_ready, daemon=True).start()

    print("autoshorts 서버를 시작합니다...")
    print(f"잠시 후 브라우저가 자동으로 열립니다 (http://127.0.0.1:{PORT}).")
    print("이 창을 닫으면 서버가 종료됩니다.")
    uvicorn.run(fastapi_app, host="127.0.0.1", port=PORT)


if __name__ == "__main__":
    main()
