@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0\.."

echo === Step 0/5: checking Python ===
python --version >nul 2>nul
if errorlevel 1 (
    echo Python was not found - or "python" is just the Windows Store shortcut.
    echo Install real Python 3.11+ from https://www.python.org/downloads/
    echo and check "Add python.exe to PATH" during setup, then close this
    echo window, open a new one, and run this script again.
    pause
    exit /b 1
)

echo === Step 1/5: setting up virtual environment ===
if not exist ".venv" (
    python -m venv .venv
    if errorlevel 1 goto :error
)
call .venv\Scripts\activate.bat

echo === Step 2/5: installing packages (requirements.txt + pyinstaller) ===
pip install -r requirements.txt pyinstaller
if errorlevel 1 goto :error

echo === Step 3/5: downloading ffmpeg ===
powershell -ExecutionPolicy Bypass -File "installer\download_ffmpeg.ps1"
if errorlevel 1 goto :error

echo === Step 4/5: building with PyInstaller ===
REM save files that only live inside dist\autoshorts, not the source tree,
REM before wiping it below - a rebuild would otherwise silently delete the
REM API key, the YouTube upload credentials, and any department logo/banner
REM images dropped into default_assets\, forcing them all to be redone
REM every single time.
set "BACKUP_DIR=%TEMP%\autoshorts_build_backup_%RANDOM%"
if exist "%BACKUP_DIR%" rmdir /s /q "%BACKUP_DIR%"
mkdir "%BACKUP_DIR%"
if exist "dist\autoshorts\.env" copy /Y "dist\autoshorts\.env" "%BACKUP_DIR%\.env" >nul
if exist "dist\autoshorts\client_secret.json" copy /Y "dist\autoshorts\client_secret.json" "%BACKUP_DIR%\client_secret.json" >nul
if exist "dist\autoshorts\youtube_token.json" copy /Y "dist\autoshorts\youtube_token.json" "%BACKUP_DIR%\youtube_token.json" >nul
if exist "dist\autoshorts\default_assets" xcopy "dist\autoshorts\default_assets" "%BACKUP_DIR%\default_assets\" /E /I /Y >nul
REM work\ holds the shorts library history and in-progress edit jobs -
REM wiping it would lose them.
if exist "dist\autoshorts\work" xcopy "dist\autoshorts\work" "%BACKUP_DIR%\work\" /E /I /Y >nul

REM clear both the output AND PyInstaller's intermediate cache - a fresh
REM zip re-download can give files timestamps that don't reliably increase
REM from the last build, which can fool PyInstaller's mtime-based cache
REM into reusing stale compiled code from the "build" folder even though
REM the source actually changed. Only clearing "dist" (as this used to)
REM isn't enough to guarantee a truly clean build.
if exist "dist\autoshorts" rmdir /s /q "dist\autoshorts"
if exist "build" rmdir /s /q "build"
pyinstaller installer\autoshorts.spec --distpath dist --workpath build --noconfirm
if errorlevel 1 goto :error

echo === Step 5/5: copying ffmpeg, .env.example, and default_assets next to the exe ===
xcopy installer\ffmpeg_bin dist\autoshorts\ffmpeg_bin /E /I /Y
copy /Y webui\.env.example dist\autoshorts\.env.example
xcopy default_assets dist\autoshorts\default_assets /E /I /Y

if exist "%BACKUP_DIR%\.env" (
    copy /Y "%BACKUP_DIR%\.env" "dist\autoshorts\.env" >nul
    echo Existing .env (API key etc.) carried over from the previous build.
)
if exist "%BACKUP_DIR%\client_secret.json" (
    copy /Y "%BACKUP_DIR%\client_secret.json" "dist\autoshorts\client_secret.json" >nul
    echo Existing client_secret.json carried over from the previous build.
)
if exist "%BACKUP_DIR%\youtube_token.json" (
    copy /Y "%BACKUP_DIR%\youtube_token.json" "dist\autoshorts\youtube_token.json" >nul
    echo Existing youtube_token.json carried over - no YouTube re-login needed.
)
if exist "%BACKUP_DIR%\default_assets" (
    xcopy "%BACKUP_DIR%\default_assets" "dist\autoshorts\default_assets\" /E /I /Y >nul
    echo Existing default_assets images carried over from the previous build.
)
if exist "%BACKUP_DIR%\work" (
    xcopy "%BACKUP_DIR%\work" "dist\autoshorts\work\" /E /I /Y >nul
    echo Existing work folder - upload records and edit jobs - carried over.
)
rmdir /s /q "%BACKUP_DIR%"

echo.
echo ============================================================
echo Done! dist\autoshorts\ is the finished, distributable build.
echo  - To run it: double-click dist\autoshorts\autoshorts.exe
echo  - To share it: zip the whole dist\autoshorts\ folder and send it -
echo    no Python or ffmpeg install needed on the other machine.
echo ============================================================
pause
exit /b 0

:error
echo.
echo Build failed. Check the log above for the error.
pause
exit /b 1
