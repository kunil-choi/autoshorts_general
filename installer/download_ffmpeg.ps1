# Downloads a static Windows ffmpeg build and extracts ffmpeg.exe/ffprobe.exe
# into installer\ffmpeg_bin\, for build.bat to bundle next to the built exe.
#
# If this URL 404s (BtbN occasionally renames release assets), grab the
# latest "ffmpeg-master-latest-win64-gpl.zip" yourself from
# https://github.com/BtbN/FFmpeg-Builds/releases and drop ffmpeg.exe +
# ffprobe.exe directly into installer\ffmpeg_bin\ instead of running this.
#
# Kept plain-ASCII on purpose - Windows PowerShell 5.1 guesses a .ps1
# file's encoding from the system's legacy code page when there's no BOM,
# so any non-ASCII text here risks the same kind of corruption a non-UTF-8
# read of a UTF-8 file causes elsewhere.

$ErrorActionPreference = "Stop"
# Invoke-WebRequest's default per-byte progress-bar rendering is known to
# make large downloads dramatically slower in Windows PowerShell 5.1 -
# disabling it just turns off that UI, the download itself is unaffected.
$ProgressPreference = "SilentlyContinue"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$binDir = Join-Path $scriptDir "ffmpeg_bin"

if ((Test-Path (Join-Path $binDir "ffmpeg.exe")) -and (Test-Path (Join-Path $binDir "ffprobe.exe"))) {
    Write-Host "ffmpeg.exe/ffprobe.exe already present in ffmpeg_bin - skipping download."
    exit 0
}

New-Item -ItemType Directory -Force -Path $binDir | Out-Null
$zipPath = Join-Path $scriptDir "ffmpeg-download.zip"
$url = "https://github.com/BtbN/FFmpeg-Builds/releases/latest/download/ffmpeg-master-latest-win64-gpl.zip"

Write-Host "Downloading ffmpeg from: $url"
Write-Host "(this is roughly 100 MB - it can take a few minutes depending on your connection)"
try {
    # curl.exe ships built into Windows 10 (1803+) and Windows 11, is fast,
    # and shows a real progress bar - unlike Invoke-WebRequest, whose own
    # progress bar (left on) makes large downloads dramatically slower in
    # Windows PowerShell 5.1, which is why $ProgressPreference is silenced
    # above for the fallback path below. Prefer curl.exe when present so
    # the download doesn't look frozen with nothing on screen.
    $curlExe = Get-Command curl.exe -ErrorAction SilentlyContinue
    if ($curlExe) {
        # curl's progress bar writes to stderr by design - with
        # $ErrorActionPreference = "Stop" (set above) PowerShell treats ANY
        # stderr output from a native command as a script-terminating error,
        # which would abort a perfectly successful download. Relax it just
        # for this call and check $LASTEXITCODE ourselves instead.
        $prevEap = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        & curl.exe -L --progress-bar -o $zipPath $url
        $ErrorActionPreference = $prevEap
        if ($LASTEXITCODE -ne 0) {
            throw "curl.exe exited with code $LASTEXITCODE"
        }
    } else {
        Write-Host "(curl.exe not found - falling back to a silent download, no progress bar will be shown; this is normal, just wait)"
        Invoke-WebRequest -Uri $url -OutFile $zipPath
    }
} catch {
    Write-Error "Failed to download ffmpeg. The URL above may no longer be valid - get the latest win64-gpl zip yourself from https://github.com/BtbN/FFmpeg-Builds/releases and put ffmpeg.exe/ffprobe.exe into $binDir"
    exit 1
}

$extractDir = Join-Path $scriptDir "ffmpeg-extract-tmp"
if (Test-Path $extractDir) { Remove-Item -Recurse -Force $extractDir }
Expand-Archive -Path $zipPath -DestinationPath $extractDir

$ffmpegExe = Get-ChildItem -Path $extractDir -Recurse -Filter "ffmpeg.exe" | Select-Object -First 1
$ffprobeExe = Get-ChildItem -Path $extractDir -Recurse -Filter "ffprobe.exe" | Select-Object -First 1

if (-not $ffmpegExe -or -not $ffprobeExe) {
    Write-Error "Could not find ffmpeg.exe/ffprobe.exe inside the downloaded archive - check $extractDir manually."
    exit 1
}

Copy-Item $ffmpegExe.FullName (Join-Path $binDir "ffmpeg.exe") -Force
Copy-Item $ffprobeExe.FullName (Join-Path $binDir "ffprobe.exe") -Force

Remove-Item $zipPath -Force
Remove-Item -Recurse -Force $extractDir

Write-Host "Done: ffmpeg.exe / ffprobe.exe ready in $binDir"
