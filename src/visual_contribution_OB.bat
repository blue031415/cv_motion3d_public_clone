@echo off
setlocal enabledelayedexpansion

REM 出力ログファイルを初期化（BOM付きUTF-8で）
powershell -Command "[System.IO.File]::WriteAllText('log.txt', '', [System.Text.Encoding]::UTF8)"

REM データセットディレクトリ
set "DATASET_DIR=..\dataset\OpenBiomechanics"

REM .c3dファイルを再帰的に探索
for /r "%DATASET_DIR%" %%f in (*.c3d) do (
    REM "model.c3d" で終わるファイルはスキップ
    echo %%f | findstr /i "model.c3d$" >nul
    if not !errorlevel! == 0 (
        echo Running scripts for: %%f
        python visual_contribution_first.py %%f 2>> log.txt
        python visual_contribution_second.py %%f 2>> log.txt
        python visual_contribution_geodesic.py %%f 2>> log.txt
    ) else (
        echo Skipping model.c3d: %%f
    )
)

echo All processing complete. See log.txt for any errors.
pause
