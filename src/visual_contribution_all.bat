@echo off
setlocal enabledelayedexpansion

REM C3Dファイルがあるフォルダを指定
set "DATASET_DIR=..\dataset"

REM 全ての .c3d ファイルを順に処理
for %%f in (%DATASET_DIR%\*.c3d) do (
    echo Running scripts for: %%f
    python visual_contribution_first.py %%f
    python visual_contribution_second.py %%f
    python visual_contribution_geodesic.py %%f
)

echo All processing complete.
pause