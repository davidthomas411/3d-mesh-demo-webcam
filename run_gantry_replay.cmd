@echo off
rem The live page on a recorded gantry sweep instead of the camera (tools/gantry_replay.py makes the two files).
rem Then open  http://127.0.0.1:8096/?bolt=1^&video=/gantry-data/replay.mp4
setlocal
if not defined GANTRY_WEB set "GANTRY_WEB=%~dp0..\recordings\gantry\web"
set "CLOUD_REPLAY=%GANTRY_WEB%\replay.npz"
if not defined PORT set PORT=8096
call "%~dp0run_local.cmd" %*
