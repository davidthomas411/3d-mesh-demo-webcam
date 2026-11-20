@echo off
rem Build the SKEL skeleton atlas for the page (INSTALL.md, section 5b). Run from the 3d-mesh-demo-webcam folder.
rem Needs: ..\body_models\skel_models_v1.1 (SKEL model files), ..\body_models\mhr2smpl_mapping.npz, the skel Python package,
rem and ..\Fast-SAM-3D-Body\mhr_rest.npz (tools\make_mhr_rest.py).
setlocal
set "DEMO=%~dp0.."
set "FAST=%DEMO%\..\Fast-SAM-3D-Body"
set "BM=%DEMO%\..\body_models"
set "PY=%FAST%\.venv\Scripts\python.exe"
set "MHR=%FAST%\checkpoints\sam-3d-body-dinov3\assets\mhr_model.pt"
set "REST=%FAST%\mhr_rest.npz"
cd /d "%DEMO%"
if not exist "%BM%\skel_models_v1.1\skel_male.pkl" (echo Missing %BM%\skel_models_v1.1 - see INSTALL.md section 5b & exit /b 1)
if not exist "%BM%\mhr2smpl_mapping.npz" (echo Missing %BM%\mhr2smpl_mapping.npz - see INSTALL.md section 5b & exit /b 1)
if not exist "%REST%" (echo Missing %REST% - see INSTALL.md section 4 & exit /b 1)
if not exist "%BM%\skel_fit_mhr_rest.npz" (
  "%PY%" tools\fit_skel_to_mhr.py "%REST%" "%BM%\mhr2smpl_mapping.npz" "%BM%\skel_models_v1.1" "%BM%\skel_fit_mhr_rest.npz" --gender male || exit /b 1
)
"%PY%" tools\build_skel_atlas.py "%BM%\skel_fit_mhr_rest.npz" atlas\atlas "%MHR%" "%REST%" atlas\skel || exit /b 1
for /d %%S in (atlas\subjects\*) do (
  if exist "%%S\atlas.json" "%PY%" tools\build_skel_atlas.py "%BM%\skel_fit_mhr_rest.npz" "%%S\atlas" "%MHR%" "%REST%" "%%S\skel" || exit /b 1
)
echo SKEL atlas built.
