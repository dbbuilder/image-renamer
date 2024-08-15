@echo off
REM This batch file activates the venv environment and runs the image_renamer.py script

REM Set the path to the virtual environment activation script
set VENV_PATH="c:\dev\imagerenamer\venv\Scripts\activate.bat"

REM Set the path to the Python script
set SCRIPT_PATH="c:\dev\imagerenamer\imagerenamer.py"

REM Set the target directory for the images
set TARGET_DIR =%USERPROFILE%\Downloads

REM Activate the virtual environment
call "%VENV_PATH%"

REM Run the Python script with the target directory as an argument
python "%SCRIPT_PATH%" %USERPROFILE%\Downloads

REM Deactivate the virtual environment (optional)
deactivate

REM Pause the command line so the user can see the output
pause
