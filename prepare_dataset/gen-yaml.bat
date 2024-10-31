@echo off

:: Get the directory where the script is located
set SCRIPT_DIR=%~dp0
set PARENT_DIR=%SCRIPT_DIR:~0,-1%
for %%i in ("%PARENT_DIR%") do set "PARENT_DIR=%%~dpi"

:: Define paths relative to the script location
set TRAIN_PATH=%PARENT_DIR%dataset\images\train
set VAL_PATH=%PARENT_DIR%dataset\images\val
set CONFIG_PATH=%PARENT_DIR%dataset\opentt.yaml

:: Print the paths for verification
echo Train Path: %TRAIN_PATH%
echo Validation Path: %VAL_PATH%
echo Config Path: %CONFIG_PATH%

:: Number of classes
set NC=1

:: Class names
set NAMES=ball

:: Output details
echo Number of classes: %NC%
echo Class names: %NAMES%

:: Write the configuration to opentt.yaml
echo # OpenTT Configuration File > %CONFIG_PATH%
echo. >> %CONFIG_PATH%
echo train: %TRAIN_PATH% >> %CONFIG_PATH%
echo val: %VAL_PATH% >> %CONFIG_PATH%
echo. >> %CONFIG_PATH%
echo # Number of classes (adjust according to your dataset) >> %CONFIG_PATH%
echo nc: %NC% >> %CONFIG_PATH%
echo. >> %CONFIG_PATH%
echo # Class names >> %CONFIG_PATH%
echo names: >> %CONFIG_PATH%
echo   0: %NAMES% >> %CONFIG_PATH%

:: Confirm file creation
echo Configuration file created at %CONFIG_PATH%
