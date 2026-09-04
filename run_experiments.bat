@echo off
REM 快速运行脚本：自动化AB测试实验 (Windows版本)

echo ========================================
echo DePLNet Volume-Compensated Gating
echo AB Testing Automation Script
echo ========================================
echo.

REM 设置默认GPU
set GPU=0
if not "%1"=="" set GPU=%1

echo Using GPU: %GPU%
echo.

echo Select experiments to run:
echo 1) Full AB Test (4 models x 1 ratio)
echo 2) Multi-ratio Test (1 model x 4 ratios)
echo 3) Single Experiment (custom)
echo.

set /p choice="Enter choice [1-3]: "

if "%choice%"=="1" goto ab_test
if "%choice%"=="2" goto multi_ratio
if "%choice%"=="3" goto single_exp
goto invalid

:ab_test
echo.
echo Running Full AB Test (20%% labeled data)
echo.

call :run_exp MyNet_MoE_VC 20 %GPU%
call :run_exp MyNet_MoE 20 %GPU%
call :run_exp MyNet_NoMoE 20 %GPU%
call :run_exp MyNet_NoDSP 20 %GPU%

goto end

:multi_ratio
echo.
echo Running Multi-ratio Test (MyNet_MoE_VC)
echo.

call :run_exp MyNet_MoE_VC 5 %GPU%
call :run_exp MyNet_MoE_VC 10 %GPU%
call :run_exp MyNet_MoE_VC 20 %GPU%
call :run_exp MyNet_MoE_VC 100 %GPU%

goto end

:single_exp
echo.
echo Custom Experiment
echo Available models:
echo   1) MyNet_MoE_VC (Full model)
echo   2) MyNet_MoE (No VC gating)
echo   3) MyNet_NoMoE (No MoE)
echo   4) MyNet_NoDSP (No DSP)
echo.

set /p model_choice="Select model [1-4]: "

if "%model_choice%"=="1" set MODEL=MyNet_MoE_VC
if "%model_choice%"=="2" set MODEL=MyNet_MoE
if "%model_choice%"=="3" set MODEL=MyNet_NoMoE
if "%model_choice%"=="4" set MODEL=MyNet_NoDSP

set /p RATIO="Enter labeled rate [5/10/20/100]: "

call :run_exp %MODEL% %RATIO% %GPU%

goto end

:run_exp
set NET_TYPE=%1
set LABELED_RATE=%2
set GPU_ID=%3

echo.
echo ========================================
echo Running Experiment
echo Network: %NET_TYPE%
echo Labeled Rate: %LABELED_RATE%%%
echo ========================================
echo.

REM 检查数据文件
if not exist "data\brats2021_%LABELED_RATE%percent_fold_1.json" (
    echo Error: Data split file not found!
    echo Please run: python data/generate_splits.py --ratios %LABELED_RATE%
    exit /b 1
)

REM 更新config.py
python -c "import re; content = open('config.py').read(); content = re.sub(r'config\[\"net\"\] = \".*?\"', 'config[\"net\"] = \"%NET_TYPE%\"', content); content = re.sub(r'config\[\"labeled_rate\"\] = \d+', 'config[\"labeled_rate\"] = %LABELED_RATE%', content); open('config.py', 'w').write(content); print('Config updated: net=%NET_TYPE%, labeled_rate=%LABELED_RATE%%%')"

REM 运行训练
echo Starting training...
python train.py --gpu %GPU_ID%

if errorlevel 1 (
    echo Training failed!
    exit /b 1
)

echo Training completed successfully!

REM 运行测试
echo.
echo Running tests for all folds...
for %%f in (1 2 3 4 5) do (
    echo Testing fold %%f...
    python test_improved.py --gpu %GPU_ID% --fold %%f --checkpoint best --save_nii
)

echo Experiment completed!
echo.

exit /b 0

:invalid
echo Invalid choice
exit /b 1

:end
echo.
echo ========================================
echo All experiments completed!
echo ========================================
echo.

pause
