@echo off

rem Define directories
set INPUT_DIR=./photos
set OUTPUT_DIR=./output
set MASK_RESULT_DIR=./mask

rem Get the absolute paths
for %%I in ("%INPUT_DIR%") do set "input_path=%%~fI"
for %%I in ("%OUTPUT_DIR%") do set "output_path=%%~fI"
for %%I in ("%MASK_RESULT_DIR%") do set "mask_path=%%~fI"

rem Get the last directory name from INPUT_DIR
for %%I in ("%input_path%") do set "last_word=%%~nxI"

rem Create output directories if they don't exist
mkdir "%OUTPUT_DIR%" 2>nul
mkdir "%MASK_RESULT_DIR%" 2>nul

rem Run Docker container (assuming 'trufor' image)
rem Use Docker without GPU (equivalent to -gpu -1)
docker run -v "%input_path%:/data" -v "%output_path%:/data_out" trufor -gpu -1 -in /data -out /data_out

rem Run Python scripts
python onlyscore_csv.py --image_dir "%input_path%" --output_dir "%output_path%" --score_file "%last_word%"
python mask.py --image_dir "%input_path%" --mask_npz_dir "%output_path%" --result_mask_dir "%mask_path%"
python accuracy.py --score_file "%last_word%"
