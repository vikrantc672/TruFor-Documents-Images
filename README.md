# TruFor
put the folder for input images path in docker_run.sh file

# For Ubuntu/Mac
Step 1: Put the folder for input images path in docker_run.sh file
step 2: cd test_docker\
Step 3: sudo bash docker_build.sh\
Step 3: Kindly specify input images path in docker_run.sh file in the variable INPUT_DIR\

CAUTION!! : KINDLY TAKE CARE THAT THERE IS NO SPACE IN YOUR INPUT FILE PATH\
step 4: sudo bash docker_run.sh >output.txt\
Step 5: Output is shown in output.txt file\


# For Windows
Step 1: Put the folder for input images path in docker_run.bat file\
Step 1.1: cd test_docker\
Step 2: try command docker_build.bat if doesn't run then install docker desktop windows from https://docs.docker.com/desktop/release-notes/ download 24.0\
Step 3: After download install the docker the computer will restart\
Step 4: Then check if docker installed on not using the command docker -v\
Step 5: docker_build.bat\
You might need to start the docker from the desktop by clicking on docker shortcut\
Step 6: docker_run.bat>output.txt\
Step 7: Output is shown in output.txt file\
Step 8:Step 6: Kindly put the csv generated in MMFUSION in test_docker folder and then specify the path in overall_accuracy.py the paths of Trufor and MMFUSION CSV and then run the command python3 overall_accuracy.py to get the overall accuracy
