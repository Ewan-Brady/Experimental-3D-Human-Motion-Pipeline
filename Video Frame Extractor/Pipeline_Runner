#Not part of original project as I just ran them all one-after-another manually.

import subprocess
import sys
import os


args = sys.argv[1:]
homeDirectory = os.getcwd()

if(len(args) != 3):
    print("Usage: python3 Pipeline_Runner.py <absolute input directory> <absolute Depth-Anything directory> <absolute mmpose directory>")

commands = [
    [sys.executable, "Video_Frame_Extractor.py", args[0]],
    [sys.executable, "mmpose_extractor_3d.py", args[2]],
    [sys.executable, "Depth_Frame_Extractor.py", args[0], args[1]],
    [sys.executable, "PointCloudCreator.py"]
]


try:
    for cmd in commands:
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        os.chdir(homeDirectory) #Make sure return to home directory each time.
except subprocess.CalledProcessError as e:
    print(f"Command failed: {' '.join(e.cmd)}")
    sys.exit(e.returncode)

print("All scripts completed successfully")