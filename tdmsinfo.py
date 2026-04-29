""" 
pipenv run tdmsinfo -p "data/2026_01 Drahtspannung Testwicklung/WTD-Vibration-20260211-133449.tdms" 
>> "data/2026_01 Drahtspannung Testwicklung/info_WTD-Vibration-20260211-133449.txt"
"""

import sys
import os.path
import subprocess
from nptdms import tdmsinfo

tdms_file_path = sys.argv[1]
if not tdms_file_path:
    sys.exit()

tdms_file_path = os.path.abspath(tdms_file_path)
if not os.path.isfile(tdms_file_path):
    print(f"Error. File not found:  {tdms_file_path}")
    sys.exit()

path, ext = os.path.splitext(tdms_file_path)
output_file = os.path.dirname(tdms_file_path) + "/info_" + os.path.basename(path) + ".txt"

print(output_file)

#subprocess.run(["pipenv", "run", "tdmsinfo -p", output_file])