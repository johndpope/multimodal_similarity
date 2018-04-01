"""
    Extract video frames using ffmpeg
"""

import os
import sys
import glob
import pandas as pd
from multiprocessing import Pool
import shutil

frame_dir='/mnt/work/honda_100h/frames/'
sample_rate = 3    # 3 fps
session_template = "{0}/{1}_{2}_{3}_ITS1/{4}/"

def func(session_id):
    if os.path.isdir(frame_dir+session_id):
#        # pass if already extrated
#        return
        shutil.rmtree(frame_dir+session_id)

    print session_id
    os.makedirs(frame_dir+session_id)
    session_folder = session_template.format('/mnt/work/honda_100h',
                                                session_id[:4],
                                                session_id[4:6],
                                                session_id[6:8],
                                                session_id)
    video_filename = glob.glob(session_folder + "camera/center/*mp4")[0]

    command = ["ffmpeg",
               '-i', video_filename,
               '-vf', 'fps='+str(sample_rate),
               frame_dir+session_id+'/frame_%04d.jpg']
    os.system(' '.join(command))


sessions=['201710031645', '201710031224', '201710061311', '201710041209', '201710041351', '201710031436', '201710060950', '201710041448', '201710061114', '201710040938', '201710031458', '201710061345', '201710031247', '201710041102']

for session_id in sessions:
    func(session_id)
