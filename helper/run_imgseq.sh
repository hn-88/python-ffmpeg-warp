#!/bin/zsh
source ~/venvs/gui_env/bin/activate
# This virtual environment was created with
# python3 -m venv ~/venvs/gui_env
# followed by
# source ~/venvs/gui_env/bin/activate
# python -m pip install --upgrade pip
# python -m pip install numpy scipy pillow
# as mentioned at https://hnsws.blogspot.com/2025/12/using-py-ffmpeg-warp-on-mac.html
cd /Users/hari/Downloads/python-ffmpeg-warp-main
python3 imgseq_to_video.py

