### Prompt for Claude.ai
 ... please write a python front-end similar to the code I pasted above (I had pasted py-ffmpeg-warp.py), which will have
1. Codec selection code similar to the above
2. Instead of resolution selection, a text box where parameters can be entered
3. the main purpose of this front-end would be to generate and run ffmpeg commands on image sequences, so an input selection browse button is required and similarly a box to choose framerate and an output selection browse button.
4. The following parameters can be auto-entered if choosing the nvenc or libx264 or libx265 codecs - example - ffmpeg -framerate 30 -i OpenSpace_%06d.png \
  -c:v libx265 -preset medium \
  -crf 12 \
  -pix_fmt yuv420p10le -profile:v main10 \
  -g 30 -keyint_min 30 -x265-params "keyint=30:min-keyint=30:scenecut=0:bframes=2" \
  -fps_mode cfr \
  -movflags +faststart \
  -y /Users/hari/.mounty/TOSHIBA\ EXT/OpenSpace-TheSearch/user/screenshots/ts_013_ancient_mars_polar.mp4
5. Another example, ffmpeg -framerate 30 -i "/home/sssvv/OpenSpace/user/screenshots/2026-03-26-11-20/OpenSpace_%06d.png" \-c:v hevc_nvenc -preset p7 \-rc constqp -qp 12 \-pix_fmt p010le -profile:v main10 \-g 30 -keyint_min 30 -sc_threshold 0 \-r 30 -vsync cfr \-movflags +faststart \-y "/media/sssvv/TOSHIBA EXT/OpenSpace-TheSearch/user/screenshots/ts_009_jupiter_europa.mp4"
