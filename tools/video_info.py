import os
import json
import cv2
from PIL import Image
import argparse



parser = argparse.ArgumentParser(description="Get FRAME_INFO")
parser.add_argument("--frame_dir", help="Folder containing frames.")
args = parser.parse_args()

frame_dir = args.frame_dir
cnt = 0
frame_cnts_dict = {}
for i, entry in enumerate(os.scandir(frame_dir)):
    if not entry.name.startswith('.') and entry.is_dir():
        video_id = entry.name.split('/')[-1]
        frames = os.listdir(entry)
        file_path = os.path.join(frame_dir, video_id, frames[0])
        #print(file_path)
        fr = Image.fromarray(cv2.imread(file_path)).convert('RGB')
        item = {'cnt': len(frames), 'res': (fr.height, fr.width)}
        frame_cnts_dict[video_id] = item
        cnt+=1
    if i%1000==0:
        print(cnt)
print(cnt)
with open('video_info.json', 'w') as fp:
    json.dump(frame_cnts_dict, fp)