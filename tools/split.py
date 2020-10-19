import csv
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import os
import math
# 0	 #	Sign gloss	Compound signs	Session	Scene	XML	MOV	Gloss start	Gloss end	Dominant handshape IDs	Non-dominant handshape IDs	Dominant handshape frames	Non-dominant handshape frames	uta Gloss start	uta Gloss end	uta no	uta hand annotation

# usage: place csv file in same directory, and enter the name below for CSV_file. 
# create folders for each session name and place scenes from that session in it's session directory.
# signs in each session will be split into video files and placed in a folder called signs in its respective session folder

CSV_file="uta_handshapes_tyler.csv"

SIGN_SEQ = 1
SIGN_NAMEi = 2
SESSIONi = 4
SCENEi = 5
MOVi = 7

GLOSS_STARTi = 8
GLOSS_ENDi = 9
catalog = {}

FPS = 60

with open(CSV_file, 'r') as f:
    reader = csv.reader(f)
    next(reader)
    next(reader)
    for row in reader:
        line = [row[SIGN_NAMEi], row[SESSIONi]]
        SEQ = row[SIGN_SEQ]
        SIGN_NAME = row[SIGN_NAMEi]
        SESSION = row[SESSIONi]
        SCENE = row[SCENEi]
        MOV = row[MOVi]
        filename = MOV.split('/')[-1]
        START = math.floor(int(row[GLOSS_STARTi])/FPS)
        END = math.ceil(int(row[GLOSS_ENDi])/FPS)
        stats = (SIGN_NAME, START, END)
        catalog.setdefault(SESSION, {})
        catalog[SESSION].setdefault(filename, [])
        catalog[SESSION][filename].append(stats)


dirs = os.listdir()

print(catalog["2008_05_12a"]["scene1-camera1.mov"])
for session in catalog:
    if session in dirs:
        print("I found ", session, "in dirs")
        vids = os.listdir("./"+session)
        for scene in catalog[session]:
            if scene in vids:
                for item in catalog[session][scene]:
                    print(item)
                    try:
                        os.makedirs(session + "/signs/")
                    except FileExistsError:
                        # directory already exists
                        pass
                    ffmpeg_extract_subclip(
                        session+"/"+scene, item[1], item[2], session+"/signs/"+str(item[0])+".mp4")


# print(vids["2008_05_12a"]["scene1-camera1.mov"])
# print("len = ", len(vids["2008_05_12a"]["scene1-camera1.mov"]))

# ffmpeg_extract_subclip("2008_05_12a/scene1-camera1.mov", 10, 15, "10to60.mp4")
