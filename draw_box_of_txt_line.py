#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 21:24:10 2025

@author: yasukazu
"""
from PIL import Image, ImageDraw, ImageFont
from path_feeder import DbPathFeeder

feeder = DbPathFeeder(month=4)
for f in feeder.feed():
    break
fullpath = feeder.dir / (f[1] + '.png')

image = Image.open(fullpath)

sql = f"SELECT day, stem, txt_lines FROM `{feeder.table_name}`;"

cur = feeder.conn.cursor()

for cc in cur.execute(sql):
    break

cur.close()

file_stem = cc[1]
import pickle
txt_lines = pickle.loads(cc[2])
draw = ImageDraw.Draw(image)
import path_feeder
font_fullpath = path_feeder.input_dir_root / 'font' / 'OCRA.ttf'
font = ImageFont.truetype(str(font_fullpath), 24)
bb = bytearray([b for b in range(ord('A'),ord('Z')+1)]).decode()
for n, txt_line in enumerate(txt_lines):
    pos = txt_line.position[0]
    draw.text([pos[0]-24, pos[1]], bb[n], 0xff, font=font)
# B1:tilte,F5:date,Q-1:wages
    