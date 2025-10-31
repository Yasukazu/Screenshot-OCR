import cv2
import matplotlib.pyplot as plt
# import numpy as np
import os
from pathlib import Path
from dotenv import dotenv_values
import tomllib

FILTER_TOML_PATH_STR = "FILTER_TOML_PATH"
filter_toml_path = None 
if not filter_toml_path:
    env_file = Path('.env')
    if env_file.exists():
        with env_file.open('r') as f:
            filter_toml_path = dotenv_values(stream=f)[FILTER_TOML_PATH_STR]
    else:
        filter_toml_path = os.environ[FILTER_TOML_PATH_STR]
if not filter_toml_path:
    raise ValueError("Error: `filter_toml_path` is empty!")
config_file = Path(filter_toml_path)
with config_file.open('rb') as f:
    config = tomllib.load(f)
# parent_dir = Path(__file__).resolve().parent.parent
image_path_config = config['image-path']
image_dir = Path(image_path_config['dir'])
if not image_dir.exists():
    raise ValueError("Error: image_dir not found: %s" % image_dir)
filename = image_path_config['filename']
image_path = Path(image_dir) / filename
if not image_path.exists():
    raise ValueError("Error: image_path not found: %s" % image_path)
image_fullpath = image_path.resolve()
image = cv2.imread(image_fullpath) #cv2.cvtColor(, cv2.COLOR_BGR2GRAY)
if image is None:
    raise ValueError("Error: Could not load image: %s" % image_fullpath)
from image_filter import taimee
threshold, fixed_image = taimee(image)

fig, ax = plt.subplots()
ax.invert_yaxis()
ax.xaxis.tick_top()
ax.imshow(fixed_image)
def press(event):
    print(event.key)
    if event.key == 'q':
        plt.close()
fig.canvas.mpl_connect('key_press_event', press)
plt.show()
print(threshold)
# print(cut_x, cut_height)
# print(y_coords[last + 1:])
# Result
'''102
103
105
106
107
108
109
334
335
603
604
829
830

x,x_cd=(28,168)'''