import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from dotenv import load_dotenv
import tomllib
env_file = Path('.env')
#env_path = Path(__file__).resolve().parent / '.env'
if not env_file.exists():
    raise ValueError("Error: .env file not found: %s" % env_file)
with env_file.open('r') as f:
    load_dotenv(override=True, stream=f)
config_file = Path(os.environ['FILTER_TOML_PATH'])
with config_file.open('rb') as f:
    config = tomllib.load(f)
# parent_dir = Path(__file__).resolve().parent.parent
image_path_config = config['image-path']
image_dir = Path(image_path_config['dir'])
if not image_dir.exists():
    raise ValueError("Error: image dir not found: %s" % image_dir)
filename = image_path_config['filename']
image_path = Path(image_dir) / filename
if not image_path.exists():
    raise ValueError("Error: image file not found: %s" % image_path)
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