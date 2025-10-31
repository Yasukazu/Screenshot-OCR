import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
# from dotenv import load_dotenv
import tomllib
config_file = Path('.env')
#env_path = Path(__file__).resolve().parent / '.env'
if not config_file.exists():
    raise ValueError("Error: .env file not found: %s" % config_file)
# with config_file.open('r') as f: load_dotenv(override=True, stream=f)
with config_file.open('rb') as f:
    config = tomllib.load(f)
# parent_dir = Path(__file__).resolve().parent.parent
image_dir = Path(config['IMAGE_DIR']) # Path(os.environ['IMAGE_DIR']) # parent_dir / 'DATA'
if not image_dir.exists():
    raise ValueError("Error: image dir not found: %s" % image_dir)
filename = config['IMAGE_FILENAME']
image_path = Path(image_dir) / filename
if not image_path.exists():
    raise ValueError("Error: image file not found: %s" % image_path)
image_fullpath = image_path.resolve()
image = cv2.imread(image_fullpath) #cv2.cvtColor(, cv2.COLOR_BGR2GRAY)
if image is None:
    raise ValueError("Error: Could not load image: %s" % image_fullpath)
from image_filter import taimee

ret_image = taimee(image)

fig, ax = plt.subplots()
ax.invert_yaxis()
ax.xaxis.tick_top()
ax.imshow(cv2.cvtColor(ret_image[1], cv2.COLOR_BGR2RGB))
plt.show()
print(cut_x, cut_height)
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