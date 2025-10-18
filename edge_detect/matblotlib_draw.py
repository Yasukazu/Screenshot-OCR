import cv2
import matplotlib.pyplot as plt
import numpy as np
# 画像を読み込む
image_path = 'data/taimee-test.png'
image = cv2.imread(image_path)
if image is None:
    raise ValueError("Error: Could not load image: %s" % image_path)

# 2. Get image dimensions
height, width = image.shape[:2]

# 3. Define the y-coordinate for the horizontal line
# Here, we get the horizontal line at the vertical center of the image.
for y_coord in range(height):
    h_line = image[y_coord, :]
    # h_line_is_non_white = (h_line < 0xff).all() & (h_line > 0).all()
    if len(np.unique(h_line)) == 1 and bool((h_line < 255).any()):
        print("%d" % y_coord)
        cv2.line(image, (0, y_coord), (width // 16, y_coord), (255,0,0))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()

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
830'''