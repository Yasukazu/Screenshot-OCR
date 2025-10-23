import cv2
import matplotlib.pyplot as plt
import numpy as np

image_path = 'DATA/taimee-test.png'
image = cv2.imread(image_path)
if image is None:
    raise ValueError("Error: Could not load image: %s" % image_path)

height, width = image.shape[:2]
y_coords = []
for y_coord in range(height):
    h_line = image[y_coord, :]
    # h_line_is_non_white = (h_line < 0xff).all() & (h_line > 0).all()
    if len(np.unique(h_line)) == 1 and bool((h_line < 255).any()):
        y_coords.append(y_coord)
        # print("%d" % y_coord)
        cv2.line(image, (0, y_coord), (width // 16, y_coord), (255,0,0))
    
# cut heading area
last = y_coords[0]
for y_coord in y_coords[1:]:
    if y_coord > last + 2:
        break
    else:
        last = y_coord

plt.imshow(cv2.cvtColor(image[last + 1:, :], cv2.COLOR_BGR2RGB))
plt.show()
print(y_coords[last + 1:])
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