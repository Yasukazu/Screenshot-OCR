import cv2
import matplotlib.pyplot as plt
import numpy as np

image_path = 'DATA/taimee-test.png'
image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2GRAY)
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
        cv2.line(image, (0, y_coord), (width // 16, y_coord), 255) #,0,0))
    
# cut heading area
last = y_coords[0]
ln = -1
for ln, y_coord in enumerate(y_coords[1:]):
    if y_coord > last + 2:
        break
    else:
        last = y_coord
cut_image = image[last + 1:, :]

# mask image of a left-top circle as blank
## find the top block
cut_height = y_coords[ln + 1] - last
assert cut_height > 0
### scan left-top area for a (non-white) shape
for x, x_cd in enumerate(range(width)):
    v_line = cut_image[:(cut_height - 1), x_cd]
    if len(np.unique(v_line)) > 1:
        break
for x_cd in range(width - x):
    v_line = cut_image[:(cut_height - 1), x_cd + x]
    if len(np.unique(v_line)) == 1 and (v_line == 255).all():
        break
cut_x = x_cd + x
mask = np.full((cut_height, width), 255, np.uint8)
# cv2.circle(mask, (width // 16, height // 16), width // 16, (255, 255, 255), -1)
#cut_image = cv2.bitwise_and(cut_image, cut_image, mask=mask)
# plt.imshow(cv2.cvtColor(cut_image, cv2.COLOR_BGR2RGB))
fig, ax = plt.subplots()
ax.invert_yaxis()
ax.xaxis.tick_top()
ax.imshow(cv2.cvtColor(cut_image, cv2.COLOR_BGR2RGB))
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
830

x,x_cd=(28,168)'''