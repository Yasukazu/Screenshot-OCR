from pathlib import Path
from cv2 import imread, hconcat, vconcat, imwrite

img_dir = Path("~/Documents/kakeibo/202510/").expanduser()
img_ext = ".png"
img_list = list(img_dir.glob(f"*{img_ext}"))
# if len(img_list) != 12: raise ValueError(f"Error: {len(img_list)} images found in '{img_dir}', expected 12.")
img_list.sort()
h_list = []
max_item_count = 2
item_count = -1
for item_count, img_path in enumerate(img_list[:max_item_count]):
	img = imread(str(img_path))
	if img is None or img.shape[0] == 0 or img.shape[1] == 0:
		raise ValueError(f"Error: {img_path} is not a valid image.")
	# draw borders
	for r in range(img.shape[2]):
		img[0, :, r] = 0
		img[-1, :, r] = 0
		img[:, 0, r] = 0
		img[:, -1, r] = 0
	h_list.append(img)
assert item_count >= 0
item_count += 1
row_count = 1
col_count = item_count // row_count
assert col_count in [1, 2]
v_list = [hconcat(h_list[:col_count])]
if row_count >= 2:
	v_list.append(hconcat(h_list[col_count:]))
img = vconcat(v_list)
output_filename = f"concat-{row_count}-{col_count}" + img_ext
output_path = img_dir / output_filename
result = imwrite(str(output_path), img)
if not result:
	print(f"Error: failed to write '{output_path}'.")
else:
	print(f"Saved {row_count} rows and {col_count} columns concatenated image to '{output_path}'.")