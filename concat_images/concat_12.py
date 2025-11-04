from pathlib import Path
from cv2 import imread, hconcat, vconcat, imwrite

img_dir = Path("~/Documents/screen/bank/").expanduser()
img_list = list(img_dir.glob("*.jpg"))
if len(img_list) != 12:
	raise ValueError(f"Error: {len(img_list)} images found in '{img_dir}', expected 12.")
img_list.sort()
h_list = []
for n, img_path in enumerate(img_list[:12]):
	img = imread(str(img_path))
	if not img or img.shape[0] == 0 or img.shape[1] == 0:
		raise ValueError(f"Error: {img_path} is not a valid image.")
	# draw borders
	for r in range(img.shape[2]):
		img[0, :, r] = 0
		img[-1, :, r] = 0
		img[:, 0, r] = 0
		img[:, -1, r] = 0
	h_list.append(img)
v_list = [hconcat(h_list[:6]), hconcat(h_list[6:])]
img = vconcat(v_list)
output_path = img_dir / "concat-12.jpg"
result = imwrite(str(output_path), img)
if not result:
	print(f"Error: failed to write '{output_path}'.")
else:
	print(f"Saved 12-concatenated image to '{output_path}'.")