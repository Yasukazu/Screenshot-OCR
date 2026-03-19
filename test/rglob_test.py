# rglob test
from pathlib import Path

path = Path("/home/yasukazu/Documents/screen")
app = 'mercari'
ext = 'png'
pattern = f"*.{app}*.{ext}"
rglob_list = list(path.rglob(pattern))

print(rglob_list)
