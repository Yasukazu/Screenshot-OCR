import pypdf
from path_feeder import input_dir_root
import sys
#input_files = [sys.argv[1], sys.argv[2]]
print(pypdf.__version__)
# 3.7.1
base_dir = (input_dir_root / "2025" / ("%02d" % 3))
merger = pypdf.PdfWriter()
patt1 = "2025_04_0*.pdf"
patt2 = "2503*.pdf"
for f in base_dir.glob(patt2):
    file = base_dir / f
    assert file.exists()
    merger.append(file)
output = base_dir / "2025_03-ym-files.pdf"
merger.write(str(output))
merger.close()
