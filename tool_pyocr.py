from PIL import Image
import pyocr
#import cv2
#from google.colab.patches import cv2_imshow
'''[<module 'pyocr.tesseract' from '/home/yasukazu/github/screen/.venv/lib/python3.13/site-packages/pyocr/tesseract.py'>,
 <module 'pyocr.libtesseract' from '/home/yasukazu/github/screen/.venv/lib/python3.13/site-packages/pyocr/libtesseract/__init__.py'>]# '''
tools = pyocr.get_available_tools()
tool = tools[0]
print(tool.get_name())
# 'Tesseract (sh)'
from path_feeder import PathFeeder
path_feeder = PathFeeder()
fullpath = path_feeder.first_fullpath
img1 = Image.open(fullpath).convert('L')
txt1 = tool.image_to_string(
    img1,
    lang='jpn+eng',
    builder=pyocr.builders.DigitLineBoxBuilder 
)
#builders.TextBuilder(tesseract_layout=3)
'''In [9]: dir(pyocr.builders)
Out[9]: 
['BaseBuilder',
 'Box',
 'DigitBuilder',
 'DigitLineBoxBuilder',
 'HTMLParser',
 'LineBox',
 'LineBoxBuilder',
 'TextBuilder',
 'WordBoxBuilder','''
'''pagesegmode values are:
0 = Orientation and script detection (OSD) only.
1 = Automatic page segmentation with OSD.
2 = Automatic page segmentation, but no OSD, or OCR
3 = Fully automatic page segmentation, but no OSD. (Default)
4 = Assume a single column of text of variable sizes.
5 = Assume a single uniform block of vertically aligned text.
6 = Assume a single uniform block of text.
7 = Treat the image as a single text line.
8 = Treat the image as a single word.
9 = Treat the image as a single word in a circle.
10 = Treat the image as a single character.'''
'''   365     image = image.convert("RGB")
    366 image.save(os.path.join(tmpdir, "input.bmp"))
    367 (status, errors) = run_tesseract("input.bmp", "output", cwd=tmpdir,
    368                                  lang=lang,
--> 369                                  flags=builder.tesseract_flags,
    370                                  configs=builder.tesseract_configs)
    371 if status:
    372     raise TesseractError(status, errors)

AttributeError: type object 'DigitLineBoxBuilder' has no attribute 'tesseract_flags''''