#from PIL import Image
import pyocr
#import cv2
#from google.colab.patches import cv2_imshow

tools = pyocr.get_available_tools()
'''[<module 'pyocr.tesseract' from '/home/yasukazu/github/screen/.venv/lib/python3.13/site-packages/pyocr/tesseract.py'>,
 <module 'pyocr.libtesseract' from '/home/yasukazu/github/screen/.venv/lib/python3.13/site-packages/pyocr/libtesseract/__init__.py'>]# '''
tool = tools[0]
print(tool.get_name())
# 'Tesseract (sh)'
