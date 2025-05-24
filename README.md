# Make a 4-page PDF file which contains 1 month every-day screenshots.

## Original motivation: to organize screenshots of niche job apps as 「タイミー」 or 「メルカリハロ」

## command / scripts:

- `tool_pyocr.py`: extract text from PNG (screenshot) files using OCR(_Tesseract_)
  - Commands: 
    - `run-ocr`: argument: month(1 to 12)
      - Example: `python3 tool_pyocr.py run-ocr 4`
- `tile.py`

## Directory structure:

1. App route: ~/screen/from contextlib import closing
2. Year: screen/2025/
3. Month: screen/2025/01/ : 01 is January.

## Necessary files:

- `7-seg.pkl`

## get_number_image

- ### `DigitImage` class in `digit_image.py`:
  
  - #### classmethod `calc_font_scale` returns `font_scale` ignores font line width.

- ### `num_to_strokes.py`
  
  ![test image](digi/get_number_image-test.PNG)

## Nombre(page numbering) decorator:

`@add_number(size: tuple[int, int]=(100, 50), pos: AddPos=AddPos.C, bgcolor=ImageFill.WHITE)`

- Needs to set `number_str`:str param. when to call the decorated function .
  
  ## (7+1)-segment display(the 8th is comma/period):
  
  SegElem(Enum)class: from A to G is assigned as a SegPath, H is CSegPath(:SegPath descendant).
  SegPath has draw(drw: ImageDraw.ImageDraw) method, 
  class Bit8(Flag): H member is for comma
   A
  F B
   G
  E C
   D

## Extract Text from PNG image file:
0. install Tesseract OCR
  - `sudo apt install tesseract-ocr`
  - `sudo apt install libtesseract-dev`
0a. install tesseract-ocr language data
  - `sudo apt install tesseract-ocr-jpn`
0b. check tesseract-ocr version and language data
  - `tesseract --version`
  - `tesseract --list-langs`
  - `tesseract --print-parameters`
  - `tesseract --print-unlv`
  - `tesseract --print-tuning-params`
0c. install python packages
  - `pip install pytesseract`
  - `pip install opencv-python`
  - `pip install opencv-python-headless`
  - `pip install Pillow`
  - `pip install numpy`
  - `pip install matplotlib`
  - `pip install pdf2image`
  - `pip install reportlab` 
1. convert png files into a pdf file: `convert *.png dest.pdf `

2. upload the pdf file onto Google Drive, then open and open-with-app:Document then download as markdown format.

3. extract data from downloaded markdown file.