# A project to make a 4-page PDF file which contains 1 month every-day screenshots of daily jobs.
## Screenshots are from one-day job applications
## Original motivation: to organize screenshots of niche/one-day job apps as "Taimee(タイミー)" or "Mercari-Hello(メルカリハロ)"
## Uses `Tesseract` to OCR screenshot images.
## Scripts are written in `Python(version 3.11)`

## command / scripts:

### OCR command:
- `tool_pyocr.py`: extract text from PNG (screenshot) files using OCR(_Tesseract_)
  - Commands: 
    - `run-ocr`: argument: month(1 to 12)
      - Example: `python3 tool_pyocr.py run-ocr 4`

### Generate PDF documents command:
 - `tile.py`
      - Example: `python3 tile.py`

### These 2 commands use SQLite database(`txt_lines.sqlite`) and `yyyy/mm/dd` directory structure under `SCREEN_BASE_DIR` directory set as an environmental variable or `.env` file in script starting path.
#### Database table name format is: `txt_lines-{month:02}` like 'txt_lines-05' for May.
#### TODO: Tables of last year must be dropped before the need for new month table comes.

## Directory structure:

1. App route: ~/screen-data/ if no `SCREEN_BASE_DIR` specified in env. val or `.env` file.
2. Year: screen-data/2025/
3. Month: screen-data/2025/01/ : 01 is January.

## get_number_image

- ### `DigitImage` class in `digit_image.py`:
  
  - #### classmethod `calc_font_scale` returns `font_scale` ignores font line width.

- ### `num_to_strokes.py`
  
  ![test image : discontiued](digi/get_number_image-test.PNG)

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
0a. install tesseract-ocr language data
```
sudo apt install tesseract-ocr
sudo apt install libtesseract-dev
sudo apt install tesseract-ocr-jpn
```

0b. check tesseract-ocr version and language data
  - `tesseract --version`
  - `tesseract --list-langs`
  - `tesseract --print-parameters`
  - `tesseract --print-unlv`
  - `tesseract --print-tuning-params`
0c. install python packages

d. install ImageMagick
  - `sudo apt install imagemagick`

0e. setup `.env` file if necessary
  - `TXT_LINES_DB=txt_lines.sqlite`
  - `TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata`
  - `TESSDATA_DIR=/usr/share/tesseract-ocr/4.00/tessdata`
  - `PYTHONPATH=/usr/local/lib/python3.8/dist-packages`
  - `LD_LIBRARY_PATH=/usr/local/lib:/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:/lib64:/lib:/usr/lib`

## Misaki Font PNG image:
### Thanks to public donated 8x8-bit Japanese bitmap font:`Misaki Font`
https://littlelimit.net/arc/misaki/misaki_png_2021-05-05a.zip

## In case of failure of OCR or unable to setup tesseract-ocr and Python packages:

1. convert png files into a pdf file: `convert *.png dest.pdf `

2. upload the pdf file onto Google Drive, then open and open-with-app:Document then download as markdown format.

3. extract data from downloaded markdown file.


## Public repositories of this project:
[GitLab repo](https://gitlab.com/Yasukazu1/screen/)
[GitHub repo](https://github.com/Yasukazu/screen/)