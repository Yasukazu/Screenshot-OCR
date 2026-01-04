# Make a 4-page PDF file which contains 1 month every-day screenshots.

## Original motivation: to organize screenshots of niche job apps as 「タイミー」 or 「メルカリハロ」


## command / scripts:

- `tool_pyocr.py`: extract text from PNG (screenshot) files using OCR(_Tesseract_)
  - Commands: 
    - `run-ocr`: argument: month(1 to 12)
      - Example: `python3 tool_pyocr.py run-ocr 4`
- `tile.py`

## Directory structure:

1. App route: `~/screen/`
2. Year: `screen/2025/`
3. Month: `screen/2025/01/` : 01 is January.

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

0a. install Tesseract OCR and its development libraries and language data of Japanese

```
sudo apt install tesseract-ocr -y
sudo apt install libtesseract-dev -y
sudo apt install tesseract-ocr-jpn -y
```

0a2. Download a better(just a bit more accurate) data and set its environment variable

```
mkdir -p ~/.local/share/tessdata/best
```

Visit "https://github.com/tesseract-ocr/tessdata_best/tree/main" and download "jpn.traineddata" then copy it to `~/.local/share/tessdata/best/`.

```
export TESSDATA_PREFIX=~/.local/share/tessdata/best
```

0b. check tesseract-ocr version and language data

  - `tesseract --version`
  - `tesseract --list-langs`
  - `tesseract --print-parameters`
  - `tesseract --print-unlv`
  - `tesseract --print-tuning-params`

0c. install python packages

#### Synchronize `pyproject.toml` by `uv` command

 - `uv sync`

  - `pip install pytesseract`
  - `pip install opencv-python`
  - `pip install opencv-python-headless`
  - `pip install Pillow`
  - `pip install pandas`
  - `pip install logbook`
  ## `requirements.txt`:  ```
        pandas
        Pillow
        click
        opencv-contrib-python
        python-dotenv
        pytesseract
        pyocr
        returns
        ipdb
        loguru
        logbook
      ```
0d. install ImageMagick
  - `sudo apt install imagemagick`
0e. setup `.env` file
```
SCREEN_BASE_DIR='/home/user1/screen'
SCREEN_YEAR='2025'
SCREEN_MONTH='05'
TXT_LINES_DB='txt_lines.sqlite'
H_PAD=20
V_PAD=40
```

  - `TESSDATA_PREFIX=~/.local/share/tessdata/best`
  - `LD_LIBRARY_PATH=/usr/local/lib:/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:/lib64:/lib:/usr/lib`

0f. Install Misaki Font

from cwd as screen root:
```
wget https://littlelimit.net/arc/misaki/misaki_png_2021-05-05a.zip
mkdir font
unzip misaki_png_2021-05-05a.zip -d font
ls font/misaki_gothic.png
```

## In case of failure of OCR or unable to setup tesseract-ocr and Python packages:
1. convert png files into a pdf file: `convert *.png dest.pdf `

2. upload the pdf file onto Google Drive, then open and open-with-app:Document then download as markdown format.

3. extract data from downloaded markdown file.

## Development

### Jupyter notebook(aka *Jupyterlab*) in GCP(Google Cloud Platform):

1. Make `.jupyter` directory and then copy `jupyter_lab_config.py` into it. Notice: the configuration is not safe-bound but no problem in restricted environment like GCP.


```python
c.LabServerApp.open_browser = False
c.ServerApp.token = ''
c.ServerApp.password = ''
c.ServerApp.allow_origin = '*'
```

2. Activate a virtual environment of the folder as `. .venv/bin/activate`


2. Run Python command with '-m jupyterlab' option to start *Jupyterlab* server: `python -m jupyterlab`

2. Click "Web preview" icon (looks like Brazil national flag, a cirgle in a rhombus or ascii figure: `[<o>]`) in Cloud Shell Editor's top menu, changing port to 8888.

3. In Jupyter page, click "iPython3" icon to open an `ipynb` notebook.

4. Paste a script (like `matplotlib_draw.py`) into a cell of the `ipynb` notebook.

5. Run the cell by clicking the *run* button (right-edged triangle icon: `|>`) in the notebook(`*.ipynb`)'s top menu.

## Current development concern:
 - edge_detect/image_filter.py: 

```python
python edge_detect/image_filter.py --help
```

```text
usage: image_filter.py [-h] [--image_ext IMAGE_EXT [IMAGE_EXT ...]] [--image_dir IMAGE_DIR]
                       [--shot_month SHOT_MONTH] [--app_stem_end APP_STEM_END]
                       [--app_border_ratio [APP_BORDER_RATIO ...]] [--app_suffix]
                       [--app {taimee,mercari}] [--save SAVE] [--nth NTH] [--glob-max GLOB_MAX] [--show]
                       [--make] [--no-ocr] [--ocr-conf OCR_CONF] [--psm PSM]
                       [--area_param_file AREA_PARAM_FILE]
                       [files ...]

positional arguments:
  files                 Image file fullpaths to commit OCR or to get parameters.

options:
  -h, --help            show this help message and exit
  --image_ext IMAGE_EXT [IMAGE_EXT ...]
                        [env var: IMAGE_FILTER_IMAGE_EXT]
  --image_dir IMAGE_DIR
                        [env var: IMAGE_FILTER_IMAGE_DIR]
  --shot_month SHOT_MONTH
                        Choose Screenshot file by its month (MM part of [YYYY-MM-DD or YYYYMMDD])
                        included in filename stem. {Jan. is 01, Dec. is 12}(specified in a list like
                        "[1,2,..]" ) [env var: IMAGE_FILTER_SHOT_MONTH]
  --app_stem_end APP_STEM_END
                        Screenshot image file name pattern of the screenshot to execute OCR:(specified
                        in format as "<app_name1>:<stem_end1>,<stem_end2>;..." ) [env var:
                        IMAGE_FILTER_APP_STEM_END]
  --app_border_ratio [APP_BORDER_RATIO ...]
                        Screenshot image file horizontal border ratio list of the app to execute
                        OCR:(specified in format as "<app_name1>:<ratio1>,<ratio2> ..." ) [env var:
                        IMAGE_FILTER_APP_BORDER_RATIO]
  --app_suffix          Screenshot image file name has suffix(sub extention) of the same as app name
                        i.e. "<stem>.<suffix>.<ext>" (default: True)
  --app {taimee,mercari}
                        Application name of the screenshot to execute OCR: choices=taimee, mercari [env
                        var: IMAGE_FILTER_APP_NAME]
  --save SAVE           Output path to save OCR text of the image file as TOML format into the image
                        file name extention as ".ocr-<app_name>.toml"
  --nth NTH             Rank(default: 1) of files descending sorted(the latest, the first) by modified
                        date as wildcard(*, ?)
  --glob-max GLOB_MAX   Pick up file max as pattern found in TOML
  --show                Show images to check
  --make                make config. from image(i.e. this arg. makes not to use param configs in any
                        config file; specify image_area_param values like "--image_area_param
                        heading:0,106,196,-1"
  --no-ocr              Do not execute OCR
  --ocr-conf OCR_CONF   Confidence threshold for OCR
  --psm PSM             PSM value for Tesseract
  --area_param_file AREA_PARAM_FILE
                        Screenshot image area parameter config file: format as: in [image_area_param]
                        section, items as "<area_name>=<p1>,<p2>,<p3>,<p4>" (e.g.
                        "heading=0,106,196,-1")

Args that start with '--' can also be set in a config file
(/home/yasukazu/github/screen/edge_detect/image-filter.toml or
/home/yasukazu/github/screen/edge_detect/image-filter.ini). Uses multiple config parser settings (in
order):  [1] TOML: Config file syntax is Tom's Obvious, Minimal Language. See https://github.com/toml-
lang/toml/blob/v0.5.0/README.md for details.  [2] INI: Uses configparser module to parse an INI file
which allows multi-line values. See https://docs.python.org/3/library/configparser.html for details.
This parser includes support for quoting strings literal as well as python list syntax evaluation.
Alternatively lists can be constructed with a plain multiline string, each non-empty line will be
converted to a list item.   In general, command-line values override environment variables which
override config file values which override defaults.
```