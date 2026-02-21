# Make a 4-page PDF file which contains 1 month every-day screenshots.

## Original motivation: to organize screenshots of 'niche job' apps
## Pre-requisites:
- Python 3.12
- Tesseract OCR
## Installation:
- `python3 -m venv .venv`
- `source .venv/bin/activate`
- `pip install uv`
- `uv sync` # install dependencies (*uv* will refer to a `pyproject.toml` file)
- `sudo apt install tesseract-ocr`
## Current development concern:
- edge_detect/image_filter.py: 
- `image_filter.py` is a script to extract text from PNG (screenshot) files using OCR(_Tesseract_)
 - It uses config file with filename of `image-filter.toml` to get parameters;the config file is searched in the current directory and upward directories;if the fullpath of the file is specified in environment variable `IMAGE_FILTER_MAIN_SETTINGS_PATH`, the config file is specified by the fullpath.
 - The config. is overwritten by command line arguments.
 - The config file is printed by running `python3 image_filter.py --toml_template`.

```python
python3 edge_detect/image_filter.py --help
```

```text
usage: image_filter.py [-h] [--app_name_to_suffix dict] [--app_names list]
                       [--app [str]] [--app_name_to_stem_end dict]
                       [--image_ext_set set] [--image_dir str]
                       [--shot_month list] [--glob_pattern str]
                       [--glob_recursive bool] [--files list]
                       [--image_area_param_section_stem str]
                       [--app_border_ratio dict] [--app_suffix bool]
                       [--save str] [--nth int] [--glob_max int] [--show bool]
                       [--bin_image bool] [--no_ocr bool] [--ocr_conf int]
                       [--psm int] [--area_param_dir str]
                       [--area_param_name_list list] [--area_param_file str]
                       [--ocr_filter_sqlite_db_name str] [--data_year int]
                       [--data_month int] [--show_ocr_area bool]
                       [--exclude_area_param_set set] [--toml_template bool]

options:
  -h, --help            show this help message and exit


  --app_name_to_suffix dict
                        Screenshot image file suffix set: suffix is the part
                        of filename before extention, delimiter is dot (.)
                        (default: {'taimee': {'taimee', 'co'}, 'mercari':
                        {'work', 'mercari'}})
  --app_names list      Application name list (default: ['TAIMEE', 'MERCARI'])
  --app [str]           Application name of the screenshot to execute OCR
                        (default: None)
  --app_name_to_stem_end dict
                        Dictionary of 'app name' to 'stem end': 'stem' means
                        the part of the filename before the extension
                        (default: {'taimee': '_jp.co.taimee', 'mercari':
                        '_jp.mercari.work.android'})
  --image_ext_set set   Image file extension set, every extention starts with
                        dot (default is {'.png'}) (default: {'.png'})
  --image_dir str       Image file root directory (default:
                        ~/Documents/screenshots)
  --shot_month list     Choose Screenshot file by its month (MM part of [YYYY-
                        MM-DD or YYYYMMDD]) included in filename stem. {Jan.
                        is 01, Dec. is 12}(specified in a list like "[1,2,..]
                        (default: [])
  --glob_pattern str    Image file name pattern as glob pattern to commit OCR
                        or to get parameters. (default: *.png)
  --glob_recursive, --noglob_recursive bool
                        Recursive glob pattern matching (default: True)
  --files list          Image file name list to commit OCR or to get
                        parameters. Every file name's pattern is:
                        <prefix>_<date>_<suffix>.<ext> (default: [])
  --image_area_param_section_stem str
                        Image area parameter section/table in image-area-
                        param.ini (default: image-area-param)
  --app_border_ratio dict
                        Screenshot image file horizontal border ratio list of
                        the app to execute OCR:(specified in format as
                        "<app_name1>:<ratio1>,<ratio2> ..." ) (default:
                        {'taimee': [2.2, 3.2]})
  --app_suffix, --noapp_suffix bool
                        Screenshot image file name has suffix(sub extention)
                        of the same as app name i.e. "<stem>.<suffix>.<ext>"
                        (default: True) (default: False)
  --save str            Output path to save OCR text of the image file as TOML
                        format into the image file name extention as
                        '.ocr-<app_name>.toml' (default: )
  --nth int             Rank(default: 1) of files descending sorted(the
                        latest, the first) by modified date as wildcard(*, ?)
                        (default: 1)
  --glob_max int        Pick up file max as pattern found in TOML (default:
                        60)
  --show, --noshow bool
                        Show images to check (default: False)
  --bin_image, --nobin_image bool
                        Use binarized image for OCR (default: False)
  --no_ocr, --nono_ocr bool
                        Do not execute OCR (default: False)
  --ocr_conf int        Confidence threshold for OCR (default: 55)
  --psm int             PSM value for Tesseract (default: 6)
  --area_param_dir str  Screenshot image area parameter config file directory
                        (default: )
  --area_param_name_list list
                        Screenshot image area parameter name list (default:
                        ['HEADING', 'SHIFT', 'BREAKTIME', 'PAYSTUB',
                        'SALARY'])
  --area_param_file str
                        Screenshot image area parameter config file: format as
                        INI or TOML(".ini" or ".toml" extention respectively):
                        in [image_area_param.<app>] section, items as
                        "<area_name>=[<p1>,<p2>,<p3>,<p4>]" (e.g.
                        "heading=[0,106,196,-1]") (default: image-area-
                        param.ini)
  --ocr_filter_sqlite_db_name str
                        SQLite DB file is created under `image_dir`/{yyyy}
                        directory(yyyy is like 2025) (default: ocr-filter.db)
  --data_year int       Year of data (like -1, 0, 2025, ...). 0 means current
                        year, negative value is difference from current year
                        (like -1 means last year), positive value means a.d.
                        year number (like 2025). If this value is larger than
                        current year, an exception might be raised. (default:
                        0)
  --data_month int      Month of data (like -1, 0, 1, 2, ...). 0 means current
                        month, negative value is difference from current month
                        (like -1 means last month), positive value means month
                        number (1: Jan, 2: Feb, ...). If this value is larger
                        than current month, data's date is treated as the last
                        year. (default: 0)
  --show_ocr_area, --noshow_ocr_area bool
                        Show every area before to commit OCR (default: False)
  --exclude_area_param_set set
                        Exclude a set of image area parameter names (default:
                        set())
  --toml_template, --notoml_template bool
                        Generate TOML format template of MainSettings
                        (default: False)
```


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


## Dynamic enum
`dev-dataclass-binder` branch

- `image_filter_main_settings.py`: `MainSettings` dataclass

```Python
from dataclass_binder import Binder
# Generate a TOML template
with open("config.toml", "wb") as f:
  for line in Binder(MainSettings()).format_toml_template(): # Need to generate an instance to get default values of default factory
      f.write(line)
# Bind(Load as a dictionary and generate a dataclass instance) a TOML file
with open("config.toml", "rb") as f:
    config = tomllib.load(f)
main_settings = Binder(MainSettings).bind(config["main-settings"])
```

### Test of dynamic enum
`dyn_enum/get_enum.py`