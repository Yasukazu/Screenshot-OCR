# Make a 4-page PDF file which contains 1 month every-day screenshots.

## main script
 - `tile.py`

## Directory structure:
 1. App route: ~/screen/
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