# Misaki font
from collections import namedtuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
import cv2
from PIL import Image, ImageSequence
from dotenv import load_dotenv
load_dotenv()
import sys,os
from pathlib import Path
import pickle

screen_base_dir_name = os.getenv('SCREEN_BASE_DIR')

if not screen_base_dir_name:
    raise ValueError(f"{screen_base_dir_name=} is not set in env.!")

screen_dir = Path(screen_base_dir_name)
font_dir = Path(screen_base_dir_name) / 'font'

if not font_dir.exists():
    raise ValueError(f"`{font_dir=}` does not exists!")


FONT_SIZE = (8, 8)
HALF_FONT_SIZE = (4, 8)

# type ku = tuple[tuple[int, int], int]

DIGIT_KU = (3, 16),10
HEX_KU = (3, 33),6
TEN_KU = (1, 4),2

def array(seq):
    return np.array(seq, np.int64)

font_size_array = array(FONT_SIZE)
font_file_name = 'misaki_gothic.png'

PNG_EXT = '.png'

class Byte(int):
    @property
    def h(self):
        return self >> 4
    @property
    def l(self):
        return self & 0xf

def byte_to_xy(byte: int):
    b = Byte(byte)
    return b.l, b.h

CwLw = namedtuple('Cw_lw', ['cw', 'lw'])
WidthHeight = namedtuple('WidthHeight', ['width', 'height'])
class FontMap(Enum):
    misaki_4x8 = WidthHeight(64, 128)
    misaki_mincho = WidthHeight(752, 752)

class MisakiFont:#(Enum):
    HEIGHT = 8
    FULL_WIDTH = 8
    HALF_WIDTH = 4
    HALF_FONT = FontMap.misaki_4x8
    HALF_NAME = HALF_FONT.name
    HALF_SIZE = HALF_FONT.value
    FULL_FONT = FontMap.misaki_mincho
    FULL_NAME = FULL_FONT.name
    FULL_SIZE = FULL_FONT.value

    @classmethod
    def get_font_dir(cls):
        return font_dir

    @classmethod
    def get_font_fullpath(cls, font=FontMap.misaki_4x8):
        font_file_name = font.name + PNG_EXT
        font_fullpath = font_dir / font_file_name
        return font_fullpath

    @classmethod
    def get_font_base_image(cls, font=FontMap.misaki_4x8)-> Image.Image:
        font_file_name = font.name + PNG_EXT
        font_fullpath = font_dir / font_file_name
        if not font_fullpath.exists():
            raise ValueError(f"`{font_fullpath=}` does not exists!")
        base_image = Image.open(str(font_fullpath))
        return base_image

    @classmethod
    def get_half_font_image(cls, c: int, font_dict: dict[int, Image.Image]={}):
        if c > 255 or c < 0:
            raise ValueError('Must be an unsigned byte(0 to 255)!')
        if c in font_dict:
            return font_dict[c]
        font_file_name = MisakiFont.HALF_NAME + PNG_EXT
        font_fullpath = font_dir / font_file_name
        if not font_fullpath.exists():
            raise ValueError(f"`{font_fullpath=}` does not exists!")
        full_image = Image.open(str(font_fullpath))
            
        def append_fonts(ku: int): 
            x_pos, y_pos = byte_to_xy(ku)
            x_offset = x_pos * 4
            y_offset = y_pos * 8
            offset = array([x_offset, y_offset])
            end_point = offset + np.array(HALF_FONT_SIZE)
            area = array([offset, end_point])
            img_area = area.ravel().tolist()
            font_part = full_image.crop(img_area)
            return font_part
        font_image = append_fonts(c)
        font_dict[c] = font_image
        return font_image

    @classmethod
    def get_line_images(cls, font_map=FontMap.misaki_4x8, images: list[Image.Image | None] | dict[int, Image.Image]={}):
        # Return: dict_keys([2, 3, 4, 5, 6, 7, 10, 11, 12, 13])
        assert isinstance(images, list) or isinstance(images, dict)
        base_image = cls.get_font_base_image(font_map)
        height = font_map.value.height
        line_image_size = (font_map.value.width, MisakiFont.HEIGHT)
        box = [0, 0, *line_image_size]
        # font_width = 4 if '4x' in font_map.name else MisakiFont.FULL_WIDTH
        def down_shift(box):
            box[1] += MisakiFont.HEIGHT
            box[3] += MisakiFont.HEIGHT
        for n in range(height // MisakiFont.HEIGHT):
            image = base_image.crop(box)
            if isinstance(images, dict):
                array = np.array(image)
                bb = array[array == 0] # black part
                if len(bb) > 0:
                    images[n] = image
            else:
                images.append(image)
            down_shift(box)
        return images
    
    @classmethod
    def line_image_to_char_image(cls, line_dict: dict[int, Image.Image],
        char_dict: dict[int, Image.Image] = {}):
        for k in line_dict.keys():
            if k in char_dict:
                continue
            line_image = line_dict[k]
            line_width = line_image.size[0]
            c_w = CwLw(4, line_width) if line_width == 64 else (CwLw(8, line_width) if line_width == 752 else None)
            assert c_w
            cs = c_w.lw // c_w.cw
            for h in range(cs):
                c = ((k << c_w.cw) | h) if c_w.cw == 4 else ((k + 1) * 100 + (h + 1))
                p = h * c_w.cw
                box = [p, 0, p + c_w.cw, 8]
                char_dict[c] = line_image.crop(box)
        return char_dict

    @classmethod
    def save_as_pickle(cls, char_dict: dict[int, Image.Image], file_stem: str, ext='.pkl'):
        pkl_fullpath = cls.get_font_dir() / (file_stem + ext)
        with pkl_fullpath.open('wb') as pkl:
            pickle.dump(char_dict, pkl)
    @classmethod
    def load_char_dict(cls, file_stem: str=FontMap.misaki_4x8.name, ext='.pkl') -> dict[int, Image.Image]:
        pkl_fullpath = cls.get_font_dir() / (file_stem + ext)
        with pkl_fullpath.open('rb') as pkl:
            char_dict = pickle.load(pkl)
        return char_dict
    @classmethod
    def load_char_dicts(cls, file_stem_list: list[str]=[FontMap.misaki_4x8.name, FontMap.misaki_mincho.name], ext='.pkl', char_dict: dict[int, Image.Image] = {}) -> dict[int, Image.Image]:
        for stem in file_stem_list:
            char_dict |= cls.load_char_dict(stem)
        return char_dict


class SecondMisakiFont(Enum):
    HALF_NAME = 'misaki_gothic_2nd_4x8'
    FULL_NAME = 'misaki_gothic_2nd'

arc_font_file_name = 'misaki_gothic-digit.tif'

def get_misaki_digit_images(scale=1):
    re_size = (array(FONT_SIZE) * scale).tolist() if scale > 1 else None
    arc_font_fullpath = font_dir / arc_font_file_name
    image_list = {}
    if arc_font_fullpath.exists():
        page = Image.open(str(arc_font_fullpath))
        for n, page in enumerate(ImageSequence.Iterator(page)):
            if re_size:
                page = page.resize(re_size)
            image_list[n]=(page)
        return image_list
    font_fullpath = font_dir / font_file_name
    if not font_fullpath.exists():
        raise ValueError(f"`{font_fullpath=}` does not exists!")
    full_image = Image.open(str(font_fullpath))
    def kuten_to_xy(*kuten: int):
        return kuten[1] - 1, kuten[0] - 1
    image_list = []
    def append_fonts(ku): 
        num_ku_ten, num_range = ku
        num_pos = kuten_to_xy(*num_ku_ten)
        num_addr = array(num_pos)
        offset = (num_addr * 8) # - array([0, 2])
        end_point = offset + np.array(FONT_SIZE)
        area = array([offset, end_point])
        for i in range(num_range):
            img_area = area.ravel().tolist()
            font_part = full_image.crop(img_area)
            #font_part = font_part.resize([32, 40])
            area += np.array([8, 0])
            image_list.append(font_part)
    append_fonts(DIGIT_KU)
    append_fonts(HEX_KU)
    append_fonts(TEN_KU)
    image_list[0].save(str(arc_font_fullpath),
        save_all=True, append_images=image_list[1:]) # compression='tiff_lzw'
    return image_list

def iskanji1(c: int):
    return 0x81 <= c <= 0x9F or 0xE0 <= c <= 0xFC
def iskanji2(c: int):
    return 0x40 <= c <= 0x7E or 0x80 <= c <= 0xFC
def iskanji(c: int):
    return 0x80 <= c <= 0xFF
import unicodedata
class MisakiFontImage:
    def __init__(self, scale=1):
        self.digit_fonts = get_misaki_digit_images(scale=scale)
        self.char_dict = MisakiFont.load_char_dicts()
        self.scale = scale
    def get_str_image(self, s: str, image_type='P'):
        ns = unicodedata.normalize('NFKC', s)
        imgs: list[Image.Image] = []
        for n in ns:
            j = n.encode('EUC-JP')
            if len(j) == 1:
                code = ord(j)
                imgs.append(self.char_dict[code])
            else:
                ku = j[0] - 0xa0
                ten = j[1] - 0xa0
                kuten = ku * 100 + ten
                img = self.char_dict[kuten]
                imgs.append(img)
        image_list = [pil2cv(img) for img in imgs]
        n_img = cv2.hconcat(image_list)
        n_img[n_img == 1] = 255
        y, x = n_img.shape
        x *= self.scale
        y *= self.scale
        r_img = cv2.resize(n_img, (x, y), interpolation=cv2.INTER_AREA) #, fx=self.scale, fy=self.scale)
        if image_type[0].upper() == 'P':
            r_img = cv2pil(r_img)
        return r_img

    def get_number_image(self, num_array: bytearray):
        scaled = font_size_array * self.scale
        image_list = [pil2cv(self.digit_fonts[b]) for b in scaled]
        n_img = cv2.hconcat(image_list)
        n_img[n_img == 1] = 255
        pil_number_image = cv2pil(n_img)
        return pil_number_image
    '''
        image_fullpath = screen_dir / 'number_image.png'
        cv2.imwrite(str(image_fullpath), n_img)
        for n, b in enumerate(num_array):
            font_img = pil2cv(self.fonts[b])
            cv2.namedWindow("image", cv2.WINDOW_NORMAL)
            cv2.imshow("image", font_img)
            image.paste(self.fonts[b], (w * n, 0))'''


def pil2cv(image: Image.Image):
    ''' PIL型 -> OpenCV型 '''
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    return new_image


def cv2pil(image):
    ''' OpenCV型 -> PIL型 '''
    new_image = image.copy()
    if new_image.ndim == 2:  # モノクロ
        new_image[new_image == 1] = 255
        #return new_image.convert('L')
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
    new_image = Image.fromarray(new_image)
    return new_image

def pickle_fonts(font=MisakiFont.FULL_FONT):
    image_line_dict = MisakiFont.get_line_images(font)
    c_to_image = MisakiFont.line_image_to_char_image(image_line_dict)
    MisakiFont.save_as_pickle(c_to_image, font.name)

from pprint import pp
import click

@click.group()
def cli():
    pass

@cli.command()
def save_full():
    pickle_fonts()

@cli.command()
def load_full():
    file_stem = MisakiFont.FULL_FONT.name
    full_fonts = MisakiFont.load_char_dict(file_stem=file_stem)
    pp(full_fonts)

@cli.command()
@click.argument('text', nargs=1)
@click.option('--scale', type=click.IntRange(1, 100), default=1)
def font_image(text: str, scale: int=1, show=False):
    image_feeder = MisakiFontImage(int(scale))
    image = image_feeder.get_str_image(text)
    if show:
        image.show()
    return image


if __name__ == '__main__':
    cli()
    import sys, os
    sys.exit(0)
    font = MisakiFont.HALF_FONT
    char_dict = MisakiFont.load_char_dict(font.name)
    num_str = "%0.2f ｸﾞﾗﾑ" % -3.14
    encoded = num_str.encode('shift-jis')
    # pil_image = (char_dict[encoded[0]]) cv_image = pil2cv(pil_image)
    pil_images = [char_dict[c] for c in encoded]
    cv_images = [pil2cv(pil_image) for pil_image in pil_images]
    cv_image = cv2.hconcat(cv_images)
    cv2.imshow('hconcat image', cv_image)
    cv2.waitkey()
    h_image = cv2pil(cv_image)
    h_image.show()


    image_line_dict = MisakiFont.get_line_images(font)
    c_to_image = MisakiFont.line_image_to_char_image(image_line_dict)
    MisakiFont.save_as_pickle(c_to_image, font.name)
    arc_fullpath = MisakiFont.get_font_dir() / (font.name + '.tif')
    #image_list[0].save(arc_fullpath, save_all=True, append_images=image_list[1:])
    sys.exit(0)

    b = sys.argv[1] # b = input('char for the font:')
    c = ord(b)
    if c > 255:
        raise ValueError(f"{c=} exceeds half font range(255)")
    font_image = MisakiFont.get_half_font_image(c)
    if font_image:
        font_image.show()
    sys.exit(0)
    from format_num import formatnums_to_bytearray, FloatFormatNum
    mfi = MisakiFontImage(8)
    fn = [FloatFormatNum(3.14)]
    ba = formatnums_to_bytearray(fn, conv_to_bin2=False)
    ni = mfi.get_number_image(ba)
    ni.show()
    images = get_misaki_digit_images(8)
    a_img = images[0xa]
    f_img = images[0xf]
    m_img = images[0xf + 1]
    p_img = images[0xf + 2]
    for n, image in (images).items():
        print(n)
        #image.show()