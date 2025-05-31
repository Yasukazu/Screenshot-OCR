import sys
from enum import Enum
from typing import Sequence, Iterator
from typing import Callable, Any
import os.path, calendar
from pathlib import Path
import logging
logger = logging.getLogger(__file__)
from dotenv import load_dotenv
load_dotenv(override=True)
from PIL import Image, ImageDraw
import img2pdf
import ttl
home_dir = Path(os.path.expanduser('~'))
from path_feeder import PathFeeder, get_last_month, FileExt
from image_fill import ImageFill
from put_number import PutPos #, put_number
from app_type import AppType
from input_dir import get_last_month, FileExt, get_input_dir_root, get_input_dir

last_month_date = get_last_month()
year = last_month_date.year
month = last_month_date.month
img_root_dir = get_input_dir_root()
img_root_dir.mkdir(exist_ok=True, parents=True)
img_dir = img_root_dir / str(year) / f'{month:02}'
img_dir.mkdir(parents=True, exist_ok=True)

IMG_SIZE = (720, 1612)
H_PAD = 20
V_PAD = 40

file_over = False

year_month_name = f"{year}-{month:02}"

class PdfLayout(Enum):
    a4pt = (img2pdf.mm_to_pt(210),img2pdf.mm_to_pt(297))
    a3lp = (img2pdf.mm_to_pt(420),img2pdf.mm_to_pt(297))

#def convert_tiff_to_png_files(): cmd = 'convert 2025-02-8x4.tif -format png -scene 1 2025-02-%d.png'

def paged_png_feeder(layout=PdfLayout.a3lp, app_type=AppType.T):
    feeder=PathFeeder(input_type=FileExt.QPNG, type_dir=True)
    img_dir = feeder.dir
    year = feeder.year
    month = feeder.month
    names = []
    match layout:
        case PdfLayout.a4pt:
            for p in range(4):
                fullpath = img_dir / f"{year}-{month:02}-{p + 1}.png"
                if not fullpath.exists():
                    raise ValueError(f"{fullpath} does not exist!")
                names.append(fullpath)
            return names
        case PdfLayout.a3lp:
            sub_ext_list = ('L', 'R')
            sub_ext = 'LR'
            for p in range(2):
                def name(lr):
                    return f"{app_type.name}-{2 * p + (sub_ext.index(lr) + 1)}.png" # {year}-{month:02}
                fullpath_l = img_dir / name('L')
                fullpath_r = img_dir / name('R')
                for fullpath in (fullpath_l, fullpath_r):
                    if not fullpath.exists():
                        raise ValueError(f"{fullpath} does not exist!")
                outpath = img_dir / f"{year}-{month:02}-{sub_ext_list[p]}.png"
                h_image = Image.open(fullpath_l)
                padding = 100
                margin = 200
                image = Image.new('L', (h_image.width * 2 + padding + 2 * margin, h_image.height))
                margin_img = Image.new('L', (margin, h_image.height), (0xff,))
                pos = 0
                image.paste(margin_img, (0, 0))
                pos += margin_img.width
                image.paste(h_image, (pos, 0))
                pos += h_image.width + padding
                h_image = Image.open(fullpath_r)
                image.paste(h_image, (pos, 0))
                pos += h_image.width
                image.paste(margin_img, (pos, 0))
                save_path = (outpath)
                image.save(save_path)
                names.append(save_path)
            return names

def convert_to_pdf(app_type: AppType, layout=PdfLayout.a3lp):
    names = paged_png_feeder(app_type=app_type, layout=layout)
    parent_dir = names[0].parent
    fullpath = parent_dir / f"{year}-{month:02}-{app_type.name}.pdf"
    layout_fun = img2pdf.get_layout_fun(layout.value)
    with open(fullpath,"wb") as f:
        for name in names:
            assert name.exists()
        name_list = [str(n) for n in names]
        f.write(img2pdf.convert(layout_fun=layout_fun, *name_list, rotation=img2pdf.Rotation.ifvalid))

def save_into_pdf(layout=PdfLayout.a3lp):
    names = paged_png_feeder(layout=layout)
    parent_dir = names[0].parent
    fullpath = parent_dir / f"{year}-{month:02}.pdf"
    layout_fun = img2pdf.get_layout_fun(layout.value)
    with open(fullpath,"wb") as f:
        for name in names:
            assert name.exists()
        name_list = [str(n) for n in names]
        f.write(img2pdf.convert(layout_fun=layout_fun, *name_list, rotation=img2pdf.Rotation.ifvalid))


def save_pages_as_pdf(feeder: PathFeeder): #fullpath=PathFeeder().first_fullpath):
    fullpath = img_dir / f"{year_month_name}.pdf"
    imges = list(draw_onto_pages(feeder))
    imges[0].save(fullpath, "PDF" ,resolution=200, save_all=True, append_imges=imges[1:])

def save_pages_as_tiff(feeder: PathFeeder):
    fullpath = img_dir / f"{year_month_name}.tif"
    imges = list(draw_onto_pages(feeder))
    imges[0].save(fullpath, save_all=True, append_imges=imges[1:])

def save_qpng_pages(app_type=AppType.T, ext_dir=FileExt.QPNG,
    save_dir: Path = get_input_dir()):
    if not save_dir.exists():
        save_dir.mkdir(parents=True)
        logger.info("Created directory '%s' for saving pages.", save_dir)
    path_feeder = DbPathFeeder(app_type=app_type)
    hdr = app_type.name
    for pg, img in enumerate(draw_onto_pages(path_feeder=path_feeder, app_type=app_type)):
        fullpath = save_dir / f"{hdr}-{pg + 1}{ext_dir.value.ext}"
        img.save(fullpath) #, 'PNG')

from input_dir import get_year_month
def check_png_files(year=0, month=0)-> set[str]:
    date = get_year_month(year, month)
    year = date.year
    month = date.month
    f_feeder = PathFeeder(input_type=FileExt.PNG, type_dir=True, year=year, month=month)
    if not f_feeder.dir.exists():
        raise ValueError(f"Directory {f_feeder.dir} does not exist!")
    if not f_feeder.dir.is_dir():
        raise ValueError(f"Path {f_feeder.dir} is not a directory!")
    img_file_set = set([f.stem for f in f_feeder.dir.iterdir() if f.is_file() and f.suffix == FileExt.PNG.value.ext])
    # just check stems
    db_stems = set()
    app_type_list = [app_type for app_type in list(AppType) if app_type != AppType.NUL]
    for app_type in app_type_list:
        db_feeder = DbPathFeeder(app_type=app_type, year=year, month=month)
        db_stems.update([f for d, f in db_feeder.feed()])
    return img_file_set - db_stems



class ArcFileExt(Enum):
    TIFF = ('.tif', {'compression':"tiff_deflate"})
    PDF = ('.pdf', {})
from tool_pyocr import AppType
def save_arc_pages(ext: ArcFileExt=ArcFileExt.TIFF, app_type=AppType.NUL):
    from path_feeder import DbPathFeeder
    feeder = DbPathFeeder(app_type=app_type) if app_type in [AppType.M, AppType.T] else PathFeeder()
    imgs: list[Image.Image] = list(draw_onto_pages(path_feeder=feeder))
    fullpath = img_dir / f"{year}-{month:02}-8x4{ext.value[0]}"
    imgs[0].save(fullpath, save_all=True, append_images=imgs[1:], **ext.value[1])

DAY_NOMBRE_H = 50
TXT_OFST = 0 # width-direction / horizontal


from path_feeder import DbPathFeeder
def draw_onto_pages(path_feeder: PathFeeder, div=64, th=H_PAD // 2,
    v_pad=16, h_pad=8, mode='L', dst_bg=ImageFill.BLACK, app_type=AppType.T)-> Iterator[Image.Image]:
    from tool_pyocr import Date
    first_fullpath = path_feeder.first_fullpath
    if not first_fullpath:
        raise ValueError(f"No '{path_feeder.ext}' file in {path_feeder.dir}!")
    # first_img_size = Image.open(first_fullpath).size

    drc_tbl = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    ll_ww = [0, 0]
    def _to(n, xy):
        ll, ww = ll_ww[0], ll_ww[1]
        x = xy[0]
        y = xy[1]
        drc = drc_tbl[n]
        return x + ll * drc[0], y + ll * drc[1]
    def xshift(offset, *xy):
        return xy[0] + offset, xy[1]
    def yshift(*xy):
        x, y = xy[0], xy[1]
        return x, y - (V_PAD - 8)
    # fill_white = (255,255,255)
    def page_dots(ct, drw, n):
        ll, ww = ll_ww[0], ll_ww[1]
        op = list(ct)
        for i in range(n):
            dp = op[0] - ww, op[1]
            drw.line((*op, *dp), fill=ImageFill.invert(dst_bg).value, width=int(th))
            op[0] -= 2 * ww
    def month_dots(ct, drw, img_w):
        ttl.set_pit_len(img_w // 32)
        ttl.set_org(*(ct[0] + (img_w // 32), ct[1]))
        for frm, to in ttl.plot(month, '>'):#ttl.Direc.RT):
            drw.line((frm[0], frm[1], to[0], to[1]), fill=ImageFill.invert(dst_bg).value, width=int(th))

    def get_image_blocks():
        day_and_names = list(path_feeder.feed(padding=True))
        name_blocks = [day_and_names[:8], day_and_names[8:16], day_and_names[16:24], day_and_names[24:]]
        pad_size = 8 - len(name_blocks[-1])
        if pad_size > 0:
            name_blocks[-1] += [(0, '')] * pad_size
        for i, block in enumerate(name_blocks):
            number = year * (1 if app_type == AppType.T else -1) # + month / 100)
            yield concat_8_pages(block, number=number) # number_str=f"{path_feeder.month:02}{(-0xa - i):x}")


    # digit_image_param_L = BASIC_DIGIT_IMAGE_PARAM_LIST[1]
    from misaki_font import MisakiFontImage
    digit_image_feeder_L = MisakiFontImage(12)
    from put_number import put_number
    @put_number(pos=PutPos.R, digit_image_feeder=digit_image_feeder_L)
    def concat_8_pages(day_stem_list: list[tuple[int, str]], number: float)-> Image.Image:

        day_stem_list_1 = day_stem_list[:4]
        day_stem_list_2 = day_stem_list[4:]

        himg1 = concat_h(day_stem_list_1)
        himg2 = concat_h(day_stem_list_2)
        return concat_v(himg1, himg2)

    img_size = Image.open(str(path_feeder.first_fullpath)).size
    def concat_h(day_stem_list: list[tuple[int, str]], pad=h_pad)-> Image.Image:
        imim_len = len(day_stem_list)
        imim = [get_numbered_img(s, number=Date(month, d).as_float) for d, s in day_stem_list]
        max_height = img_size[1]
        im_width = img_size[0]
        width_sum = imim_len * img_size[0]
        dst_size = (width_sum + (imim_len - 1) * pad, max_height)
        dst: Image.Image = Image.new(mode, dst_size, color=dst_bg.value)
        cur = 0
        for im in imim:
            if im:
                dst.paste(im, (cur, 0))
            cur += im_width + pad
        return dst
    def concat_v(im1: Image.Image, im2: Image.Image)-> Image.Image:
        pad = v_pad
        dst_size = (im1.width, im1.height + pad + im2.height)
        dst: Image.Image = Image.new(mode, dst_size, color=dst_bg.value)
        dst.paste(im1, (0, 0))
        dst.paste(im2, (0, im1.height + pad))
        return dst

    # digit_image_param_S = BasicDigitImage.calc_scale_from_height(50)
    digit_image_feeder_S = MisakiFontImage(6) # BasicDigitImage(digit_image_param_S) # scale=24, line_width=6, padding=(4, 4))
    @put_number(pos=PutPos.L, digit_image_feeder=digit_image_feeder_S) #, line_width=digit_image_param_S.line_width))
    def get_numbered_img(stem: str, number: float)-> Image.Image | None:
        fullpath = path_feeder.dir / (stem + path_feeder.ext)
        if fullpath.exists():
            img = Image.open(fullpath).convert('L')
            return img

    for pg, img in enumerate(get_image_blocks()):
        ct = (img.width // 2, img.height // 2) # s // 2 for s in img.size)
        ll_ww[0] = img.height // div
        ll_ww[1] = img.width // 128
        #assert img.width == IMG_SIZE[0] * 4 + H_PAD * 3
        #assert img.height == IMG_SIZE[1] * 2 + V_PAD
        drw = ImageDraw.Draw(img)
        dst = _to(pg, list(ct))
        lct = list(ct)
        dstp = lct[0] + dst[0], lct[1] + dst[1]
        month_dots(ct, drw, img.width)
        page_dots(ct, drw, pg + 1) # drw.line((*ct, *dstp), fill=(255, 255, 255), width=int(th))
        text = f"{' ' * 8}{year}-{month:02}({pg + 1}/4)"
        drw.text((*xshift(TXT_OFST, *ct), *yshift(*dstp)), text, fill=ImageFill.invert(dst_bg).value)
        # draw_digit(pg + 1, drw, offset=(10, 10), scale=30, width_ratio=8)
        #name = f"{year_month_name}-{pg + 1}.png"
        yield img #, name #.convert('L') #.save(img_dir / name, 'PNG')

from collections import namedtuple
WidthHeight = namedtuple('WidthHeight', ['width', 'height'])
def concat_8_pages(img_size: tuple[int, int], dir: Path, ext: str, names: Sequence[str], h_pad: int=0, v_pad: int=0)-> Image.Image:
    def open_img(f: str)-> Image.Image | None:
        fullpath = dir / (f + ext)
        img =  Image.open(fullpath) if fullpath.exists() else None
        return img

    names_1 = list(names[:4])
    names_2 = list(names[4:])

    img_count = len(list(names))
    himg1 = get_concat_h(img_count, img_size, [open_img(n) for n in names_1], pad=h_pad) # (dq()))
    himg2 = get_concat_h(img_count, img_size, [open_img(n) for n in names_2], pad=h_pad) # (dq()))h4img()
    return get_concat_v(2, img_size, (himg1, himg2), pad=v_pad)

def get_img_file_names_(glob=True):
    days = calendar.monthrange(year, month)[1]
    for i in range(days):
        yield f"{'??' if glob else month}{(i + 1):02}.png"
    pad = 32 - days
    for n in range(pad):
        yield None


blank_img = Image.new('L', IMG_SIZE, (0xff,))

def open_image(dir: Path, name: str, glob=False):
    if not name:
        global file_over 
        file_over = True
        return blank_img
    if glob:
        fullpath_list = list(dir.glob(name))
        fpthslen = len(fullpath_list)
        assert fpthslen < 2
        if fpthslen == 1: #(fp:=fullpath_list[0]).exists():
            img = Image.open(fullpath_list[0]).convert('L')
            assert img.size == IMG_SIZE
            return img
        else:
            return blank_img
    else:
        assert (dir / name).exists()
        return Image.open(dir/ name)
def get_concat_h(imim_len: int, img_size: tuple[int, int], imim: Sequence[Image.Image | None], pad=0, mode='L', dst_bg=(0xff,))-> Image.Image:
    max_height = img_size[1]
    im_width = img_size[0]
    width_sum = imim_len * img_size[0]
    dst_size = (width_sum + (imim_len - 1) * pad, max_height)
    dst: Image.Image = Image.new(mode, dst_size, dst_bg)
    cur = 0
    for im in imim:
        if im:
            dst.paste(im, (cur, 0))
        cur += im_width + pad
    return dst

def get_concat_v(imim_len: int, img_size: tuple[int, int], imim: Sequence[Image.Image | None], pad: int=0, mode='L', dst_bg=(0xff,))-> Image.Image:
    max_width = img_size[0]
    im_height = img_size[1]
    height_sum = imim_len * img_size[1]
    dst_size = (max_width, height_sum + (imim_len - 1) * pad)
    dst: Image.Image = Image.new(mode, dst_size, dst_bg)
    cur = 0
    for im in imim:
        if im:
            dst.paste(im, (0, cur))
        cur += im_height + pad
    return dst
class FunctionItem:
    def __init__(self, title: str, func: Callable, kwargs: dict[str, Any]={}):
        self.title = title
        self.func = func
        self.kwargs = kwargs
    def exec(self):
        self.func(**self.kwargs)
def get_options():
        return [
        FunctionItem('None', None),
        FunctionItem('save TM screenshots as TIFF', save_arc_pages, kwargs={'app_type': AppType.T}),
        FunctionItem('save MH screenshots as TIFF', save_arc_pages, kwargs={'app_type': AppType.M}),
        FunctionItem('T save_pages_as_4 png files into qpng dir.', save_qpng_pages),
        FunctionItem('M save_pages_as_4 png files into qpng dir.', save_qpng_pages, kwargs={'app_type': AppType.M}),
        FunctionItem('save_pages_as_TIFF', save_pages_as_tiff),
        FunctionItem('T convert_to_pdf', convert_to_pdf, kwargs={'layout':PdfLayout.a3lp, 'app_type': AppType.T}),
        FunctionItem('M convert_to_pdf', convert_to_pdf, kwargs={'layout':PdfLayout.a3lp, 'app_type': AppType.M}),
    ]
def main(options=get_options()):
    for n, option in enumerate(options):
        print(f"{n}. {option.title}")
    choice = int(input(f"Choose(0 to {len(options)}):"))
    breakpoint()
    if choice:
        options[choice].exec()

if __name__ == '__main__':
    main()
    '''import click
    @click.group
    def cli():
        pass
    '''