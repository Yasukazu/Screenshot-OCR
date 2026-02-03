from dataclasses import dataclass, field
# import typed_settings as tst
def image_area_param_names():
	from image_filter import ImageAreaParamName
	return list(ImageAreaParamName)
def app_names():
	from image_filter import APP_NAME
	return [n.name.lower() for n in APP_NAME]

@dataclass
class MainSettings:
	"""
	Extract/OCR paystub text from an image file: Needs to specify the image file for OCR by files option (like '--files *.APP_NAME.png') or by shot_month option (like '--shot_month 1 2 3')
	"""
	# image_ext: str = help='Image file extension', default='.png')
	# image_dir: str = help='Directory of image files', default='~/github/screen/DATA/')
	app: str = ''
	f"""Application name of the screenshot to execute OCR: choices={', '.join(app_names())}"""
	app_name_to_stem_end: dict[str, str] = field(default_factory=lambda: {'taimee': '_jp.co.taimee', 'mercari': '_jp.mercari.work.android'})
	"""Dictionary of app name to stem end"""
	image_ext: list[str] = field(default_factory=lambda: [".png"])
	"""Image file extension, starting with dot (e.g. ".png")"""
	image_dir: str = "~/Documents/screenshots"
	"""Image file root directory"""
	shot_month: list[int] = field(default_factory=list)
	"""Choose Screenshot file by its month (MM part of [YYYY-MM-DD or YYYYMMDD]) included in filename stem. {Jan. is 01, Dec. is 12}(specified in a list like "[1,2,..]"""
	files: list[str] = field(default_factory=list)
	"""Image file fullpaths to commit OCR or to get parameters."""
	app_stem_end: dict[str, str] = field(default_factory=lambda: {"taimee":"_jp.co.taimee", "mercari":"_jp.mercari.work.android"})
	"""Screenshot image file name endswith of the sclass APP_NAME(StrEnum): (specified in format as "<app_name1>:<stem_end1>,<stem_end2> ..." )"""
	image_area_param_section_stem: str = "image_area_param"
	app_border_ratio: dict[str, list[float]] = field( default_factory=lambda:{"taimee":[2.2,3.2]})
	"""Screenshot image file horizontal border ratio list of the app to execute OCR:(specified in format as "<app_name1>:<ratio1>,<ratio2> ..." )"""
	app_suffix: bool = True
	"""Screenshot image file name has suffix(sub extention) of the same as app name i.e. "<stem>.<suffix>.<ext>" (default: True)"""
	save: str = ''
	"""Output path to save OCR text of the image file as TOML format into the image file name extention as '.ocr-<app_name>.toml' """
	nth: int =1
	"""Rank(default: 1) of files descending sorted(the latest, the first) by modified date as wildcard(*, ?)"""
	glob_max: int = 60
	"""Pick up file max as pattern found in TOML"""
	show: bool = False
	"""Show images to check"""
	make: bool = False
	"""make a image area param config file from image in TOML format(i.e. this arg. makes not to use param configs in any config file;  specify image_area_param values like "--image_area_param heading:0,106,196,-1"""
	bin_image: bool = False
	"""Use binarized image for OCR"""
	no_ocr: bool = False
	"""Do not execute OCR"""
	ocr_conf: int = 55
	"""Confidence threshold for OCR"""
	psm: int = 6
	"""PSM value for Tesseract"""
	area_param_dir: str = ''
	"""Screenshot image area parameter config file directory"""
	area_param_file: str = "image-area-param.ini"
	"""Screenshot image area parameter config file: format as INI or TOML(".ini" or ".toml" extention respectively): in [image_area_param.<app>] section, items as "<area_name>=[<p1>,<p2>,<p3>,<p4>]" (e.g. "heading=[0,106,196,-1]") """
	ocr_filter_sqlite_db_name: str = "ocr-filter.db"
	"""SQLite DB file is created under `image_dir`/{yyyy} directory(yyyy is like 2025)"""
	data_year: int|None = None
	"""Year for DB data (like 2025)"""
	show_ocr_area: bool = False
	"""Show every area before commit OCR"""
	exclude_area_param_set: list[str] = field(default_factory=list)
	f"""Exclude a set of image area parameter names : { {f'{n}' for n in image_area_param_names()} }"""