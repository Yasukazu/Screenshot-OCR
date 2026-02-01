import typed_settings as tst
def image_area_param_names():
	from image_filter import ImageAreaParamName
	return list(ImageAreaParamName)
def app_names():
	from image_filter import APP_NAME
	return [n.name.lower() for n in APP_NAME]

@tst.settings
class MainSettings:
	"""
	Extract/OCR paystub text from an image file: Needs to specify the image file for OCR by files option (like '--files *.APP_NAME.png') or by shot_month option (like '--shot_month 1 2 3')
	"""
	# image_ext: str = tst.option(help='Image file extension', default='.png')
	# image_dir: str = tst.option(help='Directory of image files', default='~/github/screen/DATA/')
	app: str = tst.option(default='',
		help=f"Application name of the screenshot to execute OCR: choices={', '.join(app_names())}",
	)
	app_name_to_stem_end: dict[str, str] = tst.option(help='Dictionary of app name to stem end', default={'taimee': '_jp.co.taimee', 'mercari': '_jp.mercari.work.android'})
	image_ext: list[str] = tst.option(default=[".png"]
	)
	image_dir: str = tst.option(
		default="~/Documents/screenshots",
	)
	shot_month: list[int] = tst.option(default=[],
		help='Choose Screenshot file by its month (MM part of [YYYY-MM-DD or YYYYMMDD]) included in filename stem. {Jan. is 01, Dec. is 12}(specified in a list like "[1,2,..]" )',
	)
	files: list[str] = tst.option(default=[],
		help="Image file fullpaths to commit OCR or to get parameters.",
	)
	app_stem_end: dict[str, str] = tst.option(
		default={"taimee":"_jp.co.taimee", "mercari":"_jp.mercari.work.android"},
		help='Screenshot image file name endswith of the sclass APP_NAME(StrEnum): (specified in format as "<app_name1>:<stem_end1>,<stem_end2> ..." )',
	)
	image_area_param_section_stem: str = tst.option(
		default="image_area_param",
	)
	app_border_ratio: dict[str, list[float]] = tst.option(
		default={"taimee":[2.2,3.2]},

		help='Screenshot image file horizontal border ratio list of the app to execute OCR:(specified in format as "<app_name1>:<ratio1>,<ratio2> ..." )',
	)
	app_suffix: bool = tst.option(
		default=True,
		help='Screenshot image file name has suffix(sub extention) of the same as app name i.e. "<stem>.<suffix>.<ext>" (default: True)',
	)

	# parser.add_argument('--filename_pattern', action='append', default=['*{app_stem_end}{image_ext}'], help='Image files to commit OCR or to get parameters. Can be specified multiple times. Default is: *{app_stem_end}{image_ext}')

	# parser.add_argument('--toml', help=f'Configuration toml file name like {OCR_FILTER}')
	save: str = tst.option(default='',
		help='Output path to save OCR text of the image file as TOML format into the image file name extention as ".ocr-<app_name>.toml"',
	)
	# parser.add_argument('--dir', help='Image dir of files: ./')
	nth: int = tst.option(
		default=1,
		help="Rank(default: 1) of files descending sorted(the latest, the first) by modified date as wildcard(*, ?)",
	)
	glob_max: int = tst.option(
		default=60,
		help="Pick up file max as pattern found in TOML",
	)
	show: bool = tst.option(
		default=False,
		help="Show images to check"
	)
	make: bool = tst.option(
		default=False,
		help='make a image area param config file from image in TOML format(i.e. this arg. makes not to use param configs in any config file;  specify image_area_param values like "--image_area_param heading:0,106,196,-1"',
	)
	bin_image: bool = tst.option(default=False, help="Use binarized image for OCR"
	)
	no_ocr: bool = tst.option(default=False, help="Do not execute OCR"
	)
	ocr_conf: int = tst.option(default=55, help="Confidence threshold for OCR"
	)
	psm: int = tst.option(default=6, help="PSM value for Tesseract")
	area_param_dir: str = tst.option(default='',
		help="Screenshot image area parameter config file directory"
	)
	area_param_file: str = tst.option(
		help='Screenshot image area parameter config file: format as INI or TOML(".ini" or ".toml" extention respectively): in [image_area_param.<app>] section, items as "<area_name>=[<p1>,<p2>,<p3>,<p4>]" (e.g. "heading=[0,106,196,-1]")',
		default="image-area-param.ini"
	)
	ocr_filter_sqlite_db_name: str = tst.option(
		default="ocr-filter.db",
		help="SQLite DB file is created under `image_dir`/{yyyy} directory(yyyy is like 2025)"
	)
	data_year: int|None = tst.option(
		default=None,
		help="Year for DB data (like 2025)",
	)
	show_ocr_area: bool = tst.option(
		default=False,
		help="Show every area before commit OCR",
	)
	exclude_area_param_set: list[str] = tst.option(default=[],
		help=f"Exclude a set of image area parameter names : { {f'{n}' for n in image_area_param_names()} }",
	)
