parser.add_argument(
	"--image_ext", nargs="+", default=[".png"]
)
parser.add_argument(
	"--image_dir",
	default="~/Documents/screenshots",
	type=Path,
)
parser.add_argument(
	"--shot_month",
	action="append",
	type=int,
	help='Choose Screenshot file by its month (MM part of [YYYY-MM-DD or YYYYMMDD]) included in filename stem. {Jan. is 01, Dec. is 12}(specified in a list like "[1,2,..]" )',
)
parser.add_argument(
	"files",
	nargs="*",
	help="Image file fullpaths to commit OCR or to get parameters.",
)
parser.add_argument(
	"--app_stem_end",
	default="taimee:_jp.co.taimee mercari:_jp.mercari.work.android",
	help='Screenshot image file name pattern of the screenshot to execute OCR:(specified in format as "<app_name1>:<stem_end1>,<stem_end2> ..." )',
)
parser.add_argument(
	"--image_area_param_section_stem",
	#env_var="IMAGE_AREA_PARAM_SECTION_STEM ",
	default="image_area_param",
)
parser.add_argument(
	"--app_border_ratio",
	default="taimee:2.2,3.2",
	nargs="*",
	help='Screenshot image file horizontal border ratio list of the app to execute OCR:(specified in format as "<app_name1>:<ratio1>,<ratio2> ..." )',
)
parser.add_argument(
	"--app_suffix",
	action="store_true",
	default=True,
	help='Screenshot image file name has suffix(sub extention) of the same as app name i.e. "<stem>.<suffix>.<ext>" (default: True)',
)

# parser.add_argument('--filename_pattern', action='append', default=['*{app_stem_end}{image_ext}'], help='Image files to commit OCR or to get parameters. Can be specified multiple times. Default is: *{app_stem_end}{image_ext}')
parser.add_argument(
	"--app",
	choices=[n.name.lower() for n in APP_NAME],
	help=f"Application name of the screenshot to execute OCR: choices={', '.join(n.name.lower() for n in APP_NAME)}",
)  #
# parser.add_argument('--toml', help=f'Configuration toml file name like {OCR_FILTER}')
parser.add_argument(
	"--save",
	help='Output path to save OCR text of the image file as TOML format into the image file name extention as ".ocr-<app_name>.toml"',
)
# parser.add_argument('--dir', help='Image dir of files: ./')
parser.add_argument(
	"--nth",
	type=int,
	default=1,
	help="Rank(default: 1) of files descending sorted(the latest, the first) by modified date as wildcard(*, ?)",
)
parser.add_argument(
	"--glob-max",
	type=int,
	default=60,
	help="Pick up file max as pattern found in TOML",
)
parser.add_argument("--show", action="store_true", help="Show images to check")
parser.add_argument(
	"--make",
	help='make a image area param config file from image in TOML format(i.e. this arg. makes not to use param configs in any config file;  specify image_area_param values like "--image_area_param heading:0,106,196,-1"',
)
parser.add_argument(
	"--bin_image", action="store_true", default=False, help="Use binarized image for OCR"
)
parser.add_argument(
	"--no-ocr", action="store_true", default=False, help="Do not execute OCR"
)
parser.add_argument(
	"--ocr-conf", type=int, default=55, help="Confidence threshold for OCR"
)
parser.add_argument("--psm", type=int, default=6, help="PSM value for Tesseract")
parser.add_argument(
	"--area_param_dir",
	help="Screenshot image area parameter config file directory",
	type=Path,
)
parser.add_argument(
	"--area_param_file",
	help='Screenshot image area parameter config file: format as INI or TOML(".ini" or ".toml" extention respectively): in [image_area_param.<app>] section, items as "<area_name>=[<p1>,<p2>,<p3>,<p4>]" (e.g. "heading=[0,106,196,-1]")',
	type=Path,
	default="image-area-param.ini",
)
parser.add_argument(
	"--ocr_filter_sqlite_db_name",
	default="ocr-filter.db",
	help="SQLite DB file is created under `image_dir`/{yyyy} directory(yyyy is like 2025)",
)
parser.add_argument(
	"--data_year",
	env_var="OCR_FILTER_DATA_YEAR",
	type=int,
	default=0,
	help="Year for DB data (like 2025)",
)
parser.add_argument(
	"--show_ocr_area",
	action="store_true",
	default=False,
	help="Show every area before commit OCR",
)
parser.add_argument(
	"--exclude_area_param_set",
	help=f"Exclude a set of image area parameter names : { {f'{n}' for n in list(ImageAreaParamName)} }",
	nargs='*',
	env_var="IMAGE_FILTER_AREA_PARAM_NAME_EXCLUDE_SET",
	)