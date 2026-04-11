from edge_detect.image_filter_main_settings import print_toml_template

if __name__ == '__main__':
	from argparse import ArgumentParser
	from sys import exit as sys_exit
	from sys import argv
	if len(argv) < 2:
		argv += ['-h']
	parser = ArgumentParser(prog="Screenshot-OCR", description="configuration file=main-settings.toml", epilog="=== End of help ===")
	parser.add_argument('--print-template', action='store_true', help='print TOML template of MainSettings; remove leading "#" to specify any item')
	parser.add_argument('--as-class', action='store_false', help='print TOML template of MainSettings as class')
	args = parser.parse_args()
	if args.print_template:
		print_toml_template(as_class=args.as_class)
		sys_exit(0)