from prompt_toolkit.completion import PathCompleter
from prompt_toolkit import prompt
from prompt_toolkit.completion import NestedCompleter
import os
from pathlib import Path
import re

def data_dir_feeder(data_dir = Path('~/github/screen/DATA')):
	data_dir = data_dir.expanduser()
	for pt in data_dir.iterdir():
		if is_20yy(pt.name) and (sub_dirs:=mm_dir_filter(pt)):
			yield pt.name, sub_dirs

def main(data_dir=Path('~/github/screen/DATA')):
	dir_sub = {}
	for dir_name, sub_dirs in data_dir_feeder(data_dir):
		dir_sub[dir_name] = {d.name:None for d in sub_dirs}
	completer = NestedCompleter.from_nested_dict(dir_sub)
	''' dir_sub
			'show': {
				'version': None,
				'clock': None,
				'ip': {
					'interface': {'brief'}
				}
			},
			'exit': None,
		})'''
	text = prompt('    # ', completer=completer)
	print('You said: %s' % text)

def mm_dir_filter(dir: Path) -> list[Path]:
	return  [d for d in dir.iterdir() if is_mm(d.name) and has_image_file(d)]

def is_20yy(s):
	return bool(re.match(r'^20\d{2}$', s))

def is_mm(s: str):
	try:
		return 1 <= int(s) <= 12
	except ValueError:
		return False

def has_image_file(p: Path) -> bool:
	return any(f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp'] for f in p.iterdir())
			
if __name__ == '__main__':
	main()