from typing import Iterator
from prompt_toolkit.completion import PathCompleter
from prompt_toolkit import prompt
from prompt_toolkit.completion import NestedCompleter
import os
from pathlib import Path
import re
from prompt_toolkit.shortcuts import choice

class ImageDirChecker:
	image_file_ext_list=['.jpg', '.jpeg', '.png', '.bmp']
	date_pattern = re.compile(r'\d{4}-\d\d-\d\d') 

	def __init__(self, suffix_list = ['.taimee']):
		self.suffix_subset = set([('.' + elem) if not elem.startswith('.') else elem for elem in suffix_list])

	def get_suffix_subset(self):
		return self.suffix_subset

	def matches(self, data_dir: Path, check_date_pattern=True, date_split_pattern=r"[_ ]") -> Iterator[tuple[Path, list[tuple[str, str] | str]]]:
		if check_date_pattern:
			def check_date(stem: str):
				for elem in re.split(date_split_pattern, stem):
					if self.date_pattern.match(elem):
						return elem
		else:
			check_date = None
		data_dir = Path(data_dir).expanduser()
		for root, dirs, files in data_dir.walk(follow_symlinks=True):
			date_list = []
			for file in files:
				if set((root / file).suffixes[:-1]) >= self.get_suffix_subset():
					try:
						if (date:= check_date(file.rsplit('.', 1)[0])):
							date_list.append((file, date))
					except TypeError:
							date_list.append((file, ))
					if date_list:
						yield root, date_list

def has_image_file(p: Path, image_file_ext_list=['.jpg', '.jpeg', '.png', '.bmp'], suffix_subset={'.mercari', '.work'}) -> bool:
	return any(f.suffix.lower() in image_file_ext_list and set(f.suffixes) >= suffix_subset for f in p.iterdir() if f.is_file() and f.stat().st_size > 0)


def date_dir_feeder(data_dir:str | Path = '~/Documents/screen/') -> Iterator[tuple[str, list[Path]]]:
	data_dir = Path(data_dir).expanduser()
	for pt in data_dir.iterdir():
		if is_yyyy(pt.name) and (sub_dirs:=mm_dir_filter(pt)):
			yield pt.name, sub_dirs

def date_dir_walker(data_dir:str | Path = '~/Documents/screen/', dir_checker=ImageDirChecker()) -> Iterator[tuple[str, list[Path]]]:
	data_dir = Path(data_dir).expanduser()
	for root, dirs, files in data_dir.walk():
		for dir in dirs:
			matching_files = [f.name for f in files if dir_checker.matches(dir)]
			if dir_checker.matches(dir): # if is_yyyy(root.name) and (sub_dirs:=mm_dir_filter(root / dir)) and dir_checker.matches(root / dir):
					yield root, dir

class NoValidChoiceException(Exception):
	pass

def choose_yyyy_mm_dir(data_dir:str | Path='~/Pictures', data_dir_feeder=date_dir_feeder, image_file_ext_list=['.jpg', '.jpeg', '.png'], suffix_subset={'.mercari', '.work'}) -> Path | None:
	''' choose files in data_dir(directory structure as YYYY/MM/ where MM is [01,02, ..., 12] and each MM directory contains image files with specified extensions and suffixes)
	image_file_ext_list: file extention starting with dot (e.g., ".jpg")
	suffix_subset: file suffix starting with dot (e.g., ".work")'''
	data_dir_dict = {dir_name:dirs for dir_name, dirs in data_dir_feeder(data_dir)} # [d.name for d in 
	options = [(k, f"[{', '.join([v.name for v in vv])}]") for k,vv in data_dir_dict.items()]
	if not options:
		raise NoValidChoiceException("No valid years found")
	if len(options) == 1:
		year = options[0][0]
	else:
		year = choice(
			message = 'Choose a year',
			options = options
		)[0]
	mon_len_dict = {}
	for m in data_dir_dict[year]:
		selected_image_files = [f for f in m.iterdir() if f.is_file() and f.stat().st_size > 0 and f.suffix.lower() in image_file_ext_list and set(f.suffixes) >= suffix_subset]
		if selected_image_files:
			mon_len_dict[m.name] = len(selected_image_files)
	if not mon_len_dict:
		raise NoValidChoiceException("No valid months found with image files")
	options = [(m, f"Month {m}: {length} image files") for m, length in mon_len_dict.items()]
	month = choice(
		message = 'Choose a month',
		options = options
	)[0]
	m = None
	for m in data_dir_dict[year]:
		if m.name == month:
			# print(f"Selected month: {m}")
			break
	return data_dir_dict[m] if m else None # int(year), int(month)


def has_image_file(p: Path, image_file_ext_list=['.jpg', '.jpeg', '.png', '.bmp'], suffix_subset={'.mercari', '.work'}) -> bool:
	return any(f.suffix.lower() in image_file_ext_list and set(f.suffixes) >= suffix_subset for f in p.iterdir() if f.is_file() and f.stat().st_size > 0)

def mm_dir_filter(dir: Path, image_file_ext_list=['.jpg', '.jpeg', '.png'], suffix_subset={'.mercari', '.work'}, file_filter=has_image_file) -> list[Path]:
	return [d for d in dir.iterdir() if is_mm(d.name) and file_filter(d, image_file_ext_list, suffix_subset)]

def is_yyyy(s: str):
	try:
		return 1950 <= int(s) < 2100
	except ValueError:
		return False
	# return len(s) == 4 and s[:2] in ['19', '20'] and s[2].isdigit() and s[3].isdigit() #bool(re.match(r'^20\d{2}$', s))

def is_mm(s: str):
	try:
		return 1 <= int(s) <= 12
	except ValueError:
		return False

			
def main(dir_name, suffix_set:str):
	image_dir_checker = ImageDirChecker(set(suffix_set.split(',')))
	for match in image_dir_checker.matches(Path(dir_name)):
		print(match)
	'''month = choose_yyyy_mm_dir(suffix_subset={'.taimee'})
	if month:
		print(f"Selected is: year={month.parent.name},month={month.name}")
	else:
		print("No month is selected.")'''

if __name__ == '__main__':
	import sys
	main(*sys.argv[1:])