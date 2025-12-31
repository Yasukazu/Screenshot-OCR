from prompt_toolkit.completion import PathCompleter
from prompt_toolkit import prompt
from prompt_toolkit.completion import NestedCompleter
import os
from pathlib import Path
import re
from prompt_toolkit.shortcuts import choice

def _choice_example():
	result = choice(
		message="Please choose a dish:",
		options=[
			("pizza", "Pizza with mushrooms"),
			("salad", "Salad with tomatoes"),
			("sushi", "Sushi"),
		],
		default="salad",
	)
	print(f"You have chosen: {result}")

def data_dir_feeder(data_dir = Path('~/github/screen/DATA')):
	data_dir = data_dir.expanduser()
	for pt in data_dir.iterdir():
		if is_20yy(pt.name) and (sub_dirs:=mm_dir_filter(pt)):
			yield pt.name, sub_dirs

def choose_yyyy_mm_dir(data_dir=Path('~/github/screen/DATA')):
	data_dir_dict = {dir_name:dirs for dir_name, dirs in data_dir_feeder(data_dir)} # [d.name for d in 
	options = [(k, f"[{', '.join([v.name for v in vv])}]") for k,vv in data_dir_dict.items()]
	if len(options) == 1:
		year = options[0][0]
	else:
		year = choice(
			message = 'Choose a year',
			options = options
		)[0]
	options = [(m.name, f"Month {m.name}: {len(list(m.iterdir()))} images") for m in data_dir_dict[year]]
	month = choice(
		message = 'Choose a month',
		options = options
	)[0]
	for m in data_dir_dict[year]:
		if m.name == month:
			# print(f"Selected month: {m}")
			break
	return m # int(year), int(month)

def main():
	month = choose_yyyy_mm_dir()
	print(f"Selected is: year={month.parent.name},month={month.name}")

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