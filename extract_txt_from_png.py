import subprocess

cmd = 'tesseract' # -l jpn+eng 

def run_cmd(input_path, output_txt_path, lang='jpn+eng'):
	try:
		# OCRmyPDF command with optimization options
		command = [cmd, '-l', lang, input_path, output_txt_path] # '--pdf-renderer', 'hocr', '--optimize', '0', 
		
		# Execute the OCRmyPDF command
		subprocess.run(command, check=True)
		
		print(f" file:{output_txt_path} is generated from:{input_path}")
	except subprocess.CalledProcessError as e:
		print(f"tesseract error: {e}")
		
import os
from pathlib import Path

def path_feeder(from_=1, to=31, input_ext='.png', output_ext='.tact'): #rng=range(0, 31)):
	home_dir = os.path.expanduser('~')
	home_path = Path(home_dir)
	input_dir = home_path / 'Documents' / 'screen' / '202501'
	assert input_dir.exists()
	for day in range(from_, to + 1):
		input_filename = f'2025-01-{day:02}{input_ext}'
		input_fullpath = input_dir / input_filename
		if not input_fullpath.exists():
			continue
		input_path_noext, _ext = os.path.splitext(input_fullpath)
		output_path = Path(input_path_noext + output_ext)
		assert not output_path.exists()
		yield input_fullpath, output_path

if __name__ == '__main__':
	for input_path, output_path in path_feeder(3, 31):
		run_cmd(input_path, output_path)