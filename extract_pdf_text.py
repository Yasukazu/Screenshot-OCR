import os
from pathlib import Path
import pdfplumber

def extract_text(pdf_path):
	return pdfplumber.open(pdf_path).pages[0].extract_text()
	#tables = pdf.pages[num_page].find_tables()
if __name__ == '__main__':
	import sys
	from pdf_from_png import path_feeder
	for input_path, output_path in path_feeder(rng=range(1, 2)):
		text = extract_text(input_path)
		if text:
			with output_path.open('w') as wf:
				wf.write(text)

