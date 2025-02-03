import os
from pathlib import Path
import pdfplumber

home_dir = os.path.expanduser('~')
home_path = Path(home_dir)
pdf_path = home_path / 'Documents' / 'screen' / '202501'
assert pdf_path.exists()
pdf_filename = '2025-01-01.pdf'
pdf_path = pdf_path / pdf_filename
assert pdf_path.exists()
pdf_path_noext, _ext = os.path.splitext(pdf_path)
txt_path = Path(pdf_path_noext + '.txt')
assert not txt_path.exists()
with pdfplumber.open(pdf_path) as pdf:
    num_page = 0
    text = pdf.pages[num_page].extract_text()
    with txt_path.open('w', encoding='utf8') as wf:
        wf.write(text)
    #tables = pdf.pages[num_page].find_tables()