import subprocess

def run_ocrmypdf(input_path, output_pdf_path, lang='jpn', image_dpi=300):
    try:
        # OCRmyPDF command with optimization options
        command = ['ocrmypdf', '-l', lang, input_path, output_pdf_path, '--image-dpi', str(image_dpi)] # '--pdf-renderer', 'hocr', '--optimize', '0', 
        
        # Execute the OCRmyPDF command
        subprocess.run(command, check=True)
        
        print(f"PDF file:{output_pdf_path} is generated from:{input_path}")
    except subprocess.CalledProcessError as e:
        print(f"OCRmyPDF error: {e}")
        
if __name__ == '__main__':
    # Example usage
    import os
    from pathlib import Path
    home_dir = os.path.expanduser('~')
    home_path = Path(home_dir)
    input_path = home_path / 'Documents' / 'screen' / '202501'
    assert input_path.exists()
    input_filename = '2025-01-02.png'
    input_path = input_path / input_filename
    assert input_path.exists()
    input_path_noext, _ext = os.path.splitext(input_path)
    pdf_path = Path(input_path_noext + '.pdf')

    run_ocrmypdf(input_path, pdf_path)