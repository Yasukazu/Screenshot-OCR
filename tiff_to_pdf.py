from PIL import Image, ImageSequence
import os

def tiff_to_pdf(tiff_path: str) -> str:
 
    pdf_path = tiff_path.replace('.tif', '.pdf')
    if not os.path.exists(tiff_path): raise Exception(f'{tiff_path} does not find.')
    image = Image.open(tiff_path)

    images = []
    for i, page in enumerate(ImageSequence.Iterator(image)):
        page = page.convert("L") # RGB
        images.append(page)
    if len(images) == 1:
        images[0].save(pdf_path)
    else:
        images[0].save(pdf_path, save_all=True,append_images=images[1:])
    return pdf_path

if __name__ == '__main__':
    from path_feeder import PathFeeder, FileExt
    feeder = PathFeeder(input_type=FileExt.TIFF)
    fullpath = feeder.dir / '2025-02-8x4.tif'
    tiff_to_pdf(str(fullpath))