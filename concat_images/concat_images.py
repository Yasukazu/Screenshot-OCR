from PIL import Image

def get_concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst

if __name__ == '__main__':
    from sys import argv
    im1_name = argv[1]
    im2_name = argv[2]
    im1 = Image.open(im1_name)
    im2 = Image.open(im2_name)
    im3 = get_concat_v(im1, im2)
    im3.save(argv[3])