import os, glob
from pathlib import Path
year = 2025
month = 1
def rename_files(dir):
    mmdd_png_files = glob.glob(f"{dir}/01??.png")
    for df in mmdd_png_files:
        d, fe = os.path.split(df)
        dir = Path(d)
        f, e = os.path.splitext(fe)
        mm = f[:2]
        dd = f[2:]
        nm = str(dir / f'2025-{mm}-{dd}{e}')
        os.rename(df, nm)

if __name__ == '__main__':
    from tile import convert_to_pdf
    direc = Path(f"{year}{month:02}")
    assert direc.exists()
    files = direc.glob("2025-01-??.png")
    names = sorted(files)
    out = f"{year}-{month:02}-taimi.pdf"
    #convert_to_pdf(names=names, fullpath=out)
    for name in names:
        day_s = str(name)[-6:-4]
        day = int(day_s)
        assert f"{day:02}" == day_s
        out = f"tai-{year}-{month:02}-{day:02}.pdf"
        convert_to_pdf(names=[name], fullpath=out)
    #rename_files('202501')
