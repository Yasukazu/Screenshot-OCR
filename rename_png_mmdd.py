# %%


# %%
from path_feeder import PathFeeder
feeder = PathFeeder()
for f in feeder.dir.glob('????.png'):
    less_stem = f.stem[2:]
    ext = f.suffix
    parent = f.parent
    new_name = parent / (less_stem + ext)
    breakpoint()
    f.rename(new_name)

f


