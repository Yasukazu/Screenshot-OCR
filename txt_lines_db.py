import os
import sqlite3
from path_feeder import input_dir_root

def get_db_name():
    try:
        return os.environ['TXT_LINES_DB']
    except KeyError:
        return 'txt_lines.sqlite'


def get_table_name(month: int):
    return f"text_lines-{month:02}"

sqlite_fullpath = input_dir_root / get_db_name()
if not sqlite_fullpath.exists():
    raise ValueError(f"sqlite fullpath:`{sqlite_fullpath}` does not exist!")
if sqlite3.threadsafety == 3:
    check_same_thread = False
else:
    check_same_thread = True
conn = sqlite3.connect(str(sqlite_fullpath), check_same_thread=check_same_thread)