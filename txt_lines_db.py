import os
from contextlib import closing
import sqlite3
from dotenv import load_dotenv
load_dotenv()
from path_feeder import input_dir_root
from logging import getLogger

logger = getLogger(__name__)
YEAR = os.environ['SCREEN_YEAR']

def get_db_name():
    try:
        return os.environ['TXT_LINES_DB']
    except KeyError:
        return 'txt_lines.sqlite'

TABLE_NAME_FORMAT = "text_lines-{:02}"
def get_table_name(month: int):
    return TABLE_NAME_FORMAT.format(month)#:02}"
    #return "text_lines-%02d" % month #:02}"

CREATE_TABLE_SQL = "CREATE TABLE if not exists '{}' (`app` INTEGER, `day` INTEGER, `wages` INTEGER, `title` TEXT, `stem` TEXT, `txt_lines` BLOB, PRIMARY KEY (app, day))"

def sqlite_fullpath(dir=input_dir_root, year=YEAR, db_name=get_db_name()):
    if not dir:
        raise ValueError("input_dir_root is not set!")
    if not year:
        raise ValueError("YEAR is not set!")
    db_fullpath = dir / year / db_name
    return db_fullpath  

def create_tbl_if_not_exists(tbl_name: str, db_fullpath=sqlite_fullpath()):
    con = connect(db_fullpath=db_fullpath)
    with closing(con.cursor()) as cur:
        cur.execute(CREATE_TABLE_SQL.format(tbl_name))
    con.commit()

_conn: list[sqlite3.Connection] = []

def connect(db_fullpath=sqlite_fullpath()):
    if _conn:
        return _conn[0]
    #if not db_fullpath.exists():raise ValueError(f"`{db_fullpath=}` does not exist!")
    if sqlite3.threadsafety == 3:
        check_same_thread = False
    else:
        check_same_thread = True
    _conn.append(sqlite3.connect(str(db_fullpath), check_same_thread=check_same_thread))
    return _conn[0]

def clear_txt_lines_of(month: int):
    tbl_name = get_table_name(month)
    clear_txt_lines_sql = f'UPDATE `{tbl_name}` SET `txt_lines` = NULL;'
    breakpoint()
    conn = connect()
    with closing(conn.cursor()) as cur:
        cur.execute(clear_txt_lines_sql)
    conn.commit()

def init():
    db_fullpath = sqlite_fullpath()
    if not db_fullpath.exists():
        db_fullpath.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Database path created: %s", db_fullpath.parent) 

import click

@click.group()
def cli():
    pass

@cli.command()
@click.argument('month')
def clear_txt_lines(month: str):
    m = int(month)
    clear_txt_lines_of(m)
    
if __name__ == '__main__':
    cli()
