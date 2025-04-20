import os
from contextlib import closing
import sqlite3
from dotenv import load_dotenv
load_dotenv()
from path_feeder import input_dir_root

YEAR = 2025

def get_db_name():
    try:
        return os.environ['TXT_LINES_DB']
    except KeyError:
        return 'txt_lines.sqlite'


def get_table_name(month: int):
    return "text_lines-%02d" % month #:02}"

create_tbl_sql = "CREATE TABLE if not exists `{}` (`app` INTEGER, `day` INTEGER, `wages` INTEGER, `title` TEXT, `stem` TEXT, `txt_lines` BLOB, PRIMARY KEY (app, day))"

def create_tbl_if_not_exists(tbl_name: str):
    con = connect()
    with closing(con.cursor()) as cur:
        cur.execute(create_tbl_sql.format(tbl_name))
    con.commit()

sqlite_fullpath = input_dir_root / str(YEAR) / get_db_name()

_conn: list[sqlite3.Connection] = []

def connect():
    if _conn:
        return _conn[0]
    if not sqlite_fullpath.exists():
        raise ValueError(f"sqlite fullpath:`{sqlite_fullpath}` does not exist!")
    if sqlite3.threadsafety == 3:
        check_same_thread = False
    else:
        check_same_thread = True
    _conn.append(sqlite3.connect(str(sqlite_fullpath), check_same_thread=check_same_thread))
    return _conn[0]

def clear_txt_lines_of(month: int):
    tbl_name = get_table_name(month)
    clear_txt_lines_sql = f'UPDATE `{tbl_name}` SET `txt_lines` = NULL;'
    breakpoint()
    conn = connect()
    with closing(conn.cursor()) as cur:
        cur.execute(clear_txt_lines_sql)
    conn.commit()

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
    import csv, re, sys, os
    cli()
    sys.exit(0)
    month = int(sys.argv[1])
    each_field_patt = re.compile(r"`(\w+)`")
    each_field = each_field_patt.findall(create_tbl_sql)
    each_field_without_last = each_field[:-1]
    csv_fullpath = sqlite_fullpath.parent / (sqlite_fullpath.stem + '.csv')
    with csv_fullpath.open('w') as out_csv:
        writer = csv.writer(out_csv)
        writer.writerow(each_field_without_last)
        conn = connect()
        with closing(conn.cursor()) as cur:
            table_name = get_table_name(month)
            sql = f"SELECT {','.join(each_field_without_last)} FROM `{table_name}` ORDER BY `day`"
            rr = cur.execute(sql)
            for r in rr:
                writer.writerow(r)

        
