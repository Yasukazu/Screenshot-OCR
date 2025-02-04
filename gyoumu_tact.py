import re
from pathlib import Path
from datetime import datetime, time
from datetime import timedelta
from pprint import pp
from collections import namedtuple
from path_feeder import path_feeder, input_dir

#gyo_kai_pat = re.compile(r'(?:業務開始){i:\*+}')
md_pat = re.compile(r'(\d+)月(\d+)日')
hhmm_pat = re.compile(r'(\d\d):(\d\d)')

MonthDay = namedtuple("MonthDay", ['month', 'day'])
HourMin = namedtuple("HourMin", ['hour', 'min'])


def read_head(txt, next_head = 'この店舗の募集状況')-> str:
	hh = []
	while (line := txt.readline()):
		if next_head in line:
			break
		hh.append(line)
	return ''.join(hh)


def read_until(txt, until_line = '差引支給額'):
	while (line := txt.readline()):
		if until_line in line:
			break
	sikyuu = line.rsplit()[-1].strip(' ¥')
	no_cms = sikyuu.replace(',', '')
	no_dots = no_cms.replace('.', '')
	return int(no_dots)

import csv

def write_csv(year=2025, month=1):
	ymstr = f"{year}{month:02}"
	out_path = input_dir / ymstr
	assert out_path.exists()
	out_csv = out_path / f"taimi-{ymstr}.csv"
	with out_csv.open('w') as out_file:
		writer = csv.writer(out_file)
		for inp_fullpath, node, day in path_feeder(year=year, month=month, direc='tact', input_ext='.txt', pad=True):
			head = ''
			sikyuu = 0
			if inp_fullpath:
				txt = inp_fullpath.open(encoding='utf8')	
				head = read_head(txt)
				sikyuu = (read_until(txt))
			writer.writerow([day, head, sikyuu])




gyoumu_kaisi = '業務開始'
def read_kaisi(txt)-> tuple[str, str]:
	while (line := txt.readline()):
		if gyoumu_kaisi in line:
			break
	assert line
	date_line = txt.readline()
	from_to_line = txt.readline()
	return date_line, from_to_line

FromToPat = namedtuple("FromToPat", ['pat', 'count'])
from_to_pat_4 = FromToPat(re.compile(r"(\d+):(\d+)\s+(\d+):(\d+)"), 4)
from_to_pat_3 = FromToPat(re.compile(r"(\d+)\s+(\d+):(\d+)"), 3)


def get_f_t_times(s)-> tuple[datetime.time, datetime.time]:
	f_time = t_time = None
	if (mt:=from_to_pat_4.pat.match(s)):
		grps = mt.groups()
		assert len(grps) == 4
		f_time = time(int(grps[0]), int(grps[1]))
		t_time = time(int(grps[2]), int(grps[3]))
	elif (mt:=from_to_pat_3.pat.match(s)):
		grps = mt.groups()
		assert len(grps) == 3
		h = grps[0][:-2]
		m = grps[0][-2:]
		f_time = time(int(h), int(m))
		t_time = time(int(grps[1]), int(grps[2]))
	return f_time, t_time 

def main(year=2025):
	for inp_fullpath, node, day in path_feeder(direc='tact'):
		txt = inp_fullpath.open(encoding='utf8')
		date_line, f_t_line = read_kaisi(txt)
		ft, tt = get_f_t_times(f_t_line)
		td = timedelta(hours=tt.hour - ft.hour, minutes=tt.minute - ft.minute)
		print(f"{node}:: from: {hh_mm(ft)}, until: {hh_mm(tt)}, dur: {hh_mm(td)}")

def hh_mm(tt):
	ss = str(tt).split(':')
	hh = ss[0]
	mm = ss[1]
	assert ss[2] == '00'
	return ':'.join([hh, mm])

def get_date(txt):
	while (line:= txt.readline()):
		md = md_pat.match(line)
		if md:
			month = int(md[1])
			day = int(md[2])
			return month, day

def get_time(txt):
	while (line:= txt.readline()):
		hm = hhmm_pat.match(line)
		if hm:
			hour = int(hm[1])
			minute = int(hm[2])
			return hour, minute

if __name__ == '__main__':
	write_csv(year=2025, month=1)