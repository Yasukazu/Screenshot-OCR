import re
from datetime import datetime, time
from datetime import timedelta
from pprint import pp
from collections import namedtuple
from path_feeder import path_feeder

#gyo_kai_pat = re.compile(r'(?:業務開始){i:\*+}')
md_pat = re.compile(r'(\d+)月(\d+)日')
hhmm_pat = re.compile(r'(\d\d):(\d\d)')

MonthDay = namedtuple("MonthDay", ['month', 'day'])
HourMin = namedtuple("HourMin", ['hour', 'min'])

gyoumu_kaisi = '業務開始'
def read_kaisi(txt)->timedelta:
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
	for inp_fullpath, node, day in path_feeder(dir='tact'):
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
	main()
