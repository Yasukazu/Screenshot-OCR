import pickle
def conv(i):
	return 0 if i == '' else int(i)
class Seg7:
	def __init__(self, a, b, c, d, e, f, g, h):
		self.a = conv(a)
		self.b = conv(b)
		self.c = conv(c)
		self.d = conv(d)
		self.e = conv(e)
		self.f = conv(f)
		self.g = conv(g)
		self.h = conv(h)
def load_seg7(pkl='7-seg.pkl'):
	seg7_list = []
	rf = open(pkl, 'rb').read()
	csv_reader = pickle.loads(rf)
	for i, row in enumerate(csv_reader):
		seg = Seg7(*row)
		assert seg.h == i
		seg7_list.append(seg)
	assert len(seg7_list) == 16
	return seg7_list

seg7_list = load_seg7()

assert len(seg7_list) == 16
NP_STRK_DICT_PKL = 'np_strk_dict.pkl'
def load_np_strk_dic(pklf=NP_STRK_DICT_PKL):
	with open(pklf, 'rb') as buf:
		return pickle.load(buf)

np_strk_dict = load_np_strk_dic() # {k: np.array(v, int) for (k, v) in strk_dic.items() }

def get_strok(n):
	seg7 = seg7_list[n]
	seg7_dict = seg7.__dict__
	strks = []
	for k in 'abcdefg':
		if seg7_dict[k]:
			strks.append(np_strk_dict[k])
	return strks

STROKE_LIST_PKL = 'stroke_list.pkl'

def get_stroke_list():
	with open(STROKE_LIST_PKL, 'rb') as rf:
		lst = pickle.load(rf)
	return lst

if __name__ == '__main__':
	strk_list = []
	for n in range(len(np_strk_dict)):
		strk_list.append(get_strok(0))
	with open(STROKE_LIST_PKL, 'wb') as wf:
		pickle.dump(strk_list, wf)
