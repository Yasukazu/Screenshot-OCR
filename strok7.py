import csv

class Strok7:
	dic = [(0,)] * 7
	def __init__(self, f, t):
		self.f = int(f)
		self.t = int(t)
	def export(self):
		return self.f, self.t

strk7 = []

with open('abcdef-7.csv', encoding='utf8') as csv_file:
	csv_reader = csv.reader(csv_file)
	for row in csv_reader:
		seg = Strok7(*row)
		strk7.append(seg)

strk_dic = {}

for i, j in enumerate('abcdef'):
	strk_dic[j] = (strk7[i].export(), strk7[i + 1].export())

strk_dic['g'] = ((0, 1), (1, 1))

import numpy as np

np_strk_dict = {k: np.array(v, int) for (k, v) in strk_dic.items() }

if __name__ == '__main__':
	import pickle

	with open('np_strk_dict.pkl', 'wb') as wf:
		pickle.dump(np_strk_dict, wf)