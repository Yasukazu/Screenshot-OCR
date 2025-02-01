import csv

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

seg7_list = []
with open('7-seg.csv', encoding='utf8') as csv_file:
    csv_reader = csv.reader(csv_file)
    #next(csv_reader)  # skipping the header
    for i, row in enumerate(csv_reader):
        seg = Seg7(*row)
        assert seg.h == i
        seg7_list.append(seg)

assert len(seg7_list) == 16

class Digi7:
    dic = [(0,)] * 16
    def __init__(self, s: Seg7):
        self.dic[seg.h] = (s.a, s.b, s.c, s.d, s.e, s.f, s.g)

for seg in seg7_list:
    Digi7(seg)
digi7_list = [
    Digi7(seg) for seg in seg7_list
]
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

def get_strok(n):
    seg7 = seg7_list[n]
    seg7_dict = seg7.__dict__
    strks = []
    for k in 'abcdefg':
        if seg7_dict[k]:
            strks.append(np_strk_dict[k])
    return strks

if __name__ == '__main__':
    from pprint import pprint
    from PIL import Image, ImageDraw
    scale = 40
    offset = np.array([20, 20], int)
    for i in range(10):
        img = Image.new('L', (80, 160), (0xff,))
        drw = ImageDraw.Draw(img)
        strk = get_strok(i)
        for stk in strk:
            seq = [tuple(offset + st * scale) for st in stk]
            drw.line(seq, fill=(0,), width=20)
        img.save(f"digi-{i}.png", 'PNG')

    pprint(np_strk_dict)
    pprint(strk_dic)
    pprint(Digi7.dic)
    for u in seg7_list:
        print(u.a, u.b, u.c, u.d, u.e, u.f, u.g, u.h)