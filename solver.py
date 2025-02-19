from collections import namedtuple
from sympy import symbols, solve, Symbol


# slvd = solve([fpd, fw, feh, few, fwt])
WPdEwEh = namedtuple('WPdEwEh', ['w', 'pd', 'ew', 'eh'])
class Solved:
	def __init__(self, n: int, pd: float, w: float):
		self.n = n
		self.w = int(w) or 1
		self.pd = int(pd) or 1
	
	@property
	def eh(self):
		return self.w * 2
		
	@property
	def wt(self):
		return self.pd + (self.w + self.pd) * self.n
		
	@property
	def ht(self):
		return 2 * self.pd + self.eh

	@property
	def scale(self)-> int:
		return self.w

	@property
	def box_size(self)-> tuple[int, int]:
		return self.wt, self.ht

	@property
	def offsets(self):
		return (((i * (self.w + self.pd)) + self.pd, self.pd) for i in range(self.n))
'''
n=2
pdr=0.3
er=0.8
fpd = w * pdr - pd
few = (w * er) - ew
feh = w * 2 - eh
fw = eh / 2 - w
fwt = pd + n * (w + pd) - wt
''' 
def solve_wt(ht: int, n: int, pdr=0.3, er=0.8)-> tuple[WPdEwEh, float]:
	'''returns wt'''
	# fw = (wt / ((1 + pdr) * n + pdr)) - w
	# wt = pd + (w + pd) * n
	w, ew, pd, eh, wt = symbols("w ew pd eh wt")
	#ht = eh + 2 * pd
	fpd = w * pdr - pd
	few = (w * er) - ew
	feh = w * 2 - eh
	fw = eh / 2 - w
	fwt = pd + n * (w + pd) - wt
	# feh = ht - 2 * pd - eh
	# feh = w * 2 - eh
	# feh = ht - 2 * pd - eh
	# w * 2 - eh = ht - 2 * pd - eh
	# w * 2 = ht - 2 * pd
	# w * 2 + 2 * pd = ht 
	# eh = 2 * w
	# w = eh / 2
	
	slvd = solve([fpd, fw, feh, few, fwt])
	return WPdEwEh(w=slvd[w], pd=slvd[pd], ew=slvd[ew], eh=slvd[eh]), slvd[wt]
# {eh: 0.689655172413793*wt, ew: 0.275862068965517*wt, pd: 0.103448275862069*wt, w: 0.344827586206897*wt}
'''
n=2
pdr=0.3
er=0.8
fpd = w * pdr - pd
few = (w * er) - ew
feh = w * 2 - eh
fw = eh / 2 - w
fht = eh + 2 * pd - ht
'''
def solve_ht(wt: int, n: int, pdr=0.3, er=0.8)-> tuple[WPdEwEh, float]:
	'''returns ht'''
	w, ew, pd, eh, ht = symbols("w ew pd eh ht")
	fpd = w * pdr - pd
	few = (w * er) - ew
	feh = w * 2 - eh
	fw = eh / 2 - w
	fht = eh + 2 * pd - ht
	# feh = w * 2 - eh
	# wt = pd + n * (w + pd)
	# wt - pd = n * (w + pd)
	# (wt - pd) / n = w + pd
	# (wt - pd ) / n - pd = w
	# fw = (wt - pd) / n - pd - w

	slvd = solve([fpd, fw, feh, few, fht])
	return WPdEwEh(w=slvd[w], pd=slvd[pd], ew=slvd[ew], eh=slvd[eh]), slvd[ht]


if __name__ == '__main__':
	from pprint import pp
	wt, ht = 180, 40 # wt, ht
	s_len = 2
	wpwh_w, solved_wt = solve_wt(ht, s_len)
	wpwh_h, solved_ht = solve_ht(wt, s_len)
	pp(solved_wt)
	pp(solved_ht)
