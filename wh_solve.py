from sympy import symbols, solve
from collections import namedtuple
WHSolved = namedtuple('WHSolved', ['eh', 'ew', 'pd', 'wt', 'w'])
class WHSolve:
    def __init__(self, whs: WHSolved, n: int, ht: int):
        self.wt = int(whs.wt)
        self.n = n
        self.w = int(whs.w)
        self.ew = int(whs.ew)
        self.eh = int(whs.eh)
        self.pd = int(whs.pd)
        self.ht = ht
    
    @property
    def scale(self)-> int:
        return self.ew

    @property
    def box_size(self)-> tuple[int, int]:
        return self.wt, self.ht

    @property
    def offsets(self):
        return (((i * (self.w + self.pd)) + self.pd, self.pd) for i in range(self.n))

def solve_wh(ht: int, n: int, pdr=0.3, er=0.8)-> WHSolve:
	# fw = (wt / ((1 + pdr) * n + pdr)) - w
	# wt = pd + (w + pd) * n
	pd = ht * pdr
	eh = ht - 2 * pd
	wt, w, ew = symbols("wt w ew")
	
	fwt = pd + n * (w + pd) - wt
	few = (w * er) - ew
	feh = w * 2 - eh
	
	slvd = solve([fwt, few, feh])
	whs = WHSolved(eh=eh, ew=slvd[ew], pd=pd, wt=slvd[wt], w=slvd[w])
	return WHSolve(whs, n=n, ht=ht)

# [{er: 0.300000000000000, ew: 24.0000000000000, w: 80.0000000000000, wt: 208.000000000000}]

if __name__ == '__main__':
	from pprint import pp
	solved = solve_wh()
	pp(solved)
'''
ht=80
pdr=0.3
n=2
er=0.8
pd = ht * pdr
eh = ht - 2 * pd
wt, w, ew = symbols("wt w ew")
fwt = pd + n * (w + pd) - wt
few = (w * er) - ew
feh = w * 2 - eh
'''