import sympy
from sympy import symbols, Symbol, Subs, expand, factor, solve
from collections import namedtuple
WHSolved = namedtuple('WHSolved', ['ew', 'w', 'wt'])

def solve_wh(ht=80, pdr=0.3, n=2, er=0.8):
	pd = ht * pdr
	eh = ht - 2 * pd
	wt, w, ew = symbols("wt w ew")
	
	fw = (wt / ((1 + pdr) * n + pdr)) - w
	few = (w * er) - ew
	feh = ew * 2 - eh
	fpd = (w * pdr) - pd
	
	solved = solve([fw, few, feh, fpd])
	return(solved)

# [{er: 0.300000000000000, ew: 24.0000000000000, w: 80.0000000000000, wt: 208.000000000000}]

if __name__ == '__main__':
	from pprint import pp
	solved = solve_wh()
	pp(solved)
