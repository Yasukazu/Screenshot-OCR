from typing import Callable, Iterator
from functools import lru_cache
import numpy as np
from numpy.typing import NDArray
from strok7 import SEGPATH_SLANT, get_segpath_for_c
from seg_7_digits import seg_7_array

NUMSTROKE_SLANT = 0.25

class NumStrokes:
	from seven_seg import SEVEN_SEG_SIZE
	f'''strokes[{SEVEN_SEG_SIZE}]: slanted strokes]'''
	def __init__(self, slant=SEGPATH_SLANT, max_cache=SEVEN_SEG_SIZE, scale: float=1.0, offset: tuple[int, int]=(0, 0)):
		self.slant: float = slant
		self.scale = scale
		self.offset: NDArray = np.array(offset)
		self.pure_strokes: Callable[[int], NDArray] = lru_cache(maxsize=max_cache)(self._strokes)
		self.strokes: Callable[[int], NDArray] = lru_cache(maxsize=max_cache)(self._scaled_offset_strokes)

	def _strokes(self, n: int)-> NDArray:
		slanted = np.array([get_segpath_for_c(c).slanted(self.slant) for c in seg_7_array[n]], np.float16)
		return slanted

	def _scaled_offset_strokes(self, n: int)-> list[tuple[int, int]]:
		strokes = [my_round2(st * self.scale + self.offset) for st in self._strokes(n)]
		return strokes



def my_round2(x, decimals=0):
    return np.sign(x) * np.floor(np.abs(x) * 10**decimals + 0.5) / 10**decimals

if __name__ == '__main__':
	from pprint import pp
	from seven_seg import SEVEN_SEG_SIZE
	num_strokes = NumStrokes()
	for i in range(SEVEN_SEG_SIZE):
		stroke = num_strokes.strokes(i)
		pp(stroke)
