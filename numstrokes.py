from typing import Callable
from functools import lru_cache
import numpy as np
from numpy.typing import NDArray
from strok7 import SEGPATH_SLANT, get_segpath_for_c
from seg_7_digits import seg_7_array

NUMSTROKE_SLANT = 0.25

class NumStrokes:
	from seven_seg import SEVEN_SEG_SIZE
	f'''strokes[{SEVEN_SEG_SIZE}]: slanted strokes]'''
	def __init__(self, slant=SEGPATH_SLANT, max_cache=SEVEN_SEG_SIZE):
		self.slant: float = slant
		self.stroke: Callable[[int], NDArray] = lru_cache(maxsize=max_cache)(self._stroke)


	def _stroke(self, n: int)-> NDArray:
		slanted = np.array([get_segpath_for_c(c).slanted(self.slant) for c in seg_7_array[n]])
		return slanted

if __name__ == '__main__':
	from pprint import pp
	from seven_seg import SEVEN_SEG_SIZE
	num_strokes = NumStrokes()
	for i in range(SEVEN_SEG_SIZE):
		stroke = num_strokes.stroke(i)
		pp(stroke)
