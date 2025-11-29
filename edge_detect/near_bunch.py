class NearBunchException(Exception):
	pass

class NoBunchException(Exception):
	pass

class NearBunch:
	def __init__(self, distance: int = 10):
		self.distance = distance
		self.elems: list[int] = []

	@property
	def bunch_count(self):
		return len(self.elems)

	def add(self, i: int) -> int:
		'''returns 1 if added'''
		if i < 0:
			raise ValueError("i must be non-negative") 
		if len(self.elems) == 0:
			self.elems.append(i)
			return 1
		else:
			last_elem = self.elems[-1]
			if i == last_elem:
				return 0
			if i - last_elem <= self.distance:
				self.elems.append(i)
				return 1
			else:
				raise NearBunchException("NearBunch over distance")



'''
class NotEnoughBunch(ValueError):
	pass

class BunchList:
	def __init__(self, iter: Iterator[int], distance: int = 5, max_bunch: int = 4):
		self.bunch_list: list[NearBunch] = [NearBunch(distance)]
		self.distance = distance
		self.max_bunch = max_bunch
		bunch = self.bunch_list[-1]
		for y in iter:
			try:
				bunch.add(y)
			except NearBunchError:
				if len(self.bunch_list) >= self.max_bunch:
					return # raise BunchOverFlow("Maximum number of bunches exceeded")
				bunch = NearBunch(self.distance)
				bunch.add(y)
				self.bunch_list.append(bunch)

	
	@property
	def main_bunch(self):
		return self.bunch_list[0]
	
	@property
	def trailing_bunches(self):
		return self.bunch_list[1:]
'''