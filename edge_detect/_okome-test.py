import dataclasses

import okome


@dataclasses.dataclass
class Foo:
	"""
	This is a comment for class `Foo`.
	"""

	a: int = 0
	""" This is valid comment for field that can be parsed by okome """
	b: str = ''
	"""
	Multi line comment
	also works!
	"""
	# KeyError: '__firstlineno__'
	@classmethod
	def okome_parse(cls):
		return okome.parse(AppSettings)

from image_filter_main_settings import AppSettings

foo = Foo.okome_parse()
print(f"{foo.name=}")
print(f"{foo.comment=}") # list of strings
for f in foo.fields:
	print(f"\t{f.name=}\t{f.comment=}") # list of strings