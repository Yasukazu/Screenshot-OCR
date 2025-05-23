from enum import Enum
class ImageFill(Enum): # single element tuple for ImageDraw color
	BLACK = 0
	WHITE = 0xff
	@classmethod
	def invert(cls, fill):
		if fill == ImageFill.BLACK:
			return ImageFill.WHITE
		return ImageFill.BLACK