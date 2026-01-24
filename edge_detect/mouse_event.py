from pathlib import Path
import sys
from typing import NamedTuple
import cv2
import numpy as np
cwd = Path(__file__).resolve().parent
sys.path.append(str(cwd.parent))
from set_logger import set_logger
logger = set_logger()

class QuitKeyException(Exception):
	pass

class NoMouseEvent(Exception):
	pass

class MouseEvent(NamedTuple):
	x: int
	y: int
	event: int
	flags: int

class MouseParam:
	def __init__(self, input_img_name):
		# event data as dict.
		self.mouseEvent = {"x":None, "y":None, "event":None, "flags":None}
		self._mouse_event: MouseEvent | None = None
		# set callback
		cv2.setMouseCallback(input_img_name, self.__CallBackFunc, None)
	
	def __CallBackFunc(self, eventType, x, y, flags, userdata):
		
		self._mouse_event = MouseEvent(x, y, eventType, flags)
		self.mouseEvent["x"] = x
		self.mouseEvent["y"] = y
		self.mouseEvent["event"] = eventType    
		self.mouseEvent["flags"] = flags    

	@property
	def data(self) -> MouseEvent:
		if not self._mouse_event:
			raise NoMouseEvent("No mouse event")
		return self._mouse_event

	@property
	def y(self) -> int:
		if not self._mouse_event:
			raise NoMouseEvent("No mouse event")
		return self._mouse_event.y

	@property
	def pos(self) -> tuple[int, int]:
		''' x, y '''
		if not self._mouse_event:
			raise NoMouseEvent("No mouse event")
		return (self._mouse_event.x, self._mouse_event.y)

	@property
	def event(self) -> int:
		if not self._mouse_event:
			raise NoMouseEvent("No mouse event")
		return self._mouse_event.event

	@property
	def flags(self) -> int:
		if not self._mouse_event:
			raise NoMouseEvent("No mouse event")
		return self._mouse_event.flags

	def getData(self):
		''' all mouse event data '''
		return self.mouseEvent
	
	def getEvent(self):
		''' mouse event '''
		return self.mouseEvent["event"]                

	def getFlags(self):
		''' mouse flags '''
		return self.mouseEvent["flags"]                

	def getX(self):
		''' mouse X co-od. '''
		return self.mouseEvent["x"]  

	def getY(self):
		''' mouse Y co-od. '''
		return self.mouseEvent["y"]  

	def getPos(self) -> tuple[int, int]:
		''' mouse (X, Y) co-od. '''
		return (self.mouseEvent["x"], self.mouseEvent["y"])
		
def get_area(window: str, image: np.ndarray,
	TL_BR_list = [] # TLpos = [0, 0], BRpos = [0, 0]
) -> list[tuple[int, int]]:
	''' returns TL_BR(Top-Left, Bottom-Right) tuple list.
	Quit key set(['Q', 'q', 17]) raises QuitKeyException '''
	usage = "Drag mouse to select a rectangle area from top-left to bottom-right, (Right click to reset the area), then hit Space key to add a bottom-right position, Esc key to unset last position, finally hit Enter key to submit"	
	print(usage)
	copy_image = image.copy()
	cv2.imshow(window, image)

	mouseData = MouseParam(window) #call back func.

	first_click = False
	is_rect = False
	is_l_button_down = False
	is_reset = False
	TLpos: tuple[int, int] = (0, 0) # top left
	BRpos: tuple[int, int] = (0, 0) # bottom right
	pos_list: list[tuple[int, int]] = [] # additional bottom right
	def redraw():
		nonlocal image
		image = copy_image.copy()
		if len(pos_list) < 2:
			return
		for p in range(len(pos_list)-1):
			tl = pos_list[p]
			br = pos_list[p + 1]
			cv2.rectangle(image, tl, br, 0, 1)
			logger.info("Redraw rectangle:%s, %s", tl, br)
		cv2.imshow(window, image)
	try:
		while not is_reset:
			key = cv2.waitKey(50)
			if key in [8, 127]: # BS or Del
				pos_list.pop()
				redraw()
			if key in [17]: # Esc
				raise QuitKeyException()
			elif key in [ord("\n"), ord("\r")]:
				is_reset = True
				break
			# show if left click
			try:
				match (data:=mouseData.data).event:
					case cv2.EVENT_LBUTTONUP:
						if is_l_button_down:
							# pos = data.pos # mouseData.getPos()
							BRpos = data.x, data.y # [0] = pos[0]
							if is_rect:
								image = copy_image.copy()
							cv2.rectangle(image, (TLpos[0], TLpos[1]), (BRpos[0], BRpos[1]), 0, 1)
							# second_click = True
							is_rect = True
							cv2.imshow(window, image)
							logger.info("Redraw rectangle: %s, %s", TLpos, BRpos)
							is_l_button_down = False
					case cv2.EVENT_LBUTTONDOWN:
						is_l_button_down = True
						pos = mouseData.getPos()
						if not first_click:
							TLpos = data.x, data.y # [0] = pos[0]
							# TLpos[0] = pos[0] TLpos[1] = pos[1]
							first_click = True
							logger.info("TLpos:%s", TLpos)
					case cv2.EVENT_MOUSEMOVE:
						if not first_click:
							image = copy_image.copy()
							pos = mouseData.getPos()
							image[pos[1], :] = 127
							image[:, pos[0]] = 127
							cv2.imshow(window, image)
							continue
						if is_l_button_down:
							pos = mouseData.getPos()
							BRpos = data.x, data.y # [0] = pos[0]
							if is_rect:
								image = copy_image.copy()
							cv2.rectangle(image, (TLpos[0], TLpos[1]), (BRpos[0], BRpos[1]), 0, 1)
							is_rect = True
							cv2.imshow(window, image)
							logger.info("Redraw rectangle:%s, %s", TLpos, BRpos)
						# image[TLpos[1]:BRpos[1], TLpos[0]:BRpos[0]] &= (255-7)

					# right click makes to reset
					case cv2.EVENT_RBUTTONDOWN:
						if not is_reset:
							first_click = False
							TLpos = BRpos = (0, 0)
							# for p in [TLpos, BRpos]: p[0] = p[1] = 0
							logger.info("Reset")
							is_reset = True
						continue
			except NoMouseEvent:
				pass
	finally:
		cv2.destroyWindow(window)
	return [TLpos, BRpos]

if __name__ == "__main__":
	from sys import argv
	#入力画像
	image = cv2.imread(argv[1], cv2.IMREAD_GRAYSCALE)
	
	#表示するWindow名
	window_name = "HEAD" + ":Left click to set TOp Left point, then keep pressing it(i.e. 'drag'), move mouse to set Bottom Right point, then unpress the left button, then type 's' to save, type 'q' to exit"
	
	#画像の表示
	TLpos = [0, 0]
	BRpos = [0, 0]
	get_area(window_name, image, TLpos, BRpos)
	print(f"TLpos: {TLpos}, BRpos: {BRpos}")
			

	print(f"Got area rectangle: {TLpos=},{BRpos=}")
'''マウスイベントの種類は以下の通りです．

EVENT_MOUSEMOVE
EVENT_LBUTTONDOWN
EVENT_RBUTTONDOWN
EVENT_MBUTTONDOWN
EVENT_LBUTTONUP
EVENT_RBUTTONUP
EVENT_MBUTTONUP
EVENT_LBUTTONDBLCLK
EVENT_RBUTTONDBLCLK
EVENT_MBUTTONDBLCLK
フラグの種類は以下の通りです．これは，マウスイベントが発生したときの状態を表しています．

EVENT_FLAG_LBUTTON
EVENT_FLAG_RBUTTON
EVENT_FLAG_MBUTTON
EVENT_FLAG_CTRLKEY
EVENT_FLAG_SHIFTKEY
EVENT_FLAG_ALTKEY
'''