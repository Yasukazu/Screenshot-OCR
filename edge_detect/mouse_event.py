from pathlib import Path
import sys
from dataclasses import dataclass # from typing import NamedTuple
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

@dataclass
class MouseData: #(NamedTuple):
	x: int
	y: int
	event: int
	flags: int
	@property
	def pos(self):
		return self.x, self.y

class MouseParam:
	def __init__(self, input_img_name):
		# event data as dict.
		self.mouseEvent = {"x":None, "y":None, "event":None, "flags":None}
		self._mouse_data: MouseData | None = None
		# set callback
		cv2.setMouseCallback(input_img_name, self.__CallBackFunc, None)
	
	def __CallBackFunc(self, eventType, x, y, flags, userdata):
		
		self._mouse_data = MouseData(x, y, eventType, flags)
		self.mouseEvent["x"] = x
		self.mouseEvent["y"] = y
		self.mouseEvent["event"] = eventType    
		self.mouseEvent["flags"] = flags    

	@property
	def data(self) -> MouseData:
		if not self._mouse_data:
			raise NoMouseEvent("No mouse event")
		return self._mouse_data

	@property
	def y(self) -> int:
		if not self._mouse_data:
			raise NoMouseEvent("No mouse event")
		return self._mouse_data.y

	@property
	def pos(self) -> tuple[int, int]:
		''' x, y '''
		if not self._mouse_data:
			raise NoMouseEvent("No mouse event")
		return (self._mouse_data.x, self._mouse_data.y)

	@property
	def event(self) -> int:
		if not self._mouse_data:
			raise NoMouseEvent("No mouse event")
		return self._mouse_data.event

	@property
	def flags(self) -> int:
		if not self._mouse_data:
			raise NoMouseEvent("No mouse event")
		return self._mouse_data.flags

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

@dataclass
class TLBRpos:
	TL: tuple[int, int]|None
	BR: tuple[int, int]|None	

def get_area(window: str, image: np.ndarray,
	TL_BR_list = [] # TLpos = [0, 0], BRpos = [0, 0]
) -> list[tuple[int, int]]:
	''' returns TL_BR(Top-Left, Bottom-Right) tuple list.
	Quit key set(['Q', 'q', 17]) raises QuitKeyException '''
	usage = "Drag mouse to select a rectangle area from top-left to bottom-right, (Right click to reset the area), then hit Space key to add a bottom-right position, Esc key to unset last position, finally hit Enter key to submit"	
	print(usage)
	copy_image = image.copy()
	cv2.imshow(window, image)

	mouse_param = MouseParam(window) #call back func.

	@dataclass
	class Status:
		first_click: bool
		is_rect: bool
		is_l_button_down: bool
		is_reset: bool	
	status = Status(False, False, False, False)
	first_click = False
	is_rect = False
	is_l_button_down = False
	is_reset = False

	tl_br = TLBRpos(None, None)
	# TLpos: tuple[int, int] | None = None #(0, 0) # top left
	# BRpos: tuple[int, int] | None = None #(0, 0) # bottom right
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
	def rm_last():
		status.first_click = False
		tl_br.TL = None
		tl_br.BR = None
		logger.info("Reset")
		status.is_reset = True
		if pos_list:
			pos_list.pop()
			redraw()
	try:
		last_data: MouseData = mouse_param.data
		while not is_reset:
			mdata: MouseData = mouse_param.data # refer this var in every loop
			key = cv2.waitKey(20)
			if key in [8, 127]: # BS or Del
				rm_last()
			elif key in [ord(" ")]: # Space
				pos_list.append(mdata.pos) 
				redraw()
			elif key in [17]: # Esc
				raise QuitKeyException()
			elif key in [ord("\n"), ord("\r")]: # Enter
				is_reset = True
				break
			# show if left click
			try:
				match mdata.event:
					case cv2.EVENT_LBUTTONUP:
						if is_l_button_down:
							# pos = data.pos # mouseData.getPos()
							tl_br.BR = mdata.x, mdata.y # [0] = pos[0]
							if tl_br.TL: #is_rect:
								image = copy_image.copy()
								cv2.rectangle(image, (tl_br.TL[0], tl_br.TL[1]), (tl_br.BR[0], tl_br.BR[1]), 0, 1)
								# second_click = True
								# is_rect = True
								cv2.imshow(window, image)
								logger.info("Redraw rectangle: %s, %s", tl_br.TL, tl_br.BR)
								is_l_button_down = False
								tl_br.TL = tl_br.BR = None # TLpos = BRpos = None 
					case cv2.EVENT_LBUTTONDOWN:
						is_l_button_down = True
						pos = mdata.pos # mouse_param.getPos()
						if not first_click:
							tl_br.TL = mdata.pos # x, mdata.y # [0] = pos[0]
							pos_list.append(mdata.pos)
							# TLpos[0] = pos[0] TLpos[1] = pos[1]
							first_click = True
							logger.info("tl_br.TL:%s", tl_br.TL)
					case cv2.EVENT_MOUSEMOVE:
						if not first_click: # show XY axis cursor
							image = copy_image.copy()
							pos = mouse_param.getPos()
							image[pos[1], :] = 127
							image[:, pos[0]] = 127
							cv2.imshow(window, image)
							continue
						elif is_l_button_down:
							# pos = mouse_param.getPos()
							tl_br.BR = mdata.pos # mdata.y # [0] = pos[0]
							if tl_br.TL: #is_rect:
								image = copy_image.copy()
								cv2.rectangle(image, (tl_br.TL[0], tl_br.TL[1]), (tl_br.BR[0], tl_br.BR[1]), 0, 1)
								# is_rect = True
								cv2.imshow(window, image)
								logger.info("Redraw rectangle:%s, %s", tl_br.TL, tl_br.BR)
						# image[TLpos[1]:BRpos[1], TLpos[0]:BRpos[0]] &= (255-7)

					# right click makes to reset
					case cv2.EVENT_RBUTTONUP:
						if not is_reset:
							first_click = False
							tl_br.TL = tl_br.BR = None # (0, 0)
							# for p in [TLpos, BRpos]: p[0] = p[1] = 0
							logger.info("Reset")
							is_reset = True
						continue
			except NoMouseEvent:
				pass
			else:
				last_data = mdata
	finally:
		cv2.destroyWindow(window)
	return [tl_br.TL or (0, 0), tl_br.BR or (0, 0)]

if __name__ == "__main__":
	from sys import argv
	#入力画像
	image = cv2.imread(argv[1], cv2.IMREAD_GRAYSCALE)
	if not image:
		raise ValueError("Image not found")
	
	#表示するWindow名
	window_name = "HEAD" + ":Left click to set TOp Left point, then keep pressing it(i.e. 'drag'), move mouse to set Bottom Right point, then unpress the left button, then type 's' to save, type 'q' to exit"
	
	#画像の表示
	#TLpos = [0, 0] BRpos = [0, 0]
	tl_br = TLBRpos(*get_area(window_name, image)) #, TLpos, BRpos)
	print(f"TLpos: {tl_br.TL}, BRpos: {tl_br.BR}")

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