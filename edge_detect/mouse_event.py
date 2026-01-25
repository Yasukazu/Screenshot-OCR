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
	def wait(self):
		while not self._mouse_data:
			cv2.waitKey(20)
	
	def __CallBackFunc(self, eventType, x, y, flags, userdata):
		
		self._mouse_data = MouseData(x, y, eventType, flags)
		self.mouseEvent["x"] = x
		self.mouseEvent["y"] = y
		self.mouseEvent["event"] = eventType    
		self.mouseEvent["flags"] = flags    

	@property
	def data_or_none(self) -> MouseData|None:
		return self._mouse_data
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
class RectPos:
	RESET = (-1, -1)
	LT: tuple[int, int]
	RB: tuple[int, int]
	def reset(self):
		self.LT = self.RESET
		self.RB = self.RESET

def get_area(window: str, image: np.ndarray,
	TL_BR_list = [] # TLpos = [0, 0], BRpos = [0, 0]
) -> list[RectPos]:#tuple[int, int]]:
	''' returns TL_BR(Top-Left, Bottom-Right) tuple list.
	Quit key set(['Q', 'q', 17]) raises QuitKeyException '''
	usage = "Drag mouse to select a rectangle area from top-left to bottom-right, (Right click to reset the area), then hit Space key to add a bottom-right position, Esc key to unset last position, finally hit Enter key to submit"	
	print(usage)
	org_image = image.copy()
	cv2.imshow(window, image)

	mouse_param = MouseParam(window) #call back func.

	@dataclass
	class Status:
		first_click: bool
		is_rect: bool
		is_l_button_down: bool
		is_reset: bool	
	status = Status(False, False, False, False)
	# first_click = False
	# is_rect = False
	l_button_clicked = False
	is_reset = False

	rect_pos = RectPos((-1, -1), (-1, -1))
	# TLpos: tuple[int, int] | None = None #(0, 0) # top left
	# BRpos: tuple[int, int] | None = None #(0, 0) # bottom right
	from collections import UserList
	class RectPosList(UserList):
		def append(self, item):
			assert isinstance(item, RectPos)
			assert item.LT is not (-1, -1)
			assert item.RB is not (-1, -1)
			self.data.append(item)
	rect_pos_list: RectPosList = RectPosList() # additional bottom right
	redraw_image = org_image
	def redraw(point: tuple[int, int]|None=None):
		nonlocal redraw_image
		# pos_step, odd = divmod(len(rect_pos_list), 2)
		if not rect_pos_list: # pos_step and not odd: #len(pos_list) < 2:
			return
		redraw_image = org_image.copy()
		for p in range(len(rect_pos_list)):#pos_step):#len(pos_list)-1):
			tl = rect_pos_list[p].LT #2 * p]
			br = rect_pos_list[p].RB #2 * p + 1]
			assert tl is not None
			assert br is not None
			cv2.rectangle(redraw_image, tl, br, 0, 2)
			logger.info("Redraw rectangle:%s, %s", tl, br)
		if point and rect_pos.LT is not RectPos.RESET:
			tl = rect_pos.LT
			copy_redraw_image = redraw_image.copy()
			cv2.rectangle(copy_redraw_image, tl, point, 0, 1)
			cv2.imshow(window, copy_redraw_image)
		else:
			cv2.imshow(window, redraw_image)

	def rm_last():
		status.first_click = False
		rect_pos.LT = (-1, -1)
		rect_pos.RB = (-1, -1)
		logger.info("Reset")
		status.is_reset = True
		if rect_pos_list:
			rect_pos_list.pop()
			redraw()
	try:
		mouse_param.wait()
		# last_data: MouseData = mouse_param.data
		while not is_reset:
			mdata: MouseData = mouse_param.data # refer this var in every loop
			key = cv2.waitKey(20)
			if key in [8, 127]: # BS or Del
				rm_last()
			#elif key in [ord(" ")]: # Space rect_pos_list.append(mdata.pos) redraw()'''
			elif key in [17]: # Esc
				raise QuitKeyException()
			elif key in [ord("\n"), ord("\r")]: # Enter
				is_reset = True
				break
		# show if left click
			match mdata.event:
				case cv2.EVENT_LBUTTONUP:
					if l_button_clicked:
						assert rect_pos.LT is not (-1, -1) 
						# pos = data.pos # mouseData.getPos()
						assert mdata.pos is not None
						if len(rect_pos_list) == 0:
							rect_pos.RB = mdata.pos
						else:
							rect_pos.RB = (mdata.x, rect_pos_list[0].RB[1])
						assert rect_pos.RB is not (-1, -1)
						assert rect_pos.LT is not (-1, -1)
						rect_pos_list.append(rect_pos)
						logger.info("Rect pos is appended: %s", rect_pos)
						redraw()
						# tl_br.BR = mdata.x, mdata.y # [0] = pos[0]
						'''if tl_br.TL: #is_rect:
							image = copy_image.copy()
							cv2.rectangle(image, (tl_br.TL[0], tl_br.TL[1]), (tl_br.BR[0], tl_br.BR[1]), 0, 1)
							# second_click = True
							# is_rect = True
							cv2.imshow(window, image)
							logger.info("Redraw rectangle: %s, %s", tl_br.TL, tl_br.BR)'''
						l_button_clicked = False
						rect_pos.LT = rect_pos.RB = (-1, -1) # TLpos = BRpos = None 
				case cv2.EVENT_LBUTTONDOWN:
					l_button_clicked = True
					xpos, ypos = mdata.pos # mouse_param.getPos()
					if rect_pos.LT == RectPos.RESET: # first_click:
						# Instead of:
						# rect_pos_list = [i for i in rect_pos_list if i is not None and i.RB is not None and i.LT is not None]

						# Create a new RectPosList and extend it with the filtered items:
						filtered = RectPosList()
						filtered.extend([i for i in rect_pos_list if i is not None and i.RB is not (-1, -1) and i.LT is not (-1, -1)])
						rect_pos_list = filtered
						if len(rect_pos_list) == 0:
							rect_pos.LT = xpos, ypos # x, mdata.y # [0] = pos[0]
						else:
							rect_pos.LT = xpos, rect_pos_list[0].LT[1] # y is aligned with the first click pos
						# rect_pos_list.append(rect_pos)
						# TLpos[0] = pos[0] TLpos[1] = pos[1]
						# first_click = True
						logger.info("tl_br.TL:%s", rect_pos.LT)
				case cv2.EVENT_MOUSEMOVE:
					if rect_pos.LT == RectPos.RESET and not len(rect_pos_list): #first_click: # show XY axis cursor
						image = org_image.copy()
						xpos, ypos = mdata.pos # mouse_param.getPos()
						#if len(rect_pos_list) > 0: ypos = rect_pos_list[0].LT[1]
						image[ypos, :] = 127
						image[:, xpos] = 127
						cv2.imshow(window, image)
						continue
					else: #if is_l_button_down:
						# pos = mouse_param.getPos()
						# if len(pos_list) == 1:
								# mdata.y # [0] = pos[0]
						if rect_pos.LT != RectPos.RESET: #is_rect:
							rect_pos.RB = mdata.pos
							redraw(rect_pos.RB)
						'''image = copy_image.copy()
							cv2.rectangle(image, (tl_br.TL[0], tl_br.TL[1]), (tl_br.BR[0], tl_br.BR[1]), 0, 1)
							# is_rect = True
							cv2.imshow(window, image)
							logger.info("Redraw rectangle:%s, %s", tl_br.TL, tl_br.BR)'''
						# image[TLpos[1]:BRpos[1], TLpos[0]:BRpos[0]] &= (255-7)
				# right click makes to reset
				case cv2.EVENT_RBUTTONUP:
					if not is_reset:
						# first_click = False
						rect_pos.LT = rect_pos.RB = (-1, -1) # (0, 0)
						# for p in [TLpos, BRpos]: p[0] = p[1] = 0
						logger.info("Reset")
						is_reset = True
					continue
	except NoMouseEvent:
		pass

	finally:
		cv2.destroyWindow(window)
	return rect_pos_list # [rect_pos.TL or (0, 0), rect_pos.BR or (0, 0)]

if __name__ == "__main__":
	from sys import argv
	#入力画像
	image = cv2.imread(argv[1], cv2.IMREAD_GRAYSCALE)
	if image is None:
		raise ValueError("Image not found")
	
	#表示するWindow名
	window_name = "HEAD" + ":Left click to set TOp Left point, then keep pressing it(i.e. 'drag'), move mouse to set Bottom Right point, then unpress the left button, then type 's' to save, type 'q' to exit"
	
	#画像の表示
	#TLpos = [0, 0] BRpos = [0, 0]
	rect_list = get_area(window_name, image) #, TLpos, BRpos)
	from pprint import pprint
	pprint(rect_list)#f"TLpos: {tl_br.TL}, BRpos: {tl_br.BR}")

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