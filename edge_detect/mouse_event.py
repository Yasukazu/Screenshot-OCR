import cv2
import numpy as np

class mouseParam:
	def __init__(self, input_img_name):
		#マウス入力用のパラメータ
		self.mouseEvent = {"x":None, "y":None, "event":None, "flags":None}
		#マウス入力の設定
		cv2.setMouseCallback(input_img_name, self.__CallBackFunc, None)
	
	#コールバック関数
	def __CallBackFunc(self, eventType, x, y, flags, userdata):
		
		self.mouseEvent["x"] = x
		self.mouseEvent["y"] = y
		self.mouseEvent["event"] = eventType    
		self.mouseEvent["flags"] = flags    

	#マウス入力用のパラメータを返すための関数
	def getData(self):
		return self.mouseEvent
	
	#マウスイベントを返す関数
	def getEvent(self):
		return self.mouseEvent["event"]                

	#マウスフラグを返す関数
	def getFlags(self):
		return self.mouseEvent["flags"]                

	#xの座標を返す関数
	def getX(self):
		return self.mouseEvent["x"]  

	#yの座標を返す関数
	def getY(self):
		return self.mouseEvent["y"]  

	#xとyの座標を返す関数
	def getPos(self) -> tuple[int, int]:
		return (self.mouseEvent["x"], self.mouseEvent["y"])
		
def main(window: str, image: np.ndarray,
	TLpos = [0, 0],
	BRpos = [0, 0]
):
	copy_image = image.copy()
	cv2.imshow(window, image)
	
	#コールバックの設定
	mouseData = mouseParam(window)

	first_click = second_click = False
	is_rect = False
	is_l_button_down = False
	while 1:
		key = cv2.waitKey(50)
		if key == ord("q"):
			break
		#左クリックがあったら表示
		match (event:=mouseData.getEvent()):
			case cv2.EVENT_LBUTTONUP:
				if is_l_button_down:
					pos = mouseData.getPos()
					BRpos[0] = pos[0]
					BRpos[1] = pos[1]
					if is_rect:
						image = copy_image.copy()
					cv2.rectangle(image, (TLpos[0], TLpos[1]), (BRpos[0], BRpos[1]), 0, 1)
					second_click = True
					is_rect = True
					cv2.imshow(window, image)
					print(f"Redraw rectangle:{TLpos=},{BRpos=}")
					is_l_button_down = False
			case cv2.EVENT_LBUTTONDOWN:
				is_l_button_down = True
				pos = mouseData.getPos()
				if not first_click:
					TLpos[0] = pos[0]
					TLpos[1] = pos[1]
					first_click = True
					print(f"{TLpos=}")

			case cv2.EVENT_MOUSEMOVE:
				if not first_click:
					continue

				if is_l_button_down:
					pos = mouseData.getPos()
					BRpos[0] = pos[0]
					BRpos[1] = pos[1]
					if is_rect:
						image = copy_image.copy()
					cv2.rectangle(image, (TLpos[0], TLpos[1]), (BRpos[0], BRpos[1]), 0, 1)
					is_rect = True
					cv2.imshow(window, image)
					print(f"Redraw rectangle:{TLpos=},{BRpos=}")
				# image[TLpos[1]:BRpos[1], TLpos[0]:BRpos[0]] &= (255-7)

			# right click makes to reset
			case cv2.EVENT_RBUTTONDOWN:
				first_click = second_click = False
				for p in [TLpos, BRpos]:
					p[0] = p[1] = 0
				print("Reset")

if __name__ == "__main__":
	from sys import argv
	#入力画像
	image = cv2.imread(argv[1], cv2.IMREAD_GRAYSCALE)
	
	#表示するWindow名
	window_name = "HEAD" + ":Left click to set TOp Left point, then keep pressing it(i.e. 'drag'), move mouse to set Bottom Right point, then unpress the left button, then type 's' to save, type 'q' to exit"
	
	#画像の表示
	TLpos = [0, 0]
	BRpos = [0, 0]
	main(window_name, image, TLpos, BRpos)
	print(f"TLpos: {TLpos}, BRpos: {BRpos}")
			
	cv2.destroyAllWindows()            
	print("Finished")

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