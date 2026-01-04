import cv2

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
		
def main(window_name, image):
	copy_image = image.copy()
	cv2.imshow(window_name, image)
	
	#コールバックの設定
	mouseData = mouseParam(window_name)
	TLpos = [0, 0]	
	BRpos = [0, 0]	
	first_click = second_click = False
	while 1:
		cv2.waitKey(200)
		#左クリックがあったら表示
		match mouseData.getEvent():
			case cv2.EVENT_LBUTTONDOWN:
				pos = mouseData.getPos()
				if not first_click:
					TLpos[0] = pos[0]
					TLpos[1] = pos[1]
					first_click = True
				else:
					if not second_click:
						BRpos[0] = pos[0]
						BRpos[1] = pos[1]
						second_click = True
			case cv2.EVENT_MOUSEMOVE:
				if not second_click:
					pos = mouseData.getPos()
					BRpos[0] = pos[0]
					BRpos[1] = pos[1]
					if first_click:
						cv2.rectangle(copy_image, (TLpos[0], TLpos[1]), (BRpos[0], BRpos[1]), 0, 1)
						cv2.imshow(window_name, copy_image)
				# image[TLpos[1]:BRpos[1], TLpos[0]:BRpos[0]] &= (255-7)

			#右クリックがあったら終了
			case cv2.EVENT_RBUTTONDOWN:
				break;

if __name__ == "__main__":
	from sys import argv
	#入力画像
	image = cv2.imread(argv[1], cv2.IMREAD_GRAYSCALE)
	
	#表示するWindow名
	window_name = "input window"
	
	#画像の表示
	main(window_name, image)

			
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