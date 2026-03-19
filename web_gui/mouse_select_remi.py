import remi.gui as gui
from remi import start, App
from typing import Optional

class MyApp(App):
	def __init__(self, *args):
		super().__init__(*args)
		self.rect: Optional[gui.SvgRectangle] = None
		self.start_x: float = 0
		self.start_y: float = 0

	def main(self):
		container = gui.VBox(width=500, height=500)
		self.svg = gui.Svg(width=400, height=400)
		self.svg.style['background-color'] = 'white'
		self.svg.style['border'] = '1px solid black'
		self.svg.onmousedown.do(self.on_mouse_down)
		self.svg.onmousemove.do(self.on_mouse_move)
		self.svg.onmouseup.do(self.on_mouse_up)
		
		self.rect = None
		self.start_x = 0
		self.start_y = 0
		
		container.append(self.svg)
		return container

	def on_mouse_down(self, widget: gui.Widget, x: float, y: float) -> None:
		self.start_x = float(x)
		self.start_y = float(y)
		self.rect = gui.SvgRectangle(self.start_x, self.start_y, 0, 0)
		self.rect.set_fill('rgba(0,0,255,0.3)')
		self.rect.set_stroke(1, 'blue')
		self.svg.append(self.rect)

	def on_mouse_move(self, widget: gui.Widget, x: float, y: float) -> None:
		if self.rect is not None:
			width = float(x) - self.start_x
			height = float(y) - self.start_y
			self.rect.set_size(abs(width), abs(height))
			if width < 0:
				self.rect.attributes['x'] = str(float(x))
			else:
				self.rect.attributes['x'] = str(self.start_x)
			if height < 0:
				self.rect.attributes['y'] = str(float(y))
			else:
				self.rect.attributes['y'] = str(self.start_y)

	def on_mouse_up(self, widget: gui.Widget, x: float, y: float) -> None:
		if self.rect is not None:
			x_pos = min(float(x), self.start_x)
			y_pos = min(float(y), self.start_y)
			w = abs(float(x) - self.start_x)
			h = abs(float(y) - self.start_y)
			print(f"Selected: X:{x_pos}, Y:{y_pos}, W:{w}, H:{h}")
			self.rect = None # Reset for next selection

if __name__ == "__main__":
	start(MyApp)
