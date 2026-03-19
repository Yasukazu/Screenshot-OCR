import remi.gui as gui
from remi import start, App

class MouseSelectionApp(App):
	def __init__(self, *args):
		super(MouseSelectionApp, self).__init__(*args)
		self.start_x = 0
		self.start_y = 0
		self.current_rect = None
		self.drawing = False

	def main(self):
		# Main container
		main_container = gui.VBox(width=600, height=450, style={'margin': '10px auto'})

		# Canvas setup
		self.canvas = gui.Svg(width=600, height=400, style={'border': '1px solid black', 'background-color': '#f0f0f0'})
		self.canvas.onmousedown.do(self.on_mouse_down)
		self.canvas.onmousemove.do(self.on_mouse_move)
		self.canvas.onmouseup.do(self.on_mouse_up)
		# Prevent default behavior (e.g., text selection)
		self.canvas.style['user-select'] = 'none'

		# Label to display coordinates
		self.label = gui.Label("Selection Area: None", style={'margin_top': '10px'})

		main_container.append(self.canvas)
		main_container.append(self.label)

		return main_container

	def on_mouse_down(self, emitter, x, y):
		"""Records the starting coordinates and begins the drawing process."""
		self.start_x = int(x)
		self.start_y = int(y)
		self.drawing = True
		# Start drawing the selection rectangle
		# A 1-pixel transparent rectangle is drawn initially to get a reference ID
		self.current_rect = self.canvas.draw_rect(self.start_x, self.start_y, 1, 1, fill_color='rgba(0,0,255,0.3)', stroke_color='blue', stroke_width=1)
		self.canvas.move_to_front(self.current_rect)
		self.label.set_text(f"Selection Area: Started at ({self.start_x}, {self.start_y})")

	def on_mouse_move(self, emitter, x, y):
		"""Updates the rectangle's size as the mouse moves."""
		if self.drawing:
			current_x = int(x)
			current_y = int(y)
			# Calculate width and height (can be negative if dragging left/up)
			width = current_x - self.start_x
			height = current_y - self.start_y

			# Update the existing rectangle using its ID
			# Remi automatically handles updating the position and size in the browser
			self.canvas.set_size_by_id(self.current_rect, width, height)
			self.canvas.set_position_by_id(self.current_rect, self.start_x, self.start_y)
			self.label.set_text(f"Selection Area: Dragging to ({current_x}, {current_y})")

	def on_mouse_up(self, emitter, x, y):
		"""Finalizes the rectangle and records the final coordinates."""
		if self.drawing:
			self.drawing = False

		end_x = float(x)
		end_y = float(y)

		# Ensure the rectangle coordinates are ordered correctly (top-left, bottom-right)
		x1 = min(self.start_x, end_x)
		y1 = min(self.start_y, end_y)
		width = abs(self.start_x - end_x)
		height = abs(self.start_y - end_y)

		# Draw the final rectangle. The previous dynamic one can be deleted if needed.
		# Here we just update the existing one to the final position.
		self.canvas.set_size_by_id(self.current_rect, width, height)
		self.canvas.set_position_by_id(self.current_rect, x1, y1)

		self.label.set_text(f"Selection Area: Finalized from ({x1}, {y1}) to ({x1 + width}, {y1 + height})")

		# Optional: Store the final rectangle ID in a list if you want multiple
		# selections, otherwise the next mousedown will overwrite current_rect.

# Starts the web server
if __name__ == "__main__":
	# Configuration for the web server
	configuration = {'address': '0.0.0.0', 'port': 8081, 'debug': True, 'enable_file_cache': False, 'update_interval': 0}
	start(MouseSelectionApp, **configuration)