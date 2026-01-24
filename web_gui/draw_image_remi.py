import remi.gui as gui
from remi import start, App
import os
from pathlib import Path
from PIL import Image
from io import BytesIO
from base64 import b64encode

class MyApp(App):

	'''def __init__(self, *args, **kwargs):
		self.userdata = kwargs["userdata"]
		super(MyApp, self).__init__(*args, **kwargs)'''

	def main(self, param1):#, image_path: Path):
		# Create a main container VBox
		main_container = gui.VBox(width=500, height=500, style={'margin':'0px auto'})
		main_container.style['text-align'] = 'center'

		# Create a Canvas widget
		# self.canvas = gui.Canvas(width=400, height=400, style={'border': '1px solid black'})
		pil_image = Image.open(param1) # Load image file by PIL
		img_byte_array = BytesIO()
		pil_image.save(img_byte_array, format='PNG')
		img_str = b64encode(img_byte_array.getvalue()).decode('utf-8')
		img_url = "data:image/png;base64," + img_str
		# img_bytes = img_byte_array.getvalue()
		
		# Create an Image widget from the local PNG file
		# The file path needs to be accessible by the remi server
		# image_path = os.path.join(os.getcwd(), 'image.png') 
		self.image = gui.Image(img_url, width=pil_image.width, height=pil_image.height)
		
		# Connect the image onload event to a function that draws it on the canvas
		# self.image.onload.connect(self.on_image_loaded)

		main_container.append(self.image)
		
		# We need to append the Image widget to the UI somewhere (e.g., make it invisible)
		# for remi to serve it correctly, or use the resource handling mechanism.
		# A simpler way in this context is to register it as a resource.
		# Alternatively, you can use the base64 method for background images (see snippets).

		return main_container

	def on_image_loaded(self, emitter):
		# This function is called when the image data is ready in the browser.
		# You can now draw the image onto the canvas using its ID.
		# The draw_image method uses the client-side canvas API via JavaScript.

		# Draw the image on the Canvas at coordinates (x=0, y=0) using its natural width and height
		self.canvas.draw_image(self.image, 0, 0, self.image.width, self.image.height)
		self.canvas.update() # Refresh the canvas display

# To run this example, ensure you have an 'image.png' file in your script's directory.
if __name__ == "__main__":
	from sys import argv
	# starts the webserver
	start(MyApp, debug=True, address='0.0.0.0', port=8000, enable_file_cache=False, userdata=(argv[1],))
