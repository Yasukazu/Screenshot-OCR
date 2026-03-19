import remi.gui as gui
from remi import start, App
import os
from pathlib import Path
from PIL import Image
from io import BytesIO
from base64 import b64encode
from nicegui import ui

def direct_show(param):
	pil_image = Image.open(param) # Load image file by PIL
	ui.image(pil_image).classes('w-64')
	ui.run()

# To run this example, ensure you have an 'image.png' file in your script's directory.
if __name__ in {"__main__", "__mp_main__"}:
	from sys import argv
	direct_show(argv[1])
	# start(MyApp, debug=True, address='0.0.0.0', port=8000, enable_file_cache=False, userdata=(argv[1],)) # starts the webserver
