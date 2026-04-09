import os
from pydantic import BaseSettings

base_dir = os.path.abspath(os.path.dirname(__file__))
		env_prefix = ""
		env_file = "image_filter.env"
		env_file_encoding = "utf-8"
		use_enum_values = True
