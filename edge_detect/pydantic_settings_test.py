from os.path import abspath, dirname
from pydantic_settings import BaseSettings, SettingsConfigDict, CliApp, CliPositionalArg
from pydantic import Field

class Settings(BaseSettings):
	env_prefix:str = "image_filter_"
	env_file:str = "image-filter.env"
	env_file_encoding:str = "utf-8"
	use_enum_values:bool = True
	image_dir_base: str = "~/DATA/"
	image_ext: str = ".bin"
	image_ext_set: set[str] = {".jpg", ".png"}
	app_name_to_stem_end:dict[str, str] = {}#'taimee': '_jp.co.taimee', 'mercari': '_jp.mercari.work.android'}
	path: CliPositionalArg[str] = Field(default="", description="Path to process")
	model_config = SettingsConfigDict(env_prefix=env_prefix, env_file=env_file, env_file_encoding=env_file_encoding, use_enum_values=use_enum_values)

	def cli_cmd(self) -> None:
		# return self.image_ext
		pass

from os.path import join as os_path_join
from typing import Any
from dotenv import load_dotenv
CONFIG_FILE_NAME = "image-filter.ini"

def main(settings: Settings = Settings(),
	base_dir = abspath(dirname(__file__))):
	config_fullpath = os_path_join(base_dir, CONFIG_FILE_NAME)
	# base_config: dict[str, Any] | None = None
	print(settings)

if __name__ == "__main__":
	from sys import argv
	s = CliApp.run(Settings, cli_args=argv[1:])
	# main()
	print(f"Parsed args: {s.args}")
