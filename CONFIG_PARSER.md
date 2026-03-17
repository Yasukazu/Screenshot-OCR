## `perprexity.ai`'s suggenstion of "Python dataclass argument parser"

 1. `argparse-dataclass`
	- https://pypi.org/project/argparse-dataclass/
	```Python
	from argparse_dataclass import dataclass
	```
 2. `Simple-Parsing`
	- https://github.com/lebrice/Simple-Parsing
	```Python
	"""Example adapted from https://github.com/eladrich/pyrallis#my-first-pyrallis-example-"""
	import simple_parsing
	cfg = simple_parsing.parse(
		config_class=TrainConfig, args=args, config_path="config-file.yaml"
	)
	```
	- Features:
	 - loading from / saving to: YAML / JSON file
	 - Enum type field in dataclass

 3. `dataparsers`
	```Python
	```

 4. `Yada`(Yet another dataclass argument parser)
	```Python
	```
 5. `tap`: typed_argparse
	- https://typed-argparse.github.io/typed-argparse/high_level_api/#getting-started
	- Features:
	 - Enum type field: Dynamic StrEnum and Enum creation (useful for from environment variables)
	 - Parser class with bind method for business logic: `tap.Parser(Config).bind(runner).run()`
	 - (? how to use is not enough) Shell auto-completion based on `argcomplete`
	```Python
	from typing import List, Optional
	import typed_argparse as tap
	# 1. Argument definition
	class Config(tap.TypedArgs):
		my_arg: str = tap.arg(help="some help")
		number_a: int = tap.arg(default=42, help="some help")
		number_b: Optional[int] = tap.arg(help="some help")
		verbose: bool = tap.arg(help="some help")
		names: List[str] = tap.arg(help="some help")

	# 2. Business logic
	def runner(args: Config):
		print(f"Running my app with args:\n{args}")

	# 3. Bind argument definition + business logic & run
	def main() -> None:
		tap.Parser(Config).bind(runner).run()

	if __name__ == "__main__":
		main()
	```

 6. `Tap`(typed_argument_parser): tap,tap-* branches
	- https://github.com/swansonk14/typed-argument-parser
	- Limitations:
	 - Dict type is not supported(Extendable by `def configure(self):self.add_argument(type=..)` in Tap successor class)
	 - Class type field is supported only for single-string argument constructor class
	 - Configuration file is a text file(with every line as a command option) or a json file
	```Python
	from tap import Tap
	class Config(Tap):
		find_env_file = True
		env_file: str = '.env' 
		"""Environment variable setting file name"""
	args = Config().parse_args()
	```
