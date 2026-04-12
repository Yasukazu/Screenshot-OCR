## `perprexity.ai`'s suggenstion of "Python dataclass argument parser"

 1. `argparse-dataclass`
	- Python Package Index: https://pypi.org/project/argparse-dataclass/
	- Repository: https://github.com/mivade/argparse-dataclass
	- Usage:
	 - field metadata
		- Alias:`metadata={"args": ["-x", "--long-name"]}` 
		- Custom type converter:`metadata={"type": some_function}` 
		- Default value:`metadata={"default": "default_value"}`
		- Required: `metadata={"required": True}`
	- Restrictions: 
	 - Nullable field must be specified as `: Optional[type]` instead of `: type | None`
	```Python
	from argparse_dataclass import dataclass
	
	@dataclass
	class Options:
		name: str = "default_name"

	parser = ArgumentParser(Options)
	args, unknown = parser.parse_known_args()
	```

 2. `Simple-Parsing`
	- Repository: https://github.com/lebrice/SimpleParsing
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
	from enum import StrEnum
	from tap import Tap
	NAME = StrEnum('NAME', ['tm', 'mrcr'])
	class Config(Tap):
		name: NAME = NAME.tm
		find_env_file = True
		env_file: str = '.env' 
		"""Environment variable setting file name"""
		def configure(self):
			self.add_argument('--name', type=NAME, choices=[m.value for m in NAME]) # for better help message
	args = Config().parse_args()
	```
 7. `typed-argparser`: a fork of `typed_argparser`
	- https://github.com/Yasukazu/typed-argparser
	- Features:
	 - Dict type support
	 - Every function defined with execution decorator(leading with `@<ArgumetClass>.execute('option_name')`) is automatically executed when all the decorator's arguments are fulfilled after parsing arguments; i.e. options are used as flags to switch these functions.
	```Python
	from typing import List, Optional, Dict, Tuple, Union
	from typed_argparser import ArgumentClass, argfield
	class Example1(ArgumentClass):
		"""This example shows how to use some of the basic types in typed_argparser."""
		# Positional arguments do not generate short or long options
		opt1: Union[int, str] = argfield(help="opt1 is a mandatory argument which can be an integer or a string")
		opt2: List[str] = argfield(help="opt2 is a mandatory argument and can be used multiple times")
		# Optional arguments generate only long option by default if no short option is provided
		opt3: Optional[str] = argfield(help="this is an optional argument.")
		# Use Dict type to accept multiple key value pairs
		opt4: Optional[Dict[str, int]] = argfield(help="arg as 'key=value' pairs can be used multiple times.")
		# Use Tuple type to accept exactly n no. of arguments
		opt5: Optional[Tuple[str, ...]] = argfield("-o", "--option5", nargs=4, help="accepts 4 params")

	cli = Example1()
	import sys
	cli.parse(sys.argv[1:])
	print(cli)

	from typed_argparser.types import Args  # noqa: E402

	class Example2(ArgumentClass):
		"""This example shows how to use the `execute` decorator to execute functions based on the arguments provided."""

		# Positional arguments do not generate short or long options
		opt1: Union[int, str] = argfield(help="opt1 is a mandatory argument which can be an integer or a string")
		opt2: List[str] = argfield(help="opt2 is a mandatory argument and can be used multiple times")
		# Use Annotated from typing to provide arguments to types as shown below
		opt3: Annotated[Optional[Path], Args(mode="w")] = argfield(help="this is an output file argument.")
		# Use Dict type to accept multiple key value pairs
		opt4: Optional[Dict[str, int]] = argfield(help="accept key value pairs. can be used multiple times.")


	cli = Example2()

	cli.parse("--opt3 output.txt 20 abc")

	@cli.execute("opt1", "opt2")
	def execute_1(opt1: str, opt2: List[str]) -> None:
		print("This function is executed when both function arguments are provided.")
		print(f"opt1: {opt1}, opt2: {opt2}")

	from io import TextIOWrapper
	@cli.execute("opt3")
	def execute_2(opt3: TextIOWrapper) -> None:
		opt3.write("This is written to the output file.")


	@cli.execute("opt4")
	def execute_3(opt4: Dict[str, int]) -> None:
		print("This will not be executed as opt4 is not provided.")
	```