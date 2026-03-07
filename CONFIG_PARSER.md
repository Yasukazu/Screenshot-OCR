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
 5. `tap`
	- https://github.com/omarish/tap
	- Limitations:
	 - Dict type is not supported(Extendable by `def configure(self):self.add_argument(type=..)` in Tap successor class)
	 - Configuration file is a text file(with every line as a command option) or a json file
	```Python
	from tap import Tap
	class ArgParser(Tap):
		find_env_file = True
		env_file: str = '.env' 
		"""Environment variable setting file name"""
	args = ArgParser().parse_args()
	```
