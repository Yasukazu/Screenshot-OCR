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
	- Benefits:
	 - loading from/to YAML/JSON files
	 - Enum type field in dataclass

 3. `dataparsers`
	```Python
	```

 4. `Yada`(Yet another dataclass argument parser)
	```Python
	```
