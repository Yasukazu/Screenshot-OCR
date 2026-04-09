from fire import Fire
from typing import Sequence

def main(*args: Sequence[str], opt: str = ""):
	for n, arg in enumerate(args):
		print(f"Argument {n}: {arg}, Option: {opt}")

def dummy():
	"""Dummy function for testing."""
	pass

if __name__ == "__main__":
	Fire({"main": main, "dummy": dummy})