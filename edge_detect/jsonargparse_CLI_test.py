from jsonargparse import CLI
from typing import List

def main(*args: List[str]):
	for n, arg in enumerate(args):
		print(f"Argument {n}: {arg}, ")

def dummy():
	"""Dummy function for testing."""
	pass

if __name__ == "__main__":
    CLI([main, dummy])#, as_positional=False)