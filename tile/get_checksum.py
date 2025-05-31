import hashlib
from pathlib import Path

def get_checksum(fullpath: Path, algo='sha256', blocksize=4096):
	bin_data = fullpath.read_bytes()
	
	h = hashlib.new(algo)
	
	hash_len = hashlib.new(algo).block_size * blocksize
	
	while bin_data:
	    part_data = bin_data[:hash_len]
	    bin_data = bin_data[hash_len:]
	    h.update(part_data)
	    
	return h.hexdigest()