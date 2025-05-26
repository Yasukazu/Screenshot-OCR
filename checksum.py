# This script calculates the checksum of a file using specified hash algorithms.
import hashlib
from pathlib import Path

def calculate_checksum(fullpath: Path, hash_algorithm='sha256'):
    """Calculates the checksum of a file using the specified hash algorithm.

    Args:
        filename: The path to the file.
        hash_algorithm: The hash algorithm to use (e.g., 'md5', 'sha256').

    Returns:
        The checksum of the file as a hexadecimal string.
    """
    hasher = hashlib.new(hash_algorithm)
    with fullpath.open('rb') as file:
        while True:
            chunk = file.read(4096)  # Read in chunks to handle large files
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()
if __name__ == "__main__":
    # Example usage:
    filename = Path('my_file.txt')
    checksum_md5 = calculate_checksum(filename, 'md5')
    checksum_sha256 = calculate_checksum(filename, 'sha256')

    print(f"MD5 checksum of {filename}: {checksum_md5}")
    print(f"SHA256 checksum of {filename}: {checksum_sha256}")