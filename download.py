"""
NOT READY YET
"""

import subprocess
import hashlib
import pathlib

def main():
    """
    
    """
    data_url = "xxxx"
    checksum_filename = "SHA256SUMS.txt"
    records_filename = "RECORDS"
    log_dir = pathlib.Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    root_dir = download_files(data_url)
    checksums = load_checksums(root_dir / checksum_filename)
    verify_files(root_dir, checksums, log_dir)

def load_checksums(file_path: str) -> str:
    """
    """
    pass

def calculate_checksum(file_path: str) -> str:
    """
    calculate SHA-256 checksum of a file
    """
    with open(file_path, "rb") as data_file:
        # if file is large, it can be computed incrementally
        data_bytes = data_file.read()
    sha256_hash = hashlib.sha256(data_bytes)    
    return sha256_hash.hexdigest()

def download_files(shell_script: str) -> str:
    """
    Simply calls the download shell script.
    TODO: Error handling, what if some downloads fail or timeout,
    When we scale up, it makes sense to download RECORDS file first,
    check ROBOTS.txt, download CHECKSUM file, then download the files
    one after another, verifying as they flow in with asyncio coroutines.

    TODO: Also add logging to the log file
    """
    subprocess.call(f"sh {shell_script}")


def verify_files(files, checksums, log_dir):
    """
    
    """
    verified_corrupt = []
    unknown = []
    for filename in files:
        try:
            sha256_hash = calculate_checksum(filename)
        except Exception as e:
            print(f"Couldn't calculate checksum for {files}")
            print(f"{type(e)} :: {str(e)}")
            unknown.append(filename)

        if sha256_hash != checksums[filename]:
            verified_corrupt.append(filename)
    
    log_file = pathlib.Path(log_dir) / "download_todo:add_timestamp.log"
    _log_verification(files, verified_corrupt, unknown, log_file)

def _log_verification(files, verified_corrupt, unknown, log_file):
    """
    TODO: use python logging library to write to log_file
    """
    print("--------------------------------------------")
    print(f"{len(verified_corrupt)} files corrupt.")
    print(f"Failed to compute sha256 hash for {len(unknown)} files.")
    print("--------------------------------------------")

