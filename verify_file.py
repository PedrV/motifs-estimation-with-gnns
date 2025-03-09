import hashlib

def verify(file_path, verbose=False):
    # Read the file and original checksum
    print(file_path)
    with open(file_path, "rb") as f:
        file_data = f.read()

    with open(str(file_path).split(".")[0] + ".sha256", "r") as f:
        original_checksum = f.read().strip()

    # Create checksum for comparison
    checksum = hashlib.sha256(file_data).hexdigest()

    # Verify the checksum
    if checksum == original_checksum:
        if verbose: print("The file is valid.")
        return True
    else:
        if verbose: print("The file has been altered.")
        return False
