import tempfile

def save_temp_file(upload_file, suffix=".jpg"):
    """Save uploaded file to temp storage and return path."""
    tmp = tempfile.mktemp(suffix=suffix)
    with open(tmp, "wb") as f:
        f.write(upload_file.file.read())
    return tmp
