from pathlib import Path
import os

if os.name == "posix":
    DATA_DIR = Path(r"/home/ftpuser/ftp/Data/THzHallEffect/")
else:
    DATA_DIR = Path(r"")
    exit("Path not set")


