import os
import sys


def add_path(path):
    if not path in sys.path:
        sys.path.insert(0, path)


# cwd
cwd = os.path.abspath(os.path.dirname(__file__))


root_dir = os.path.split(cwd)[0]
add_path(root_dir)

