import os

def is_mini():
    return int(os.environ.get('FULLRUN', '0')) == 0
