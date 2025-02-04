import os
from util import mini

dirname = os.path.dirname(__file__)

RESULTS_PATH = os.path.join(dirname, f"../data/results{'-mini' if mini.is_mini() else ''}.generated.csv")
