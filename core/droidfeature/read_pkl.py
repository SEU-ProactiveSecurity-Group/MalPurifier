import pickle
import builtins
from pprint import pprint
import sys


def print(*args, **kwargs):
    # Open the file in append mode
    with open('feature_example.txt', 'a') as f:
        # Call the original print function and set the output to the file
        builtins.print(*args, **kwargs, file=f)

    # Call the original print function to print to the console
    builtins.print(*args, **kwargs)


# apk_feature = pickle.load( open( "./feature_example.pkl", "rb" ) )
print(sys.path)
