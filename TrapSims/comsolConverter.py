import numpy as np
import pandas as pd
import os



# Try to prompt user for filename, append this to the current directory, and open file to read.
try:
    f = open(os.path.dirname(os.path.realpath(__file__)) + "/" + input("Import Filename: "), "r")

except FileNotFoundError as err:
    print("File Not Found! Run this script in the same directory as the target file, or check the filename provided.")

else:
    # Read all lines, discard initial lines without data, and form into list of lists of 
    # coordinates and associated potentials which is converted to a dataframe.
    pd.DataFrame([line.split() for line in f.readlines() if line[0] 
        != "%"]).to_csv(os.path.dirname(os.path.realpath(__file__)) + "/" + input("Export Filename: "))

    f.close()