import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys

debug = False
if len(sys.argv) == 2:
	print("[INFO] Debugging on")
	debug = True

def list_files(directory):
    files = os.listdir(directory)
    file_list = []

    for f in files:
        file = os.path.join(directory, f)
        
        if not os.path.isfile(file):
        	continue

        file_list.append(file)

    return file_list

def read_csv(file):
	if debug:
		print("[INFO] Reading:", file)
	try:
		csv = pd.read_csv(file, sep=";")
		return csv
	except Exception as e:
		if debug:
			print("[ERROR] Error while reading:", file, "-", e)
		return None

def csvs_to_df(file_list):
	df = pd.DataFrame()

	for file in file_list:
		csv = read_csv(file)
		if csv is not None:
			df = pd.concat([df, csv], ignore_index=True)

	return df


def main():
	files = list_files("data")
	df = csvs_to_df(files)

	# print(df.head())
	# print(df.isnull().sum())
	print(df[df.Conforme == "Não"])

	return 0

if __name__ == "__main__":
	main()