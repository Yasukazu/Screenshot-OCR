import csv
import pandas as pd

SEVEN_SEG_STEM = '7-seg'
SEVEN_SEG_MAX = 15
CSV_EXT = '.csv'

def load_7_seg_csv(fname: str=SEVEN_SEG_STEM + CSV_EXT):
	with open(fname, encoding='ASCII') as f:
		reader = csv.reader(f)
		return [r for r in reader]

SEVEN_SEG_NUM_STEM = '7-seg-num'

def load_7_seg_num_csv_as_df()-> pd.DataFrame:
	return pd.read_csv(SEVEN_SEG_NUM_STEM + CSV_EXT, index_col=0).fillna(value=0).astype(int)