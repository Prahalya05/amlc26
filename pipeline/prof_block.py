"""Profile blocking on a small validation sample: python prof_block.py N"""
import cProfile
import pstats
import sys

from blocking import block
from m4_block_eval import s1_rows_for_fold

rows, _ = s1_rows_for_fold("train", [0], int(sys.argv[1]))
cfg = {"cap": 2000, "k": 100, "types": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]}
cProfile.run("block('train', rows, cfg)", "prof.out")
pstats.Stats("prof.out").sort_stats("tottime").print_stats(14)
