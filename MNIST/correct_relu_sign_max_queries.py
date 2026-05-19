#! /usr/bin/env python3

import os
import sys

# Remove the original output neurons from the list of outputs.
# Only keep newly created output neurons.
STR_OLD = """0,1784
1,1785
2,1786
3,1787
4,1788
5,1789
6,1790
7,1791
8,1792
9,1793
10,1805
11,1806
12,1807
13,1808
14,1809
15,1810
16,1811
17,1812
18,1813
19,1814"""

STR_NEW = """0,1805
1,1806
2,1807
3,1808
4,1809
5,1810
6,1811
7,1812
8,1813
9,1814"""

save_dir_relu_sign_max = sys.argv[1]

assert os.path.isdir(save_dir_relu_sign_max)
for obj in os.listdir(save_dir_relu_sign_max):
    file = os.path.join(save_dir_relu_sign_max, obj)
    if os.path.isfile(file):
        contents = ""
        with open(file, "r", encoding="utf-8") as fp:
            contents = fp.read()
            contents = contents.replace(STR_OLD, STR_NEW)
        with open(file, "w", encoding="utf-8") as fp:
            fp.write(contents)
