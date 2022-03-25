"""
Code to count effect of synthetic links
"""

import h5py
import argparse
from tqdm import tqdm


def count_different(seq):
    prev = None
    tot = 0
    for el in seq:
        if (prev is None or el != prev) and el != '':
            tot += 1
        prev = el
    return tot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    args = parser.parse_args()
    handle = h5py.File(args.path, "r")
    tokens = 0
    labels = 0
    handle_0 = handle['0']
    handle_1 = handle['1']
    ndocs = len(handle_0)
    for i in tqdm(range(ndocs)):
        tokens += len(handle_0[i].split("\n"))
        labels += count_different(handle_1[i].split("\n"))

    print(f"tokens = {tokens}")
    print(f"labels = {labels}")
    print(f"documents = {ndocs}")

if __name__ == "__main__":
    main()
