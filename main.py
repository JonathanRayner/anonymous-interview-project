#!/usr/bin/env python3
from pathlib import Path
from data import TripletFaceDataset

# Implement functions here
...


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path_train_set', type=Path)
    parser.add_argument('path_test_set', type=Path)
    args = parser.parse_args()

    # Implement execution here
    ...
