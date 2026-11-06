#!/usr/bin/env bash
# Download DeepCrack. Run this OUTSIDE the repository (for example in a Colab working directory).
# DeepCrack is not redistributed here. See datasets/README.md for the licence terms.
set -e
git clone https://github.com/yhlleo/DeepCrack.git
unzip -q DeepCrack/dataset/DeepCrack.zip -d DeepCrack/dataset/
