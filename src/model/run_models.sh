#!/bin/bash

for sr in politics movies nba gaming leagueoflegends nfl science technology CFB
do
  python train_bm.py --sr $sr >> "output/output_bm_$sr.txt" 2>> "output/errors_bm_$sr.txt"
  python train_fblp.py --sr $sr >> "output/output_fblp_$sr.txt" 2>> "output/errors_fblp_$sr.txt"
  python train_nc.py --sr $sr --n_epochs 600 >> "output/output_nc_$sr.txt" 2>> "output/errors_nc_$sr.txt"
  python train_ncplus.py --sr $sr --n_epochs 600 >> "output/output_ncplus_$sr.txt" 2>> "output/errors_ncplus_$sr.txt"
  python train_cm.py --sr $sr --lr 0.001 --n_epochs 20 --cuda "$1" >> "output/output_char_$sr.txt" 2>> "output/errors_char_$sr.txt"
  python train_dga.py --sr $sr --n_epochs 600 >> "output/output_dga_$sr.txt" 2>> "output/errors_dga_$sr.txt"
  python train_dgaplus.py --sr $sr --n_epochs 600 >> "output/output_dgaplus_$sr.txt" 2>> "output/errors_dgaplus_$sr.txt"
  python evaluation.py --sr $sr
done
python summary.py
