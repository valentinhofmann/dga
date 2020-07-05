#!/bin/bash

for sr in politics movies nba gaming leagueoflegends nfl science technology CFB
do
  python evaluation.py --sr $sr
done
python summary.py
