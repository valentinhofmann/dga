# Derivational Graph Auto-encoder (DGA)

This repository contains the code and data for the ACL paper [A Graph Auto-encoder Model of 
Derivational Morphology](https://www.aclweb.org/anthology/2020.acl-main.106.pdf).
The paper introduces the **Derivational Graph Auto-encoder (DGA)**, a model that learns 
embeddings capturing information about the compatibility of affixes and stems in derivation.

## Dependencies

The code requires `Python>=3.5`, `torch`, `torch_geometric`, `pickle`,  `numpy`, `pandas`, `scipy`, and `sklearn`.


## Data

You can find the Derivational Graphs (DGs) for the nine subreddits in `data/graphs`.

The derivational embeddings trained on the DGs are located in `src/model/embeddings`.

## Experiments

To replicate the experiments from the paper, run `run_models.sh` in `src/model`.

## Citation

If you use the code or data in this repository, please cite the following paper:

```
@inproceedings{hofmann2020dga,
    title = {A Graph Auto-encoder Model of Derivational Morphology},
    author = {Hofmann, Valentin and Sch{\"u}tze, Hinrich and Pierrehumbert, Janet},
    booktitle = {Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
    year = {2020}
}

```
