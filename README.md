# Two-Stage-Chinese-AMR-Parsing
Source code for paper "A Two-Stage Graph-Based Method for Chinese AMR Parsing with Explicit Word Alignment" @ CAMRP-2022, CCL'2022.

Our system won the second place at [CAMRP-2022](https://github.com/GoThereGit/Chinese-AMR#%E8%AF%84%E6%B5%8B%E6%8E%92%E5%90%8D) evaluation held with CCL'2022 conference.

## Preprocess

First, we need to transform the original Chinese AMR Annotations into 4 different sub-tasks in the Two-Stage method. 

The tasks include
1. Surface Tagging 
2. Normalization Tagging
3. Non-Aligned Concept Tagging
4. Relation Classification

`./scripts/preprocess.py` will automaticly generate preprocessed data for the four tasks given correct path for CAMR2.0 dataset and the tuple version of the dataset.

We have also provided a processed version of data under `./preprocessed`, that you could directly use.

~ To Be Continued
