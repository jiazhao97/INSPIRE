# INSPIRE

*INSPIRE: interpretable, flexible and spatially-aware integration of multiple spatial transcriptomics datasets from diverse sources*

An effective method for integrating and interpreting multiple spatial transcriptomics datasets.

![INSPIRE\_pipeline](demo/overview.jpg)

We develop INSPIRE, a deep learning method for integrative analyses of multiple spatial transcriptomics datasets from diverse sources. With designs of graph neural networks and an adversarial learning mechanism, INSPIRE enables spatially informed and adaptable integration of data from varying sources. By incorporating non-negative matrix factorization, INSPIRE uncovers interpretable spatial factors with corresponding gene programs, revealing tissue architectures, cell type distributions, and biological processes.

INSPIRE takes gene expression count matrices and spatial coordinates from multiple ST slices as inputs, and provides three sets of outputs: latent representations of cells or spatial spots, non-negative spatial factors on cells or spatial spots, and non-negative gene loadings.


## Installation
* Portal can be downloaded from GitHub:
```bash
git clone https://github.com/jiazhao97/INSPIRE.git
cd INSPIRE
conda env update --f environment.yml
conda activate INSPIRE
```
