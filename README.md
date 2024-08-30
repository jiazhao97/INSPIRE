# INSPIRE

*INSPIRE: interpretable, flexible and spatially-aware integration of multiple spatial transcriptomics datasets from diverse sources*

An effective method for integrating and interpreting multiple spatial transcriptomics datasets.

![INSPIRE\_pipeline](demo/overview.jpg)

We develop INSPIRE, a deep learning method for integrative and interpretable analyses of multiple spatial transcriptomics (ST) datasets from diverse sources. INSPIRE takes gene expression count matrices and spatial coordinates from multiple ST slices as input. It provides three sets of outputs: latent representations of cells or spatial spots, non-negative spatial factors on cells or spatial spots, and non-negative gene loadings.

Once multiple ST datasets are integrated by INSPIRE, users can:
* Identify spatial trajectories and major spatial regions consistently across datasets using latent representations of cells or spatial spots.
* Discover detialed tissue architectures, spatial distributions of cell types, and the organization of biological processes using non-negative spatial factors on cells or spatial spots.
* Detect spatial variable genes, identify gene programs associated with the detailed spatial organizations in tissues, and perform pathway enrichment analysis using non-negative gene loadings.


## Installation
* Portal can be downloaded from GitHub:
```bash
git clone https://github.com/jiazhao97/INSPIRE.git
cd INSPIRE
conda env update --f environment.yml
conda activate INSPIRE
```
