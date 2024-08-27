import pandas as pd
import numpy as np
import scanpy as sc
import anndata as ad
import scipy.sparse
import os
from sklearn.metrics import pairwise_distances
from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from annoy import AnnoyIndex


def preprocess(adata_st_list, # list of spatial transcriptomics anndata objects
               num_hvgs, # number of highly variable genes to be selected for each anndata
               min_genes_qc, # minimum number of genes expressed required for a cell to pass quality control filtering
               min_cells_qc, # minimum number of cells expressed required for a gene to pass quality control filtering
               spot_size, # spot size used in "sc.pl.spatial" for visualization spatial data
               min_concat_dist=50, # minimum distance among data used to re-calculating spatial locations for better visualizations
               limit_num_genes=False, # whether datasets to be integrated only have a limited number of shared genes
              ):
    ## If limit_num_genes=True, get shared genes from datasets before performing any other preprocessing step.
    # Get shared genes
    if limit_num_genes == True:
        print("Get shared genes among all datasets...")
        for i, adata_st in enumerate(adata_st_list):
            if i == 0:
                genes_shared = adata_st.var.index
            else:
                genes_shared = genes_shared & adata_st.var.index
        genes_shared = sorted(list(genes_shared))
        for i, adata_st in enumerate(adata_st_list):
            adata_st_list[i] = adata_st_list[i][:, genes_shared]
        print("Find", str(len(genes_shared)), "shared genes among datasets.")

    
    ## Find shared highly varialbe genes among anndata as features
    print("Finding highly variable genes...")
    for i, adata_st in enumerate(adata_st_list):
        # Remove mt-genes
        adata_st_list[i].var_names_make_unique()
        adata_st_list[i] = adata_st_list[i][:, np.array(~adata_st_list[i].var.index.isna())
                                             & np.array(~adata_st_list[i].var_names.str.startswith("mt-"))
                                             & np.array(~adata_st_list[i].var_names.str.startswith("MT-"))]
        # Remove cells and genes for quality control
        print("shape of adata "+str(i)+" before quality control: ", adata_st_list[i].shape)
        sc.pp.filter_cells(adata_st_list[i], min_genes=min_genes_qc)
        sc.pp.filter_genes(adata_st_list[i], min_cells=min_cells_qc)
        print("shape of adata "+str(i)+" after quality control: ", adata_st_list[i].shape)
        # Find hvgs
        sc.pp.highly_variable_genes(adata_st_list[i], flavor='seurat_v3', n_top_genes=num_hvgs)
        hvgs = adata_st_list[i].var[adata_st_list[i].var.highly_variable == True].sort_values(by="highly_variable_rank").index
        if i == 0:
            hvgs_shared = hvgs
        else:
            hvgs_shared = hvgs_shared & hvgs
        # Add slice label
        adata_st_list[i].obs['slice'] = i
        adata_st_list[i].obs["slice"] = adata_st_list[i].obs["slice"].values.astype(int)
        # Add slice label to barcodes
        adata_st_list[i].obs.index = adata_st_list[i].obs.index + "-" + str(i)
    hvgs_shared = sorted(list(hvgs_shared))
    print("Find", str(len(hvgs_shared)), "shared highly variable genes among datasets.")

    
    ## Concatenate datasets as a full anndata for better visualization
    print("Concatenate datasets as a full anndata for better visualization...")
    ads = []
    for i, adata_st in enumerate(adata_st_list):
        if i == 0:
            ads.append(adata_st.copy())
        else:
            ad_tmp = adata_st.copy()
            xmax_1 = np.max(ads[i-1].obsm["spatial"][:,0])
            xmin_2 = np.min(ad_tmp.obsm["spatial"][:,0])
            ymax_1 = np.max(ads[i-1].obsm["spatial"][:,1])
            ymax_2 = np.max(ad_tmp.obsm["spatial"][:,1])
            ad_tmp.obsm["spatial"][:,0] = ad_tmp.obsm["spatial"][:,0] + (xmax_1 - xmin_2) + min_concat_dist
            ad_tmp.obsm["spatial"][:,1] = ad_tmp.obsm["spatial"][:,1] + (ymax_1 - ymax_2)
            ads.append(ad_tmp.copy())
    del ad_tmp
    adata_full = ad.concat(ads, join="outer")
    sc.pl.spatial(adata_full, spot_size=spot_size)
    del ads


    ## Store counts and library sizes for Poisson modeling, and normalize data for encoder
    print("Store counts and library sizes for Poisson modeling...")
    print("Normalize data...")
    target_sum = 1e4
    if limit_num_genes == True:
        target_sum = 1e3
    for i, adata_st in enumerate(adata_st_list):
        # Store counts and library sizes for Poisson modeling
        st_mtx = adata_st[:, hvgs_shared].X.copy()
        if scipy.sparse.issparse(st_mtx):
            st_mtx = st_mtx.toarray()
        adata_st_list[i].obsm["count"] = st_mtx
        st_library_size = np.sum(st_mtx, axis=1)
        adata_st_list[i].obs["library_size"] = st_library_size
        # Normalize data
        sc.pp.normalize_total(adata_st_list[i], target_sum=target_sum)
        sc.pp.log1p(adata_st_list[i])
        adata_st_list[i] = adata_st_list[i][:, hvgs_shared]
        if scipy.sparse.issparse(adata_st_list[i].X):
            adata_st_list[i].X = adata_st_list[i].X.toarray()

    return adata_st_list, adata_full



def build_graph_GAT(adata_st_list, # list of spatial transcriptomics anndata objects after "preprocess" step
                    rad_cutoff=None, # radius for finding neighbors of spots/cells
                    rad_coef=None, # if rad_cutoff is not provided, we calculate mininal distance between spots/cells (min_dist_ref) and set rad_cutoff=rad_coef*min_dist_ref
                   ):
    print("Start building graphs...")
    
    ## Calculate radius cutoff if it is not provided
    if rad_cutoff is None:
        print("Calculate radius cutoff based on 'rad_coef' and mininal distance between spots/cells within a dataset...")
        loc_ref = np.array(adata_st_list[0].obsm["spatial"]).copy()
        pair_dist_ref = pairwise_distances(loc_ref)
        min_dist_ref = np.sort(np.unique(pair_dist_ref), axis=None)[1]
        rad_cutoff = min_dist_ref * rad_coef
    print("Radius for graph connection is %.4f." % rad_cutoff)


    ## Calculate graph connecting neighboring spots/cells
    print("Build graphs for GAT networks")
    for i, adata_st in enumerate(adata_st_list):
        # Calculate adjacent matrix describing neighboring graph
        loc = pd.DataFrame(adata_st_list[i].obsm["spatial"]).values
        pair_dist = pairwise_distances(loc)
        G = (pair_dist < rad_cutoff).astype(float)
        print('%.4f neighbors per cell on average.' % (np.mean(np.sum(G, axis=1)) - 1))
        adata_st_list[i].obsm["graph"] = G
        
        # Calculate cosine similarity of features among spots/cells
        pair_dist_cos = pairwise_distances(adata_st.X, metric="cosine") # 1 - cosine_similarity
        adata_st_list[i].obsm["graph_cos"] = 1 - pair_dist_cos

    return adata_st_list



def build_graph_LGCN(adata_st_list, # list of spatial transcriptomics anndata objects after "preprocess" step
                     rad_cutoff_list, # list of radius for finding neighbors of spots/cells
                     k_lgcn=1, # number of aggregation steps for constructing features for LGCN
                    ):
    print("Start building graphs...")

    ## Calculate graph connecting neighboring spots/cells
    print("Build graphs and prepare node features for LGCN networks")
    assert len(rad_cutoff_list) == len(adata_st_list)
    for i, adata_st in enumerate(adata_st_list):
        # Calculate graphs
        rad_cutoff = rad_cutoff_list[i]
        print("Radius for graph connection is %.4f." % rad_cutoff)
        loc = pd.DataFrame(adata_st_list[i].obsm["spatial"]).values
        pair_dist = pairwise_distances(loc)
        G = (pair_dist < rad_cutoff).astype(float)
        print('%.4f neighbors per cell on average.' % (np.mean(np.sum(G, axis=1)) - 1))

        # Calculate normalized adjacency matrix
        sG = scipy.sparse.coo_matrix(G)
        row, col, edge_weight = sG.row, sG.col, sG.data
        deg = np.sum(G, axis=1)
        deg_inv_sqrt = np.power(deg, -0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        adjG = scipy.sparse.coo_array((edge_weight, (row, col)), shape=G.shape).toarray()

        # Calculate node features for LGCN 
        if scipy.sparse.issparse(adata_st_list[i].X):
            Xf = adata_st_list[i].X.toarray()
        else:
            Xf = adata_st_list[i].X
        Xfs = [Xf]
        for j in range(k_lgcn):
            Xf = adjG @ Xfs[-1]
            Xfs.append(Xf)
        node_features = np.concatenate(Xfs, axis=1)
        print("Node features for slice", str(i), ":", node_features.shape)
        adata_st_list[i].obsm["node_features"] = node_features

    return adata_st_list



def calculate_node_features_LGCN(adata, # anndata object of one spatial transcriptomics dataset
                                 slice_name, # name of spatial transcriptomics dataset
                                 preprocessed_data_path, # node features will be saved into "preprocessed_data_path/node_features" as "slice_name.h5ad"
                                 min_genes_qc, # minimum number of genes expressed required for a cell to pass quality control filtering
                                 min_cells_qc, # minimum number of cells expressed required for a gene to pass quality control filtering
                                 rad_cutoff, # radius for finding neighbors of spots/cells
                                ):
    if not os.path.exists(preprocessed_data_path+"/raw_counts"):
        os.makedirs(preprocessed_data_path+"/raw_counts")
    if not os.path.exists(preprocessed_data_path+"/node_features"):
        os.makedirs(preprocessed_data_path+"/node_features")

    # Remove mt-genes
    adata.var_names_make_unique()
    adata = adata[:, np.array(~adata.var.index.isna())
                   & np.array(~adata.var_names.str.startswith("mt-"))
                   & np.array(~adata.var_names.str.startswith("MT-"))]

    # Remove cells and genes for quality control
    print("Shape of adata before quality control: ", adata.shape)
    sc.pp.filter_cells(adata, min_genes=min_genes_qc)
    sc.pp.filter_genes(adata, min_cells=min_cells_qc)
    print("Shape of adata after quality control: ", adata.shape)

    # Save raw counts
    adata.write(preprocessed_data_path+"/raw_counts/"+slice_name+".h5ad")

    # Log-normalized data
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    if scipy.sparse.issparse(adata.X):
        adata.X = adata.X.toarray()

    # Calculate graph
    loc = adata.obsm["spatial"]
    pair_dist = pairwise_distances(loc)
    G = (pair_dist < rad_cutoff)
    del pair_dist
    G = G.astype(float)
    print('%.4f neighbors per cell on average.' % (np.mean(np.sum(G, axis=1)) - 1))

    # Calculate normalized adjacency matrix
    deg = np.sum(G, axis=1)
    sG = scipy.sparse.coo_matrix(G)
    n_spots = G.shape[0]
    del G
    row, col, edge_weight = sG.row, sG.col, sG.data
    del sG
    deg_inv_sqrt = np.power(deg, -0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    adjG = scipy.sparse.coo_array((edge_weight, (row, col)), shape=(n_spots, n_spots)).toarray()

    # Calculate node features for LGCN 
    adata.X = adjG @ adata.X
    adata.write(preprocessed_data_path+"/node_features/"+slice_name+".h5ad")
    print("Shape of node features: ", adata.X.shape)



def prepare_inputs_LGCN(slice_name_list, # list of slice names
                        preprocessed_data_path, # same preprocessed_data_path for "calculate_node_features_LGCN" and "prepare_inputs_LGCN"
                        num_hvgs, # number of highly variable genes to be selected for each anndata
                        spot_size, # s used in "sc.pl.spatial" for visualization spatial data
                        min_concat_dist=50, # minimum distance among data used to re-calculating spatial locations for better visualizations
                       ):
    n_slices = len(slice_name_list)

    ## Find shared highly varialbe genes among anndata as features
    print("Finding highly variable genes...")
    for i in range(n_slices):
        # Load data
        print("Load data", slice_name_list[i])
        adata = sc.read_h5ad(os.path.join(preprocessed_data_path, "raw_counts/"+slice_name_list[i]+".h5ad"))
        # Find hvgs
        sc.pp.highly_variable_genes(adata, flavor='seurat_v3', n_top_genes=num_hvgs)
        hvgs = adata.var[adata.var.highly_variable == True].sort_values(by="highly_variable_rank").index
        if i == 0:
            hvgs_shared = hvgs
        else:
            hvgs_shared = hvgs_shared & hvgs
    hvgs_shared = sorted(list(hvgs_shared))
    print("Find", str(len(hvgs_shared)), "shared highly variable genes among datasets.")


    ## Store counts and library sizes for Poisson modeling, and normalize data for encoder
    print("Store counts and library sizes for Poisson modeling...")
    print("Normalize data...")
    adata_st_list = []
    for i in range(n_slices):
        # Load data
        print("Load data", slice_name_list[i])
        adata = sc.read_h5ad(os.path.join(preprocessed_data_path, "raw_counts/"+slice_name_list[i]+".h5ad"))
        
        # Create anndata snd store counts
        st_mtx = adata[:, hvgs_shared].X.copy()
        adata_st_slice = ad.AnnData(np.zeros(st_mtx.shape))
        adata_st_slice.var.index = hvgs_shared
        adata_st_slice.obs.index = adata.obs.index
        adata_st_list.append(adata_st_slice)
        if scipy.sparse.issparse(st_mtx):
            st_mtx = st_mtx.toarray()
        adata_st_list[i].obsm["count"] = st_mtx
        
        # Store library sizes for Poisson modeling
        st_library_size = np.sum(st_mtx, axis=1)
        adata_st_list[i].obs["library_size"] = st_library_size
        
        # Store log-normalized data
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        st_lognorm = adata[:, hvgs_shared].X.copy()
        if scipy.sparse.issparse(st_lognorm):
            st_lognorm = st_lognorm.toarray()
        adata_st_list[i].obsm["node_features"] = st_lognorm

        # Store spot/cell spatial locations
        adata_st_list[i].obsm["spatial"] = adata.obsm["spatial"]

        # Add slice label
        adata_st_list[i].obs['slice'] = i
        adata_st_list[i].obs["slice"] = adata_st_list[i].obs["slice"].values.astype(int)


    ## Load and prepare node features for LGCN
    print("Load and prepare node features for LGCN...")
    for i in range(n_slices):
        # Load pre-calculated node features AX
        print("Load node features", slice_name_list[i])
        adata = sc.read_h5ad(os.path.join(preprocessed_data_path, "node_features/"+slice_name_list[i]+".h5ad"))
        adata = adata[adata_st_list[i].obs.index, :]
        node_fts = adata[:, hvgs_shared].X.copy()
        if scipy.sparse.issparse(node_fts):
            node_fts = node_fts.toarray()

        # Concate AX with X
        adata_st_list[i].obsm["node_features"] = np.concatenate([adata_st_list[i].obsm["node_features"], node_fts], axis=1)
        print("Node features for slice", str(i), ":", adata_st_list[i].obsm["node_features"].shape)

        # Add slice label to barcodes
        adata_st_list[i].obs.index = adata_st_list[i].obs.index + "-" + str(i)


    ## Prepare an adata containing full spot locations and slice labels for later visualization of results...
    print("Prepare an adata containing full spot locations and slice labels for better visualization...")
    for i in range(n_slices):
        # Re-calculate spatial locations
        if i > 0:
            xmax_1 = np.max(adata_st_list[i-1].obsm["spatial"][:,0])
            xmin_2 = np.min(adata_st_list[i].obsm["spatial"][:,0])
            ymax_1 = np.max(adata_st_list[i-1].obsm["spatial"][:,1])
            ymax_2 = np.max(adata_st_list[i].obsm["spatial"][:,1])
            adata_st_list[i].obsm["spatial"][:,0] = adata_st_list[i].obsm["spatial"][:,0] + (xmax_1 - xmin_2) + min_concat_dist
            adata_st_list[i].obsm["spatial"][:,1] = adata_st_list[i].obsm["spatial"][:,1] + (ymax_1 - ymax_2)
    loc_full = np.concatenate([adata_st_list[i].obsm["spatial"] for i in range(n_slices)], axis=0)

    # Create anndata
    for i in range(n_slices):
        ad_tmp = ad.AnnData(np.zeros((adata_st_list[i].shape[0], 1)))
        ad_tmp.obs.index = adata_st_list[i].obs.index
        ad_tmp.var.index = ["gene1"]
        ad_tmp.obs["slice_label"] = i
        ad_tmp.obs["slice_label"] = ad_tmp.obs["slice_label"].values.astype(int)
        if i == 0:
            adata = ad_tmp.copy()
        else:
            adata = ad.concat([adata, ad_tmp.copy()], join="outer")
    del ad_tmp
    adata.obsm["spatial"] = loc_full
    sc.pl.spatial(adata, spot_size=spot_size)

    return adata_st_list, adata



def calculate_ASW(latent_representations, # latent representations of cells or spatial spots
                  annotations, # annotations for cells or spatial spots
                 ):
    ASW = silhouette_score(latent_representations, annotations)
    return ASW



def calculate_ARI(annotations, # annotations for cells or spatial spots
                  spatial_domain_identification, # spatial domains assigned to cells or spatial spots
                 ):
    ARI = adjusted_rand_score(annotations, spatial_domain_identification)
    return ARI



def calculate_NMI(annotations, # annotations for cells or spatial spots
                  spatial_domain_identification, # spatial domains assigned to cells or spatial spots
                 ):
    NMI = normalized_mutual_info_score(annotations, spatial_domain_identification)
    return NMI



def calculate_factor_diversity(basis, #non-negative gene loading matrix
                               n_top_genes, # number of top genes considered for each spatial factor
                              ):
    n_spatial_factors = basis.shape[0]
    top_gene_factors = np.zeros((n_spatial_factors, n_top_genes))
    for i in range(n_spatial_factors):
        top_gene_factors[i,:] = np.argsort(-basis[i,:])[:n_top_genes]

    unique_gene_factors = np.zeros((n_spatial_factors, n_top_genes))
    for i in range(n_spatial_factors):
        for j in range(n_top_genes):
            unique_gene_factors[i,j] = np.sum(top_gene_factors - top_gene_factors[i,j] == 0.) - 1

    factor_diversity = np.mean(np.sum(unique_gene_factors == 0., axis=1) / n_top_genes)
    return factor_diversity



def calculate_factor_coherence(basis, #non-negative gene loading matrix
                               n_top_genes, # number of top genes considered for each spatial factor
                               gene_counts, # cell/spot by gene count matrix
                              ):
    n_spatial_factors = basis.shape[0]
    top_gene_factors = np.zeros((n_spatial_factors, n_top_genes))
    for i in range(n_spatial_factors):
        top_gene_factors[i,:] = np.argsort(-basis[i,:])[:n_top_genes]

    gene_counts[gene_counts < 0] = 0
    gene_counts[gene_counts > 0] = 1

    npmi = []
    for i in range(n_spatial_factors):
        for m in range(n_top_genes):
            for n in range(m+1, n_top_genes):
                gene_1 = top_gene_factors[i, m]
                gene_2 = top_gene_factors[i ,n]
                co_cnts = gene_counts[:, [int(gene_1), int(gene_2)]]
                p_joint = np.sum(np.sum(co_cnts, axis=1) == 2) / co_cnts.shape[0]
                p_1 = np.sum(co_cnts[:,0] == 1) / co_cnts.shape[0]
                p_2 = np.sum(co_cnts[:,1] == 1) / co_cnts.shape[0]
                npmi_val = - np.log(p_joint/(p_1*p_2)) / np.log(p_joint)
                npmi.append(npmi_val)
    
    factor_coherence = np.mean(npmi)
    return factor_coherence



def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[m-1,:] *= -1
       R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R,centroid_A.T)

    # homogeneous transformation
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t



def transform(point_cloud, T):
    point_cloud_align = np.ones((point_cloud.shape[0], 3))
    point_cloud_align[:,0:2] = np.copy(point_cloud)
    point_cloud_align = np.dot(T, point_cloud_align.T).T
    return point_cloud_align[:, :2]



def acquire_pairs(X, Y, k=30, metric='angular'):
    f = X.shape[1]
    t1 = AnnoyIndex(f, metric)
    t2 = AnnoyIndex(f, metric)
    for i in range(len(X)):
        t1.add_item(i, X[i])
    for i in range(len(Y)):
        t2.add_item(i, Y[i])
    t1.build(10)
    t2.build(10)

    mnn_mat = np.bool_(np.zeros((len(X), len(Y))))
    sorted_mat = np.array([t2.get_nns_by_vector(item, k) for item in X])
    for i in range(len(sorted_mat)):
        mnn_mat[i,sorted_mat[i]] = True
    _ = np.bool_(np.zeros((len(X), len(Y))))
    sorted_mat = np.array([t1.get_nns_by_vector(item, k) for item in Y])
    for i in range(len(sorted_mat)):
        _[sorted_mat[i],i] = True
    mnn_mat = np.logical_and(_, mnn_mat).astype(int)
    return mnn_mat


