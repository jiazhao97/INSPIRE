import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import scipy.sparse
import umap
from tqdm import tqdm
from INSPIRE.networks import *


class Model_GAT():
    def __init__(self,
                 adata_st_list, # list of spatial transcriptomics anndata objects after "preprocess" and "build_graph" steps
                 n_spatial_factors, # number of spatial factors in biologically interpretable dimension reduction
                 n_training_steps, # number of training steps
                 hidden_dims=[512,32], # dimensionalities of hidden layers in "IntegrationNet"
                 coef_recon=1.0, # coefficient of reconstruction loss
                 coef_geom=0.02, # coefficient of geometry loss
                 use_margin=True, # whether use the margin design in discriminators
                 margin_warmup_step=100, # margin will be activated after #margin_warmup_step steps
                 lr_d=5e-4, # learning rate for training "DiscriminatorNet"
                 seed=1234, # random seed
                ):

        # set hyperparameters
        self.n_slices = len(adata_st_list)
        self.n_heads = 1 # number of attention heads in GAT layers
        self.n_hidden_d = 512 # dimensionality of hidden layer in "DiscriminatorNet"
        self.slice_emb_dim = 4 # dimensionality of embedding space encoding slice labels
        self.lr = 5e-4 # learning rate for training "IntegrationNet"
        self.weight_decay = 1e-4 # weight decay for training "IntegrationNet"
        self.weight_decay_d = 1e-4 # weight decay for training "DiscriminatorNet"
        self.step_interval = 500 # interval of steps for showing objective values

        self.coef_fe = 1.0 # coefficient of auto-encoder loss for features
        self.coef_beta = 1.0 # coefficient of topic proportion penalty (Dirichlet distribution prior)
        self.coef_gan = 1.0 # coefficient of GAN loss

        self.n_spatial_factors = n_spatial_factors
        self.n_training_steps = n_training_steps
        self.hidden_dims = [adata_st_list[0].shape[1]] + hidden_dims
        self.coef_recon = coef_recon
        self.coef_geom = coef_geom
        self.use_margin = use_margin
        self.lr_d = lr_d
        self.seed = seed

        self.margin_warmup_step = margin_warmup_step
        self.margin = 5.0
        if self.use_margin != True:
            self.margin = 50.0

        self.n_spot = 0
        for i in range(self.n_slices):
            self.n_spot = self.n_spot + adata_st_list[i].shape[0]

        # record hvg names
        self.shared_hvgs = adata_st_list[0].var.index

        # set device and random seed
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.benchmark = True

        # setup data
        self.node_feats_dict = {}
        self.adj_matrix_dict = {}
        self.count_matrix_dict = {}
        self.library_size_dict = {}
        self.slice_label_dict = {}
        self.graph_cos_dict = {}
        for i in range(self.n_slices):
            if scipy.sparse.issparse(adata_st_list[i].X):
                self.node_feats_dict[i] = torch.from_numpy(adata_st_list[i].X.toarray()).float().to(self.device)
            else:
                self.node_feats_dict[i] = torch.from_numpy(adata_st_list[i].X).float().to(self.device)
            self.adj_matrix_dict[i] = torch.from_numpy(np.array(adata_st_list[i].obsm["graph"])).float().to(self.device)
            self.count_matrix_dict[i] = torch.from_numpy(np.array(adata_st_list[i].obsm["count"])).float().to(self.device)
            self.library_size_dict[i] = torch.from_numpy(np.array(adata_st_list[i].obs["library_size"].values.reshape(-1, 1))).float().to(self.device)
            self.slice_label_dict[i] = torch.from_numpy(np.array(adata_st_list[i].obs["slice"].values)).long().to(self.device)
            self.graph_cos_dict[i] = torch.from_numpy(np.array(adata_st_list[i].obsm["graph_cos"])).float().to(self.device)

        # setup networks and optimizers
        self.net = IntegrationNet_GAT(hidden_dims=self.hidden_dims,
                                      n_heads=self.n_heads,
                                      n_factors=self.n_spatial_factors,
                                      n_slices=self.n_slices,
                                      slice_emb_dim=self.slice_emb_dim
                                     ).to(self.device)
        self.optimizer_net = optim.Adamax(list(self.net.parameters()), lr=self.lr, weight_decay=self.weight_decay)

        self.discriminator = {}
        d_params = []
        for i in range(self.n_slices-1):
            self.discriminator[i] = DiscriminatorNet(n_input=self.hidden_dims[2], 
                                                     n_hidden=self.n_hidden_d
                                                    ).to(self.device)
            d_params = d_params + list(self.discriminator[i].parameters())
        self.optimizer_d = optim.Adam(d_params, lr=self.lr_d, weight_decay=self.weight_decay_d)


    def train(self, record_final_loss=False):
        self.net.train()
        for i in range(self.n_slices-1):
            self.discriminator[i].train()

        # steps to record loss
        step_list = [self.n_training_steps-1, self.n_training_steps-51, self.n_training_steps-101, self.n_training_steps-151, 
                     self.n_training_steps-201, self.n_training_steps-301, self.n_training_steps-401]

        for step in tqdm(range(self.n_training_steps)):
            # outputs from networks
            Zs, betas, alphas, node_feats_recons, basis_val, gammas = self.net(self.adj_matrix_dict, self.node_feats_dict, self.slice_label_dict)

            # discriminator loss
            self.optimizer_d.zero_grad()
            loss_d = 0.
            if step <= self.margin_warmup_step:
                for i in range(self.n_slices-1):
                    # batch [i] as real; batch [i+1] as fake
                    loss_d = loss_d + torch.mean(torch.log(1 + torch.exp(-self.discriminator[i](Zs[i])))) + torch.mean(torch.log(1 + torch.exp(self.discriminator[i](Zs[i+1]))))
            else:
                for i in range(self.n_slices-1):
                    # batch [i] as real; batch [i+1] as fake
                    loss_d = loss_d + torch.mean(torch.log(1 + torch.exp(-torch.clamp(self.discriminator[i](Zs[i]), -self.margin, self.margin)))) + torch.mean(torch.log(1 + torch.exp(torch.clamp(self.discriminator[i](Zs[i+1]), -self.margin, self.margin))))
            loss_d_opt = self.coef_gan * loss_d
            loss_d_opt.backward(retain_graph=True)
            self.optimizer_d.step()

            # auto-encoder loss of node features, reconstruction loss, geometry loss
            features_loss = 0.
            recon_loss = 0.
            geom_loss = 0.
            log_lam_dict = {}
            Zs_norm_dict = {}
            for i in range(self.n_slices):
                features_loss = features_loss + torch.mean(torch.sqrt(torch.sum(torch.pow(self.node_feats_dict[i]-node_feats_recons[i], 2), axis=1)))
                log_lam_dict[i] = torch.log(torch.matmul(betas[i], basis_val) + 1e-6) + gammas[self.slice_label_dict[i]]
                recon_loss = recon_loss - torch.mean(torch.sum(self.count_matrix_dict[i] * 
                                                         (torch.log(self.library_size_dict[i] + 1e-6) + log_lam_dict[i]) - self.library_size_dict[i] * torch.exp(log_lam_dict[i]), axis=1))
                Zs_norm_dict[i] = F.normalize(Zs[i], p=2)
                geom_loss = geom_loss + torch.mean(torch.sum(torch.pow(torch.matmul(Zs_norm_dict[i], torch.transpose(Zs_norm_dict[i],0,1)) - self.graph_cos_dict[i], 2), axis=1))

            # beta loss
            for i in range(self.n_slices):
                if i == 0:
                    p_aver = torch.sum(betas[i], axis=0)
                else:
                    p_aver = p_aver + torch.sum(betas[i], axis=0)
            p_aver = p_aver / self.n_spot
            prior_val = 5.0
            beta_loss = - torch.sum((prior_val - 1.) * torch.log(p_aver + 1e-6)) * 1.0

            # gan loss
            gan_loss = 0.
            if step <= self.margin_warmup_step:
                for i in range(self.n_slices-1):
                    gan_loss = gan_loss + torch.mean(torch.log(1 + torch.exp(-self.discriminator[i](Zs[i+1]))))
            else:
                for i in range(self.n_slices-1):
                    gan_loss = gan_loss + torch.mean(torch.log(1 + torch.exp(-torch.clamp(self.discriminator[i](Zs[i+1]), -self.margin, self.margin))))

            # total loss
            self.optimizer_net.zero_grad()
            loss_total = self.coef_fe*features_loss + self.coef_recon*recon_loss + self.coef_geom*geom_loss + self.coef_beta*beta_loss + self.coef_gan*gan_loss
            loss_total.backward()
            self.optimizer_net.step()

            if not step % self.step_interval:
                print("Step: %s, d_loss: %.4f, Loss: %.4f, recon_loss: %.4f, fe_loss: %.4f, geom_loss: %.4f, beta_loss: %.4f, gan_loss: %.4f" % (step, loss_d.item(), loss_total.item(), recon_loss.item(), features_loss.item(), geom_loss.item(), beta_loss.item(), gan_loss.item()))

            if record_final_loss == True:
                if step in step_list:
                    loss_values = {}
                    loss_values["step"] = step
                    loss_values["d_loss"] = loss_d.item()
                    loss_values["total_loss"] = loss_total.item()
                    loss_values["recon_loss"] = recon_loss.item()
                    loss_values["fe_loss"] = features_loss.item()
                    loss_values["geom_loss"] = gan_loss.item()
                    loss_values["beta_loss"] = beta_loss.item()
                    loss_values["gan_loss"] = gan_loss.item()
                    self.loss_val.append(loss_values)

        if record_final_loss == True:
            return self.loss_val


    def eval(self,
             adata_full, # full concatenated anndata of all datasets generated by "preprocess" step
             eval_d_scores=False, # whether evaluate discriminator scores
             ):
        self.net.eval()
        Zs, betas, alphas, node_feats_recons, basis_val, gammas = self.net(self.adj_matrix_dict, self.node_feats_dict, self.slice_label_dict)

        # betas
        print("Add cell/spot proportions of spatial factors into adata_full.obs...")
        for i in range(self.n_slices):
            b = betas[i].detach().cpu().numpy()
            if i == 0:
                b_full = b
            else:
                b_full = np.concatenate((b_full, b), axis=0)
        decon_res = pd.DataFrame(b_full, columns=["Proportion of spatial factor "+str(j+1) for j in range(b_full.shape[1])])
        decon_res.index = adata_full.obs.index
        adata_full.obs = adata_full.obs.join(decon_res)

        # visualize Zs
        print("Add cell/spot latent representations into adata_full.obsm['latent']...")
        for i in range(self.n_slices):
            Z = Zs[i].detach().cpu().numpy()
            cell_reps = pd.DataFrame(Z)
            cell_reps.index = ["slice-"+str(i)+"-cell-"+str(j) for j in range(cell_reps.shape[0])]
            if i == 0:
                cell_reps_total = cell_reps
            else:
                cell_reps_total = pd.concat([cell_reps_total, cell_reps])
        adata_full.obsm['latent'] = cell_reps_total.values

        # evaluate discriminator scores
        if eval_d_scores == True:
            print("Evaluate discriminator scores...")
            for i in range(self.n_slices-1):
                self.discriminator[i].eval()
            d_score_dict = {}
            for i in range(self.n_slices-1):
                d_score_dict[i] = {}
                d_score_dict[i][0] = self.discriminator[i](Zs[i]).detach().cpu().numpy()
                d_score_dict[i][1] = self.discriminator[i](Zs[i+1]).detach().cpu().numpy()

        # basis
        basis = basis_val.detach().cpu().numpy()
        basis_df = pd.DataFrame(basis, columns=self.shared_hvgs)

        if eval_d_scores == True:
            return adata_full, basis_df, d_score_dict
        else:
            return adata_full, basis_df



class Model_LGCN():
    def __init__(self,
                 adata_st_list, # list of spatial transcriptomics anndata objects after "preprocess" and "build_graph" steps
                 n_spatial_factors, # number of spatial factors in biologically interpretable dimension reduction
                 n_training_steps, # number of training steps
                 batch_size, # number of samples in a mini-batch
                 hidden_dims=[512,32], # dimensionalities of hidden layers in "IntegrationNet"
                 coef_recon=1.0, # coefficient of reconstruction loss
                 coef_geom=0.02, # coefficient of geometry loss
                 use_margin=True, # whether use the margin design in discriminators
                 lr_d=5e-4, # learning rate for training "DiscriminatorNet"
                 different_platforms=False, # whether integrate datasets across different platforms
                 seed=1234, # random seed
                ):

        # set hyperparameters
        self.n_slices = len(adata_st_list)
        self.n_hidden_d = 512 # dimensionality of hidden layer in "DiscriminatorNet"
        self.slice_emb_dim = 4 # dimensionality of embedding space encoding slice labels
        self.lr = 5e-4 # learning rate for training "IntegrationNet"
        self.weight_decay = 1e-4 # weight decay for training "IntegrationNet"
        self.weight_decay_d = 1e-4 # weight decay for training "DiscriminatorNet"
        self.step_interval = 500 # interval of steps for showing objective values

        self.coef_fe = 1.0 # coefficient of auto-encoder loss for features
        self.coef_beta = 1.0 # coefficient of topic proportion penalty (Dirichlet distribution prior)
        self.coef_gan = 1.0 # coefficient of GAN loss

        self.n_spatial_factors = n_spatial_factors
        self.n_training_steps = n_training_steps
        self.batch_size = batch_size
        self.hidden_dims = hidden_dims
        self.n_features_lgcn = adata_st_list[0].obsm["node_features"].shape[1]
        self.n_hvgs = adata_st_list[0].obsm["count"].shape[1]
        self.coef_recon = coef_recon
        self.coef_geom = coef_geom
        self.use_margin = use_margin
        self.lr_d = lr_d
        self.different_platforms = different_platforms
        self.seed = seed

        self.margin_warmup_step = 100
        self.margin = 5.0
        if self.use_margin != True:
            self.margin = 50.0

        self.n_spot_list = [adata_st.shape[0] for adata_st in adata_st_list]

        # record hvg names
        self.shared_hvgs = adata_st_list[0].var.index

        # set device and random seed
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.benchmark = True

        # setup networks and optimizers
        self.net = IntegrationNet_LGCN(hidden_dims=self.hidden_dims,
                                       n_features_lgcn=self.n_features_lgcn,
                                       n_hvgs=self.n_hvgs,
                                       n_factors=self.n_spatial_factors,
                                       n_slices=self.n_slices,
                                       slice_emb_dim=self.slice_emb_dim,
                                       different_platforms=self.different_platforms
                                      ).to(self.device)
        self.optimizer_net = optim.Adamax(list(self.net.parameters()), lr=self.lr, weight_decay=self.weight_decay)

        self.discriminator = {}
        d_params = []
        for i in range(self.n_slices-1):
            self.discriminator[i] = DiscriminatorNet(n_input=self.hidden_dims[1],
                                                     n_hidden=self.n_hidden_d
                                                    ).to(self.device)
            d_params = d_params + list(self.discriminator[i].parameters())
        self.optimizer_d = optim.Adam(d_params, lr=self.lr_d, weight_decay=self.weight_decay_d)


    def train(self,
              adata_st_list, # list of spatial transcriptomics anndata objects after "preprocess" and "build_graph" steps
             ):
        self.net.train()
        for i in range(self.n_slices-1):
            self.discriminator[i].train()

        for step in tqdm(range(self.n_training_steps)):
            # sample minibatch and send data to device
            node_feats_dict = {}
            count_matrix_dict = {}
            library_size_dict = {}
            slice_label_dict = {}
            for i in range(self.n_slices):
                index_batch = np.random.choice(np.arange(self.n_spot_list[i]), size=self.batch_size)
                node_feats_dict[i] = torch.from_numpy(adata_st_list[i].obsm["node_features"][index_batch, :]).float().to(self.device)
                count_matrix_dict[i] = torch.from_numpy(np.array(adata_st_list[i].obsm["count"][index_batch, :])).float().to(self.device)
                library_size_dict[i] = torch.from_numpy(np.array(adata_st_list[i].obs["library_size"].values.reshape(-1, 1))[index_batch, :]).float().to(self.device)
                slice_label_dict[i] = torch.from_numpy(np.array(adata_st_list[i].obs["slice"].values))[index_batch].long().to(self.device)

            # outputs from networks
            Zs, betas, alphas, node_feats_recons, basis_val, gammas = self.net(node_feats_dict, slice_label_dict)

            # discriminator loss
            self.optimizer_d.zero_grad()
            loss_d = 0.
            if step <= self.margin_warmup_step:
                for i in range(self.n_slices-1):
                    # batch [i] as real; batch [i+1] as fake
                    loss_d = loss_d + torch.mean(torch.log(1 + torch.exp(-self.discriminator[i](Zs[i])))) + torch.mean(torch.log(1 + torch.exp(self.discriminator[i](Zs[i+1]))))
            else:
                for i in range(self.n_slices-1):
                    # batch [i] as real; batch [i+1] as fake
                    loss_d = loss_d + torch.mean(torch.log(1 + torch.exp(-torch.clamp(self.discriminator[i](Zs[i]), -self.margin, self.margin)))) + torch.mean(torch.log(1 + torch.exp(torch.clamp(self.discriminator[i](Zs[i+1]), -self.margin, self.margin))))
            loss_d_opt = self.coef_gan * loss_d
            loss_d_opt.backward(retain_graph=True)
            self.optimizer_d.step()

            # auto-encoder loss of node features, reconstruction loss, geometry loss
            features_loss = 0.
            recon_loss = 0.
            geom_loss = 0.
            log_lam_dict = {}
            Zs_norm_dict = {}
            Xs_norm_dict = {}
            for i in range(self.n_slices):
                features_loss = features_loss + torch.mean(torch.sqrt(torch.sum(torch.pow(node_feats_dict[i][:,:self.n_hvgs]-node_feats_recons[i], 2), axis=1)))
                log_lam_dict[i] = torch.log(torch.matmul(betas[i], basis_val) + 1e-6) + gammas[slice_label_dict[i]]
                recon_loss = recon_loss - torch.mean(torch.sum(count_matrix_dict[i] * 
                                                         (torch.log(library_size_dict[i] + 1e-6) + log_lam_dict[i]) - library_size_dict[i] * torch.exp(log_lam_dict[i]), axis=1))
                Zs_norm_dict[i] = F.normalize(Zs[i], p=2)
                Xs_norm_dict[i] = F.normalize(node_feats_dict[i][:,:self.n_hvgs], p=2)
                geom_loss = geom_loss + torch.mean(torch.sum(torch.pow(torch.matmul(Zs_norm_dict[i],torch.transpose(Zs_norm_dict[i],0,1)) - torch.matmul(Xs_norm_dict[i],torch.transpose(Xs_norm_dict[i],0,1)), 2), axis=1))

            # beta loss
            for i in range(self.n_slices):
                if i == 0:
                    p_aver = torch.sum(betas[i], axis=0)
                else:
                    p_aver = p_aver + torch.sum(betas[i], axis=0)
            # n_spot = self.batch_size * self.n_slices
            n_spot = self.batch_size * 4
            p_aver = p_aver / n_spot
            prior_val = 5.0
            beta_loss = - torch.sum((prior_val - 1.) * torch.log(p_aver + 1e-6)) * 1.0

            # gan loss
            gan_loss = 0.
            if step <= self.margin_warmup_step:
                for i in range(self.n_slices-1):
                    gan_loss = gan_loss + torch.mean(torch.log(1 + torch.exp(-self.discriminator[i](Zs[i+1]))))
            else:
                for i in range(self.n_slices-1):
                    gan_loss = gan_loss + torch.mean(torch.log(1 + torch.exp(-torch.clamp(self.discriminator[i](Zs[i+1]), -self.margin, self.margin))))

            # total loss
            self.optimizer_net.zero_grad()
            loss_total = self.coef_fe*features_loss + self.coef_recon*recon_loss + self.coef_geom*geom_loss + self.coef_beta*beta_loss + self.coef_gan*gan_loss
            loss_total.backward()
            self.optimizer_net.step()

            if not step % self.step_interval:
                print("Step: %s, d_loss: %.4f, Loss: %.4f, recon_loss: %.4f, fe_loss: %.4f, geom_loss: %.4f, beta_loss: %.4f, gan_loss: %.4f" % (step, loss_d.item(), loss_total.item(), recon_loss.item(), features_loss.item(), geom_loss.item(), beta_loss.item(), gan_loss.item()))


    def eval(self,
             adata_st_list, # list of spatial transcriptomics anndata objects after "preprocess" and "build_graph" steps
             adata_full, # full concatenated anndata of all datasets generated by "preprocess" step
             eval_d_scores=False, # whether evaluate discriminator scores
             ):
        # send full data to device
        node_feats_dict = {}
        slice_label_dict = {}
        for i in range(self.n_slices):
            node_feats_dict[i] = torch.from_numpy(adata_st_list[i].obsm["node_features"]).float().to(self.device)
            slice_label_dict[i] = torch.from_numpy(np.array(adata_st_list[i].obs["slice"].values)).long().to(self.device)

        # evaluate
        self.net.eval()
        Zs, betas, alphas, node_feats_recons, basis_val, gammas = self.net(node_feats_dict, slice_label_dict)

        # betas
        print("Add cell/spot proportions of spatial factors into adata_full.obs...")
        for i in range(self.n_slices):
            b = betas[i].detach().cpu().numpy()
            if i == 0:
                b_full = b
            else:
                b_full = np.concatenate((b_full, b), axis=0)
        decon_res = pd.DataFrame(b_full, columns=["Proportion of spatial factor "+str(j+1) for j in range(b_full.shape[1])])
        decon_res.index = adata_full.obs.index
        adata_full.obs = adata_full.obs.join(decon_res)

        # visualize Zs
        print("Add cell/spot latent representations into adata_full.obsm['latent']...")
        for i in range(self.n_slices):
            Z = Zs[i].detach().cpu().numpy()
            cell_reps = pd.DataFrame(Z)
            cell_reps.index = ["slice-"+str(i)+"-cell-"+str(j) for j in range(cell_reps.shape[0])]
            if i == 0:
                cell_reps_total = cell_reps
            else:
                cell_reps_total = pd.concat([cell_reps_total, cell_reps])
        adata_full.obsm['latent'] = cell_reps_total.values

        # evaluate discriminator scores
        if eval_d_scores == True:
            print("Evaluate discriminator scores...")
            for i in range(self.n_slices-1):
                self.discriminator[i].eval()
            d_score_dict = {}
            for i in range(self.n_slices-1):
                d_score_dict[i] = {}
                d_score_dict[i][0] = self.discriminator[i](Zs[i]).detach().cpu().numpy()
                d_score_dict[i][1] = self.discriminator[i](Zs[i+1]).detach().cpu().numpy()

        # basis
        basis = basis_val.detach().cpu().numpy()
        basis_df = pd.DataFrame(basis, columns=self.shared_hvgs)

        if eval_d_scores == True:
            return adata_full, basis_df, d_score_dict
        else:
            return adata_full, basis_df


    def eval_minibatch(self,
                       adata_st_list, # list of spatial transcriptomics anndata objects after "prepare_inputs_LGCN" step
                       adata_full, # full concatenated anndata of all datasets generated by "prepare_inputs_LGCN" step
                       batch_size, # batch size used for evaluating Zs and betas
                      ):
        # evaluate Zs and betas using minibatch
        self.net.eval()
        print("Evaluate Z and beta using minibatch...")
        s = 0
        for i in range(self.n_slices):
            print("Evaluation for slice", str(i))
            n_parts = adata_st_list[i].shape[0] // batch_size + 1
            for j in range(n_parts):
                if j < (n_parts-1):
                    node_feats = torch.from_numpy(adata_st_list[i].obsm["node_features"][batch_size*j:batch_size*(j+1), :]).float().to(self.device)
                    slice_label = torch.from_numpy(np.array(adata_st_list[i].obs["slice"].values[batch_size*j:batch_size*(j+1)])).long().to(self.device)
                else:
                    node_feats = torch.from_numpy(adata_st_list[i].obsm["node_features"][batch_size*j:, :]).float().to(self.device)
                    slice_label = torch.from_numpy(np.array(adata_st_list[i].obs["slice"].values[batch_size*j:])).long().to(self.device)

                #encoder
                Z = self.net.encoder(node_feats)
                # deconvolutioner
                slice_label_emb = self.net.slice_emb(slice_label)
                beta, _ = self.net.deconvolutioner(Z, slice_label_emb)

                if s == 0:
                    Z_full = Z.detach().cpu().numpy()
                    beta_full = beta.detach().cpu().numpy()
                else:
                    Z_full = np.concatenate([Z_full, Z.detach().cpu().numpy()], axis=0)
                    beta_full = np.concatenate([beta_full, beta.detach().cpu().numpy()], axis=0)
                s = s + 1
        assert Z_full.shape[0] == adata_full.shape[0]
        assert beta_full.shape[0] == adata_full.shape[0]
        decon_res = pd.DataFrame(beta_full, columns=["Proportion of spatial factor "+str(j+1) for j in range(beta_full.shape[1])])
        decon_res.index = adata_full.obs.index
        adata_full.obs = adata_full.obs.join(decon_res)
        adata_full.obsm['latent'] = Z_full

        # basis
        basis = self.net.get_basis().detach().cpu().numpy()
        basis_df = pd.DataFrame(basis, columns=self.shared_hvgs)

        return adata_full, basis_df

