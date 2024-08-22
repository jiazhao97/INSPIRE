import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class DenseLayer(nn.Module):

    def __init__(self, 
                 c_in, # dimensionality of input features
                 c_out, # dimensionality of output features
                 zero_init=False, # initialize weights as zeros; use Xavier uniform init if zero_init=False
                 ):
        super().__init__()
        
        self.linear = nn.Linear(c_in, c_out)

        # Initialization
        if zero_init:
            nn.init.zeros_(self.linear.weight.data)
        else:
            nn.init.uniform_(self.linear.weight.data, -np.sqrt(6 / (c_in + c_out)), np.sqrt(6 / (c_in + c_out)))
        nn.init.zeros_(self.linear.bias.data)

    def forward(self, 
                node_feats, # input node features
                ):
        node_feats = self.linear(node_feats)
        return node_feats



class GATSingleHead(nn.Module):

    def __init__(self, 
                 c_in, # dimensionality of input features
                 c_out, # dimensionality of output features
                 temp=1, # temperature parameter
                 ):

        super().__init__()

        self.linear = nn.Linear(c_in, c_out)
        self.v0 = nn.Parameter(torch.Tensor(c_out, 1))
        self.v1 = nn.Parameter(torch.Tensor(c_out, 1))
        self.temp = temp

        # Initialization
        nn.init.uniform_(self.linear.weight.data, -np.sqrt(6 / (c_in + c_out)), np.sqrt(6 / (c_in + c_out)))
        nn.init.zeros_(self.linear.bias.data)
        nn.init.uniform_(self.v0.data, -np.sqrt(6 / (c_out + 1)), np.sqrt(6 / (c_out + 1)))
        nn.init.uniform_(self.v1.data, -np.sqrt(6 / (c_out + 1)), np.sqrt(6 / (c_out + 1)))

    def forward(self, 
                node_feats, # input node features
                adj_matrix, # adjacency matrix including self-connections
                ):

        # Apply linear layer and sort nodes by head
        node_feats = self.linear(node_feats)
        f1 = torch.matmul(node_feats, self.v0)
        f2 = torch.matmul(node_feats, self.v1)
        attn_logits = adj_matrix * (f1 + f2.T)
        unnormalized_attentions = (F.sigmoid(attn_logits) - 0.5).to_sparse()
        attn_probs = torch.sparse.softmax(unnormalized_attentions / self.temp, dim=1)
        attn_probs = attn_probs.to_dense()
        node_feats = torch.matmul(attn_probs, node_feats)

        return node_feats



class GATMultiHead(nn.Module):

    def __init__(self, 
                 c_in, # dimensionality of input features
                 c_out, # dimensionality of output features
                 n_heads=1, # number of attention heads
                 concat_heads=True, # concatenate attention heads or not
                 ):

        super().__init__()

        self.n_heads = n_heads
        self.concat_heads = concat_heads
        if self.concat_heads:
            assert c_out % n_heads == 0, "The number of output features should be divisible by the number of heads."
            c_out = c_out // n_heads

        self.block = nn.ModuleList()
        for i_block in range(self.n_heads):
            self.block.append(GATSingleHead(c_in=c_in, c_out=c_out))

    def forward(self, 
                node_feats, # input node features
                adj_matrix, # adjacency matrix including self-connections
                ):

        res = []
        for i_block in range(self.n_heads):
            res.append(self.block[i_block](node_feats, adj_matrix))
        
        if self.concat_heads:
            node_feats = torch.cat(res, dim=1)
        else:
            node_feats = torch.mean(torch.stack(res, dim=0), dim=0)

        return node_feats



class DiscriminatorNet(nn.Module):

    def __init__(self, 
                 n_input, # dimensionality of input layer
                 n_hidden, # dimensionality of hidden layers
                 ):
        super().__init__()

        self.discriminator_layer1 = DenseLayer(n_input, n_hidden)
        self.discriminator_layer2 = DenseLayer(n_hidden, n_hidden)
        self.discriminator_layer3 = DenseLayer(n_hidden, 1)

    def forward(self, Z):
        H = F.relu(self.discriminator_layer1(Z))
        H = F.relu(self.discriminator_layer2(H))
        score = torch.clamp(self.discriminator_layer3(H), min=-50.0, max=50.0)
        return score



class IntegrationNet_GAT(nn.Module):

    def __init__(self, 
                 hidden_dims, # dimensionality of hidden layers
                 n_heads, # number of attention heads
                 n_factors, # number of topics in biologically interpretable dimension reduction
                 n_slices, # number of slices
                 slice_emb_dim, # dimensionality of slice id embedding
                 ):

        super().__init__()
        
        # define layers
        # encoder layers
        self.encoder_layer1 = GATMultiHead(hidden_dims[0], hidden_dims[1], n_heads=n_heads, concat_heads=True)
        self.encoder_layer2 = DenseLayer(hidden_dims[1], hidden_dims[2])
        # decoder layers
        self.decoder_layer1 = GATMultiHead(hidden_dims[2]+slice_emb_dim, hidden_dims[1], n_heads=n_heads, concat_heads=True)
        self.decoder_layer2 = DenseLayer(hidden_dims[1], hidden_dims[0])
        # deconvolution layers
        self.deconv_beta_layer = DenseLayer(hidden_dims[2], n_factors) # initialize
        self.factors = nn.Parameter(torch.Tensor(n_factors, hidden_dims[0]).uniform_(-np.sqrt(6/(n_factors+hidden_dims[0])), np.sqrt(6/(n_factors+hidden_dims[0]))))
        
        self.deconv_alpha_layer = DenseLayer(hidden_dims[2]+slice_emb_dim, 1, zero_init=True)
        self.gamma = nn.Parameter(torch.Tensor(n_slices, hidden_dims[0]).zero_())

        self.n_slices = n_slices
        self.slice_emb = nn.Embedding(self.n_slices, slice_emb_dim)

    def forward(self, 
                adj_matrix_dict, # dict of adjacency matrices that include self-connections
                node_feats_dict, # dict of input node features
                slice_label_dict, # dict of slice labels
                ):
        # create dicts
        Z_dict = {}
        slice_label_emb_dict = {}
        beta_dict = {}
        alpha_dict = {}
        node_feats_recon_dict = {}

        basis = self.get_basis()
        
        for i in range(self.n_slices):
            # encoder
            Z_dict[i] = self.encoder(adj_matrix_dict[i], node_feats_dict[i])
            
            # deconvolutioner
            slice_label_emb_dict[i] = self.slice_emb(slice_label_dict[i])
            beta_dict[i], alpha_dict[i] = self.deconvolutioner(Z_dict[i], slice_label_emb_dict[i])
            alpha_dict[i] = torch.clamp(alpha_dict[i], -0.1, 0.1)
            
            # decoder
            node_feats_recon_dict[i] = self.decoder(adj_matrix_dict[i], Z_dict[i], slice_label_emb_dict[i])

        return Z_dict, beta_dict, alpha_dict, node_feats_recon_dict, basis, self.gamma

    def encoder(self, adj_matrix, node_feats):
        H = node_feats
        H = F.elu(self.encoder_layer1(H, adj_matrix))
        Z = self.encoder_layer2(H)
        return Z

    def get_basis(self):
        basis = F.softmax(self.factors, dim=1)
        return basis

    def deconvolutioner(self, Z, slice_label_emb):
        beta = self.deconv_beta_layer(F.elu(Z))
        beta = F.softmax(beta, dim=1)
        H = F.elu(torch.cat((Z, slice_label_emb), axis=1))
        alpha = self.deconv_alpha_layer(H)
        return beta, alpha

    def decoder(self, adj_matrix, Z, slice_label_emb):
        H = torch.cat((Z, slice_label_emb), axis=1)
        H = F.elu(self.decoder_layer1(H, adj_matrix))
        X_recon = self.decoder_layer2(H)
        return X_recon



class IntegrationNet_LGCN(nn.Module):

    def __init__(self, 
                 hidden_dims, # dimensionality of hidden layers
                 n_features_lgcn, # number of features for LGCN layer
                 n_hvgs, # number of shared hvgs among datasets
                 n_factors, # number of topics in biologically interpretable dimension reduction
                 n_slices, # number of slices
                 slice_emb_dim, # dimensionality of slice id embedding
                 different_platforms=False, # whether integrate datasets across different platforms
                 ):

        super().__init__()

        # mode
        self.different_platforms = different_platforms
        
        # define layers
        # encoder layers
        self.encoder_layer1 = DenseLayer(n_features_lgcn, hidden_dims[0])
        self.encoder_layer2 = DenseLayer(hidden_dims[0], hidden_dims[1])
        # decoder layers
        self.decoder_layer1 = DenseLayer(hidden_dims[1]+slice_emb_dim, hidden_dims[0])
        self.decoder_layer2 = DenseLayer(hidden_dims[0], n_hvgs)
        # deconvolution layers
        if self.different_platforms == False:
            self.deconv_beta_layer = DenseLayer(hidden_dims[1], n_factors) # initialize
        else:
            self.deconv_beta_layer = DenseLayer(hidden_dims[1]+slice_emb_dim, n_factors) # initialize
        self.factors = nn.Parameter(torch.Tensor(n_factors, n_hvgs).uniform_(-np.sqrt(6/(n_factors+n_hvgs)), np.sqrt(6/(n_factors+n_hvgs))))
        
        self.deconv_alpha_layer = DenseLayer(hidden_dims[1]+slice_emb_dim, 1, zero_init=True)
        self.gamma = nn.Parameter(torch.Tensor(n_slices, n_hvgs).zero_())

        self.n_slices = n_slices
        self.slice_emb = nn.Embedding(self.n_slices, slice_emb_dim)

    def forward(self, 
                node_feats_dict, # dict of input node features
                slice_label_dict, # dict of slice labels
                ):
        # create dicts
        Z_dict = {}
        slice_label_emb_dict = {}
        beta_dict = {}
        alpha_dict = {}
        node_feats_recon_dict = {}

        basis = self.get_basis()

        for i in range(self.n_slices):
            # encoder
            Z_dict[i] = self.encoder(node_feats_dict[i])

            # deconvolutioner
            slice_label_emb_dict[i] = self.slice_emb(slice_label_dict[i])
            beta_dict[i], alpha_dict[i] = self.deconvolutioner(Z_dict[i], slice_label_emb_dict[i])
            alpha_dict[i] = torch.clamp(alpha_dict[i], -0.1, 0.1)

            # decoder
            node_feats_recon_dict[i] = self.decoder(Z_dict[i], slice_label_emb_dict[i])

        return Z_dict, beta_dict, alpha_dict, node_feats_recon_dict, basis, self.gamma

    def encoder(self, node_feats):
        H = node_feats
        H = F.elu(self.encoder_layer1(H))
        Z = self.encoder_layer2(H)
        return Z

    def get_basis(self):
        basis = F.softmax(self.factors, dim=1)
        return basis

    def deconvolutioner(self, Z, slice_label_emb):
        if self.different_platforms == False:
            beta = self.deconv_beta_layer(F.elu(Z))
        else:
            beta = self.deconv_beta_layer(F.elu(torch.cat((Z, slice_label_emb), axis=1)))
        beta = F.softmax(beta, dim=1)
        H = F.elu(torch.cat((Z, slice_label_emb), axis=1))
        alpha = self.deconv_alpha_layer(H)
        return beta, alpha

    def decoder(self, Z, slice_label_emb):
        H = torch.cat((Z, slice_label_emb), axis=1)
        H = F.elu(self.decoder_layer1(H))
        X_recon = self.decoder_layer2(H)
        return X_recon


