import copy
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions.beta import Beta
import torch_geometric.nn as gnn
import torch_geometric

device = 'cuda' if torch.cuda.is_available() else 'cpu'
radius = 10

class psudo_hetero_transformer(nn.Module):
    """
    A pseudo-heterogeneous graph transformer layer using TransformerConv.
    It applies a separate TransformerConv for each predefined edge type in a protein complex
    (e.g., TCR-TCR, TCR-peptide, peptide-MHC). The outputs from these different
    message-passing operations are then averaged.
    """
    def __init__(self,in_dim,out_dim,edge_attr_dim,drop_p):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.edge_attr_dim = edge_attr_dim
        self.drop_p = drop_p
        super(psudo_hetero_transformer,self).__init__()

        self.TCR2TCR = gnn.conv.TransformerConv(self.in_dim,self.out_dim,edge_dim=self.edge_attr_dim,dropout=self.drop_p)
        self.TCR2HLA = gnn.conv.TransformerConv(self.in_dim,self.out_dim,edge_dim=self.edge_attr_dim,dropout=self.drop_p)
        self.TCR2PEP = gnn.conv.TransformerConv(self.in_dim,self.out_dim,edge_dim=self.edge_attr_dim,dropout=self.drop_p)

        self.HLA2TCR = gnn.conv.TransformerConv(self.in_dim,self.out_dim,edge_dim=self.edge_attr_dim,dropout=self.drop_p)
        self.HLA2HLA = gnn.conv.TransformerConv(self.in_dim,self.out_dim,edge_dim=self.edge_attr_dim,dropout=self.drop_p)
        self.HLA2PEP = gnn.conv.TransformerConv(self.in_dim,self.out_dim,edge_dim=self.edge_attr_dim,dropout=self.drop_p)

        self.PEP2TCR = gnn.conv.TransformerConv(self.in_dim,self.out_dim,edge_dim=self.edge_attr_dim,dropout=self.drop_p)
        self.PEP2HLA = gnn.conv.TransformerConv(self.in_dim,self.out_dim,edge_dim=self.edge_attr_dim,dropout=self.drop_p)
        self.PEP2PEP = gnn.conv.TransformerConv(self.in_dim,self.out_dim,edge_dim=self.edge_attr_dim,dropout=self.drop_p)

    def forward(self,x,edge_index_dict,edge_attrs_dict,batch):

        x1 = self.TCR2TCR(x,edge_index_dict['tcr2tcr_index'],edge_attrs_dict['tcr2tcr_attr'])
        x2 = self.TCR2HLA(x,edge_index_dict['tcr2hla_index'],edge_attrs_dict['tcr2hla_attr'])
        x3 = self.TCR2PEP(x,edge_index_dict['tcr2pep_index'],edge_attrs_dict['tcr2pep_attr'])

        x4 = self.HLA2TCR(x,edge_index_dict['hla2tcr_index'],edge_attrs_dict['hla2tcr_attr'])
        x5 = self.HLA2HLA(x,edge_index_dict['hla2hla_index'],edge_attrs_dict['hla2hla_attr'])
        x6 = self.HLA2PEP(x,edge_index_dict['hla2pep_index'],edge_attrs_dict['hla2pep_attr'])

        x7 = self.PEP2TCR(x,edge_index_dict['pep2tcr_index'],edge_attrs_dict['pep2tcr_attr'])
        x8 = self.PEP2HLA(x,edge_index_dict['pep2hla_index'],edge_attrs_dict['pep2hla_attr'])
        x9 = self.PEP2PEP(x,edge_index_dict['pep2pep_index'],edge_attrs_dict['pep2pep_attr'])

        x = (x1+x2+x3+x4+x5+x6+x7+x8+x9)/9
        return x

# STAG model
class STAG(nn.Module):
    """
    STAG (Structural TCR And pMHC binding specificity prediction Graph neural network) model.
    This module processes graph representations of protein complexes.
    In this work we utilize pseudo-heterogeneous transformer blocks for our convolutions
    """
    def __init__(self):
        self.node_feats_in = 320
        self.edge_feats_in = 15
        self.edge_hidden = 32
        self.node_hidden = 320
        self.drop_p = 0.125
        self.conv_type = 'trans'
        self.num_conv = 3
        self.intermediate_MLPs = False

        super(STAG,self).__init__()

        self.global_pool = gnn.global_max_pool
        self.edge_mlp1_lin1 = nn.Linear((self.node_hidden*2)+self.edge_feats_in,self.edge_hidden)
        self.edge_mlp1_lin2 = nn.Linear(self.edge_hidden,self.edge_hidden)
        self.conv_block1 = psudo_hetero_transformer(self.node_hidden,self.node_hidden,self.edge_hidden,self.drop_p)
        self.gnn_norm1 = gnn.norm.LayerNorm(self.node_hidden)
        self.conv_block2 = psudo_hetero_transformer(self.node_hidden,self.node_hidden,self.edge_hidden,self.drop_p)
        self.gnn_norm2 = gnn.norm.LayerNorm(self.node_hidden)
        self.conv_block3 = psudo_hetero_transformer(self.node_hidden,self.node_hidden,self.edge_hidden,self.drop_p)
        self.gnn_norm3 = gnn.norm.LayerNorm(self.node_hidden)

    def forward(self,x,edge_index,edge_index_dict,edge_attr,edge_attrs_dict,batch):

        for edge_type in ['tcr2tcr','hla2tcr','pep2tcr',   'tcr2hla','hla2hla','pep2hla',   'tcr2pep','hla2pep','pep2pep']:
            row, col = edge_index_dict[edge_type+'_index']
            temp_edge_attr = torch.cat([x[row], x[col], edge_attrs_dict[edge_type+'_attr']], dim=-1)
            temp_edge_attr = self.edge_mlp1_lin1(temp_edge_attr)
            temp_edge_attr = F.leaky_relu(temp_edge_attr)
            temp_edge_attr = F.dropout(temp_edge_attr,p=self.drop_p,training=self.training)
            temp_edge_attr = self.edge_mlp1_lin2(temp_edge_attr)
            edge_attrs_dict[edge_type+'_attr'] = temp_edge_attr
        x_ = self.conv_block1(x,edge_index_dict,edge_attrs_dict,batch)
        x_ = F.dropout(x_,p=self.drop_p,training=self.training)
        x_ = F.leaky_relu(x_)
        x_ = self.gnn_norm1(x_,batch)
        x = 0.5*(x+x_)
        x_ = self.conv_block2(x,edge_index_dict,edge_attrs_dict,batch)
        x_ = F.dropout(x_,p=self.drop_p,training=self.training)
        x_ = F.leaky_relu(x_)
        x_ = self.gnn_norm2(x_,batch)
        x = 0.5*(x+x_)
        x_ = self.conv_block3(x,edge_index_dict,edge_attrs_dict,batch)
        x_ = F.dropout(x_,p=self.drop_p,training=self.training)
        x_ = F.leaky_relu(x_)
        x_ = self.gnn_norm3(x_,batch)
        x = 0.5*(x+x_)
        #global
        global_attr = self.global_pool(x,batch)

        return global_attr

# full model
class LLM_transfer(nn.Module):
    """
    The full STAG-LLM model combining a pre-trained LLM (ESM-2 encoder)
    with a STAG graph neural network. It can take sequence-only input,
    graph-only input, or a combination of both.
    """
    def __init__(self,seq_LLM,embedding):
        super (LLM_transfer,self).__init__()

        embedding_size = 320
        embedding_layer = 6

        self.embedding_size = embedding_size
        self.embedding_layer = embedding_layer

        fc_hidden_size = 320

        self.fc_hidden_size = fc_hidden_size

        self.seq_LLM = copy.deepcopy(seq_LLM)
        self.embedding = embedding

        self.bn1 = nn.BatchNorm1d(fc_hidden_size)
        self.bn2 = nn.BatchNorm1d(fc_hidden_size)

        self.fc1 = nn.Linear(embedding_size,fc_hidden_size)
        self.fc2 = nn.Linear(fc_hidden_size,fc_hidden_size)
        self.fc3 = nn.Linear(fc_hidden_size,1)

        self.stag = STAG()

        self.tuning_LLM=False
        self.drop_p = 0.4
        self.beta = Beta(torch.tensor([0.25]), torch.tensor([0.25]))

    def forward(self,seq_toks,graphs,lens,batch_size,seq_tune,pHLA_binding,manifold_mixup):
        if seq_toks is not None:
            with torch.no_grad():
                seq_rep = self.embedding(seq_toks)

            if not self.tuning_LLM:
                with torch.no_grad():
                    seq_embed = self.seq_LLM(seq_rep)[0]
            else:
                seq_embed = self.seq_LLM(seq_rep)[0]
            seq_rep = seq_embed[:,0,:]

        else:
            seq_rep = torch.zeros((batch_size,self.embedding_size)).to(device)

        if graphs is not None:
            for i,graph in enumerate(graphs):
                g_seq_embed = seq_embed[i]
                sep_mask = torch.tensor(seq_toks[i].detach().to('cpu').numpy()!=29).to(device)
                g_seq_embed = g_seq_embed[sep_mask,:]
                g_seq_embed = g_seq_embed[1:lens[i]-1]

                graph[2] = g_seq_embed.cpu()

            batch = [torch_geometric.data.Data(x=graph[2],edge_index=graph[3],edge_attr=graph[4],edge_index_dict=graph[5],edge_attrs_dict=graph[6],y=graph[7]) for graph in graphs]
            batch = torch_geometric.loader.DataLoader(batch,batch_size=batch_size,shuffle=False)

            for data in batch:
                graph_rep = data.to(device)
                graph_rep = self.stag(graph_rep.x,graph_rep.edge_index,graph_rep.edge_index_dict,graph_rep.edge_attr,graph_rep.edge_attrs_dict,graph_rep.batch)

        else:
            graph_rep = torch.zeros((batch_size,self.embedding_size)).to(device)

        if graphs is None:
            out = seq_rep
        elif seq_tune is None:
            out = graph_rep
        else:
            out = 0.5*(seq_rep+graph_rep)

        out_ = self.fc1(out)
        out_ = F.dropout(out_,p=self.drop_p,training=self.training)
        out_ = F.leaky_relu(out_)
        out_ = self.bn1(out_)
        out = 0.5*(out+out_)
        out_ = self.fc2(out)
        out_ = F.dropout(out_,p=self.drop_p,training=self.training)
        out_ = F.leaky_relu(out_)
        out_ = self.bn2(out_)
        out = 0.5*(out+out_)
        out = self.fc3(out)

        return out.squeeze()
