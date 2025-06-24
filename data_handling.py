import pickle
import torch
import torch_geometric.nn as gnn
import numpy as np
from torch.utils.data import Dataset
from Bio.PDB import PDBParser

from utils import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
radius = 10

def get_edge_type(x,y,chains):
    """
    Determines the type of interaction between two residues based on their chain IDs.

    Args:
        x (str): Chain ID of the first residue.
        y (str): Chain ID of the second residue.
        chains (str): A string containing all chain IDs in the order they appear
                      (e.g., 'ABCD' where A and B are TCR, C is peptide, D is MHC).

    Returns:
        str: A string representing the edge type (e.g., 'tcr2tcr', 'tcr2pep', 'pep2hla').
    """
    if x == chains[0]:
        if y == chains[0]:
            return 'tcr2tcr'
        if y == chains[1]:
            return 'tcr2tcr'
        if y == chains[2]:
            return 'tcr2pep'
        if y == chains[3]:
            return 'tcr2hla'
    if x == chains[1]:
        if y == chains[0]:
            return 'tcr2tcr'
        if y == chains[1]:
            return 'tcr2tcr'
        if y == chains[2]:
            return 'tcr2pep'
        if y == chains[3]:
            return 'tcr2hla'

    if x == chains[2]:
        if y == chains[0]:
            return 'pep2tcr'
        if y == chains[1]:
            return 'pep2tcr'
        if y == chains[2]:
            return 'pep2pep'
        if y == chains[3]:
            return 'pep2hla'

    if x == chains[3]:
        if y == chains[0]:
            return 'hla2tcr'
        if y == chains[1]:
            return 'hla2tcr'
        if y == chains[2]:
            return 'hla2pep'
        if y == chains[3]:
            return 'hla2hla'
    return 'unknown' # Should ideally not be reached

def get_graph(pdb,name,label):
    """
    Parses a PDB file to extract protein structure information and construct a graph.

    Args:
        pdb_path (str): Path to the PDB file.
        name (str): Name associated with the structure (e.g., structure_name from dataframe).
        label (int): The label associated with the interaction (e.g., binding/non-binding).

    Returns:
        list: A list containing graph information: [name, full_sequence, node_features,
              all_edges, all_edge_attributes, dictionary_of_typed_edges,
              dictionary_of_typed_edge_attributes, label].
              Node features are initially None as they are later obtained from LLM.
    """
    full_seq = ''
    label = torch.tensor(label)
    nodes = [] # Stores (chain_id, C-alpha_coordinate)
    sep_indecies = [] # Stores indices where chains separate in the full_seq
    edge_attrs = [] # Stores RBF-encoded distances for all edges
    edge_index_dict = { # Stores edge indices for different interaction types
        'tcr2tcr_index': [], 'tcr2hla_index': [], 'tcr2pep_index': [],
        'hla2tcr_index': [], 'hla2hla_index': [], 'hla2pep_index': [],
        'pep2tcr_index': [], 'pep2hla_index': [], 'pep2pep_index': []
    }
    edge_attrs_dict = { # Stores edge attributes for different interaction types
        'tcr2tcr_attr': [], 'tcr2hla_attr': [], 'tcr2pep_attr': [],
        'hla2tcr_attr': [], 'hla2hla_attr': [], 'hla2pep_attr': [],
        'pep2tcr_attr': [], 'pep2hla_attr': [], 'pep2pep_attr': []
    }

    parser = PDBParser()
    try:
        structure = parser.get_structure('roi',pdb)
    except Exception as e:
        print('Error parsing PDB: ',pdb)
        print(e)
        structure = parser.get_structure('roi',pdb[:-5]+'1.pdb')
    chains = ''
    for chain in structure[0]:
        chains = chains+chain.id
    i = 0
    for chain in structure[0]:
        for res in chain:
            for atom in res.get_atoms():
                if atom.get_name() in ['CA']:
                    nodes.append((chain.id,np.array(atom.get_coord())))
            full_seq += dic[res.get_resname()]
            i += 1
        full_seq += '.'
        sep_indecies.append(i)
        i += 1

    coords = torch.from_numpy(np.array([node[1] for node in nodes]).astype(int))
    edges = gnn.radius_graph(coords,r=radius,loop=False)

    for i,(src, dst) in enumerate(edges.T):
        d = np.linalg.norm(nodes[src][1] - nodes[dst][1])
        length_scale_list = [1.5 ** x for x in range(15)]
        rbf_dists_ij = np.exp(-np.array([(d**2)/ls for ls in length_scale_list]))
        edge_attrs.append(rbf_dists_ij)
        edge_type = get_edge_type(nodes[src][0],nodes[dst][0],chains)
        edge_attrs_dict[edge_type+'_attr'].append(rbf_dists_ij)
        edge_index_dict[edge_type+'_index'].append([src,dst])

    edge_attrs = torch.from_numpy(np.array(edge_attrs)).float()
    edge_index_dict['tcr2tcr_index'] = torch.from_numpy(np.array(edge_index_dict['tcr2tcr_index'])).T
    edge_index_dict['tcr2hla_index'] = torch.from_numpy(np.array(edge_index_dict['tcr2hla_index'])).T
    edge_index_dict['tcr2pep_index'] = torch.from_numpy(np.array(edge_index_dict['tcr2pep_index'])).T
    edge_index_dict['hla2tcr_index'] = torch.from_numpy(np.array(edge_index_dict['hla2tcr_index'])).T
    edge_index_dict['hla2hla_index'] = torch.from_numpy(np.array(edge_index_dict['hla2hla_index'])).T
    edge_index_dict['hla2pep_index'] = torch.from_numpy(np.array(edge_index_dict['hla2pep_index'])).T
    edge_index_dict['pep2tcr_index'] = torch.from_numpy(np.array(edge_index_dict['pep2tcr_index'])).T
    edge_index_dict['pep2hla_index'] = torch.from_numpy(np.array(edge_index_dict['pep2hla_index'])).T
    edge_index_dict['pep2pep_index'] = torch.from_numpy(np.array(edge_index_dict['pep2pep_index'])).T

    edge_attrs_dict['tcr2tcr_attr'] = torch.from_numpy(np.array(edge_attrs_dict['tcr2tcr_attr'])).float()
    edge_attrs_dict['tcr2hla_attr'] = torch.from_numpy(np.array(edge_attrs_dict['tcr2hla_attr'])).float()
    edge_attrs_dict['tcr2pep_attr'] = torch.from_numpy(np.array(edge_attrs_dict['tcr2pep_attr'])).float()
    edge_attrs_dict['hla2tcr_attr'] = torch.from_numpy(np.array(edge_attrs_dict['hla2tcr_attr'])).float()
    edge_attrs_dict['hla2hla_attr'] = torch.from_numpy(np.array(edge_attrs_dict['hla2hla_attr'])).float()
    edge_attrs_dict['hla2pep_attr'] = torch.from_numpy(np.array(edge_attrs_dict['hla2pep_attr'])).float()
    edge_attrs_dict['pep2tcr_attr'] = torch.from_numpy(np.array(edge_attrs_dict['pep2tcr_attr'])).float()
    edge_attrs_dict['pep2hla_attr'] = torch.from_numpy(np.array(edge_attrs_dict['pep2hla_attr'])).float()
    edge_attrs_dict['pep2pep_attr'] = torch.from_numpy(np.array(edge_attrs_dict['pep2pep_attr'])).float()

    return [name,full_seq,None,edges,edge_attrs,edge_index_dict,edge_attrs_dict,label]

def pad_collate(batch):
    """
    Custom collate function for DataLoader to handle padding of sequences and batching of graphs.

    Args:
        batch (list): A list of samples, where each sample is a tuple
                      (graph_info, sequence_tokens, label, id).

    Returns:
        tuple: Batched graph information, padded sequence tokens, sequence lengths, labels, and IDs.
    """
    (gg,ss,yy,ii) = zip(*batch)

    lens = [len(x) for x in ss]
    ss = torch.nn.utils.rnn.pad_sequence(ss, batch_first=True, padding_value=1)

    return gg,ss,lens,yy,ii

class TCRpHLA_dataset(Dataset):
    """
    PyTorch Dataset for TCR-pMHC interaction data, handling graph and sequence data.
    Loads pre-processed graph data and tokenized sequences.
    """
    def __init__(self,dataframe):
        self.dataframe = dataframe
        self.dataframe.reset_index()
    def __len__(self):
        return self.dataframe.shape[0]
    def __getitem__(self,i):
        graph = pickle.load(open('hetero_edge_graphs/'+self.dataframe['structure_name'].iloc[i],'rb'))
        seq_toks = self.dataframe['seq_tok'].iloc[i]
        label =  self.dataframe['label'].iloc[i]
        id = self.dataframe['structure_name'].iloc[i]
        return graph, seq_toks, label, id
