import pickle
import os
import warnings
import numpy as np
import pandas as pd
import ipyparallel as ipp

from utils import *
from data_handling import *

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
warnings.filterwarnings('ignore')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
radius = 10

"""
prep file to create hetero_graphs
"""

# load df
structure_df = pd.read_csv('data/final_dataset_modeled.csv')
structure_df['full_seq'] = structure_df['TCR_A_sequence'] + '.' + structure_df['TCR_B_sequence'] + '.' + structure_df['peptide'] + '.' + structure_df['MHC Sequence'] + '.'
structure_df['structure_name'] = structure_df['peptide'] + '_' + structure_df['CDR3a'] + '_' + structure_df['CDR3b']
# remove errant structures (eg. TCRs docked to the underside of HLAs...)
failed_models = ['ILMHATYFL_CALSDLDTPLVF_CASSPRGGVYNEQFF','FLDCKSYIL_CAESRGRDDKIIF_CASSSGGELFF','KLYPFLWFA_CAARRGAGTALIF_CASSGITNVLTF','KLYPFLWFA_CVPHRGGTALIF_CSAARGGEQFF','ILHTHVPEV_CASHGGFQKLVF_CSARPYMEAKNIQYF','TVYPYGTSL_CVPHRGGTALIF_CSAARGGEQFF','TVYPYGTSL_CAARRGAGTALIF_CASSGITNVLTF','GENALTYAL_CAFIPGVNAGNMLTF_CSARGLRDEHNEQFF','YENGSTPVL_CATGNQGGKLIF_CASTGQGGNTIYF','SEESAFYVL_CATGNQGGKLIF_CASTGQGGNTIYF','ILHTHVPEV_CAVSAMNTGNQFYF_CSAAQGDSQETQYF','ILMHATYFL_CVPHRGGTALIF_CSAARGGEQFF','TVYPYGTSL_CAVISPNSGYALNF_CASSSRRLAASYNEQFF','ILHTHVPEV_CAAGHGGATNKLIF_CASSGVGGKTQYF','YESYIPGAL_CATGNQGGKLIF_CASTGQGGNTIYF','ILHTHVPEV_CAASALGAGSYQLTF_CASSSGVWRAGEQFF','YESYIPGAL_CAVGALRNARLMF_CASSSKVDQETQYF','KLYPFLWFA_CAVISPNSGYALNF_CASSSRRLAASYNEQFF','SQFNWTIYL_CVPHRGGTALIF_CSAARGGEQFF','ILHTHVPEV_CVPHRGGTALIF_CSAARGGEQFF','TVYPYGTSL_CAVGGEAGTASKLTF_CASSPGSGSTQYF','GENALTYAL_CAASPNRGSTLGRLYF_CSARVVTLNEQFF','MTDYDYLEV_CAVGGEAGTASKLTF_CASSPGSGSTQYF']
structure_df = structure_df[~structure_df['structure_name'].isin(failed_models)]

# get hetero_graphs from all pdbs referenced by df
iii = np.arange(structure_df.shape[0])
def get_graph_from_df(i):
    if os.path.exists('hetero_edge_graphs/'+structure_df.iloc[i,structure_df.columns.get_loc('structure_name')]):
        return None

    structure = 'data/top_structures/'+structure_df.iloc[i,structure_df.columns.get_loc('structure_name')]+'.pdb'
    graph = get_graph(structure,structure_df.iloc[i,structure_df.columns.get_loc('structure_name')],structure_df.iloc[i,structure_df.columns.get_loc('label')])
    with open('hetero_edge_graphs/'+structure_df.iloc[i,structure_df.columns.get_loc('structure_name')], 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(graph, outp, pickle.HIGHEST_PROTOCOL)
    return graph

with ipp.Cluster(n=6) as rc:
    dview = rc[:]
    print('dview: ',dview)
    dview.execute("""
    import numpy as np
    import pandas as pd
    import torch
    import os
    from Bio.PDB import PDBIO, PDBParser
    import torch_geometric.nn as gnn
    import pickle
    """)
    dview["structure_df"] = structure_df
    dview["get_graph"] = get_graph
    dview["get_edge_type"] = get_edge_type
    dview["dic"] = dic
    dview["radius"] = radius
    print([i["get_graph"] for i in rc])
    view = rc.load_balanced_view()
    asyncresult = view.map_async(get_graph_from_df,iii)
    asyncresult.wait_interactive()
    graphs = asyncresult.get()
