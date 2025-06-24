import torch
import esm
import numpy as np
import pandas as pd
import argparse
import os
import warnings
from transformers import AutoTokenizer, EsmForMaskedLM

# Import custom modules
from model import *
from data_handling import *

# Suppress warnings
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings('ignore')

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def predict_binding(model_path, pdb_file_path):
    """
    Loads a pretrained STAG-LLM model and makes a binding prediction for a given PDB file.

    Args:
        model_path (str): Path to the pretrained model's state_dict (.pt file).
        pdb_file_path (str): Path to the input PDB file for prediction.

    Returns:
        float: The predicted binding score (logit).
    """
    print(f"Loading model from: {model_path}")
    print(f"Processing PDB file: {pdb_file_path}")

    # --- 1. Initialize LLM components (ESM-2) ---
    model_checkpoint = "facebook/esm2_t6_8M_UR50D"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    # Initialize a dummy ESMForMaskedLM to get the encoder and embedding layer
    # This is a workaround as we only need the architecture, not necessarily a fully trained MLM
    temp_esm_model = EsmForMaskedLM.from_pretrained(model_checkpoint)
    seq_LLM_encoder = temp_esm_model.esm.encoder
    input_embeddings = temp_esm_model.get_input_embeddings()

    # --- 2. Initialize and Load STAG-LLM Model ---
    model = LLM_transfer(seq_LLM_encoder, input_embeddings)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval() # Set model to evaluation mode

    # --- 3. Prepare Input Data from PDB ---
    # The 'get_graph' function expects a name and label. For prediction, label can be dummy.
    # The name should be unique and descriptive, perhaps derived from the filename.
    pdb_name = os.path.splitext(os.path.basename(pdb_file_path))[0]

    # Create a dummy label as it's not used for prediction
    dummy_label = 0

    # Generate graph data from the PDB file
    # Ensure get_graph handles the parameters correctly according to its definition in data_handling.py
    # get_graph returns: [name, full_seq, None, edges, edge_attrs, edge_index_dict, edge_attrs_dict, label]
    try:
        graph_data_tuple = get_graph(pdb_file_path, pdb_name, dummy_label)
    except Exception as e:
        print(f"Error generating graph from PDB file: {e}")
        return None

    # Extract full sequence and tokenize
    full_sequence = graph_data_tuple[1]
    tokenized_sequence = tokenizer(full_sequence, return_tensors="pt").input_ids.to(device)

    # Get sequence length for the model's forward pass
    sequence_length = [len(full_sequence)] # Expects a list for batching

    # Prepare graph data for the model's forward pass
    # It expects a list of graph_info tuples, even for a single sample.
    graphs = [graph_data_tuple] # Wrap in a list for batch compatibility

    # --- 4. Make Prediction ---
    with torch.no_grad():
        # The forward method signature of LLM_transfer is:
        # forward(self, seq_toks, graphs, lens, batch_size, seq_tune, pHLA_binding, manifold_mixup)
        # For prediction, we likely want to use both sequence and graph, so seq_tune=True.
        # pHLA_binding and manifold_mixup are typically False for inference.
        prediction_logit = model(
            seq_toks=tokenized_sequence,
            graphs=graphs, # Pass the list of graph data
            lens=sequence_length,
            batch_size=1, # Single sample prediction
            seq_tune=True,
            pHLA_binding=False,
            manifold_mixup=False
        )

    # Convert logit to probability if desired (sigmoid for BCEWithLogitsLoss)
    prediction_probability = torch.sigmoid(prediction_logit).item()

    return prediction_probability

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict TCR-pMHC binding using a pretrained STAG-LLM model.")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the pretrained STAG-LLM model's state_dict (.pt file).")
    parser.add_argument("--pdb_file", type=str, required=True,
                        help="Path to the input PDB file for which to make a prediction.")

    args = parser.parse_args()

    # Call the prediction function
    predicted_score = predict_binding(args.model_path, args.pdb_file)

    if predicted_score is not None:
        print(f"\nPrediction for {args.pdb_file}: {predicted_score:.4f}")
        # You can add a threshold here to classify as binding/non-binding, e.g.:
        # if predicted_score > 0.5:
        #     print("Predicted: Binding")
        # else:
        #     print("Predicted: Non-binding")
