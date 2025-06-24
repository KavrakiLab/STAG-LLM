import os
import warnings
import torch
import esm
from torch import nn
from torch.distributions.beta import Beta
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, EsmForMaskedLM, DataCollatorForLanguageModeling, Seq2SeqTrainingArguments, Trainer
import datasets as hf_datasets

from model import *
from data_handling import *
from utils import *

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings('ignore')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
radius = 10

print("Pretraining LLM for masked token prediction")

# Load ESM model and tokenizer
model_checkpoint = "facebook/esm2_t6_8M_UR50D"
model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
batch_converter = alphabet.get_batch_converter()

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = EsmForMaskedLM.from_pretrained(model_checkpoint)
model = model.to(device)

# Load and prepare sequence data for pretraining
sequence_df = pd.read_csv('data/full_seq_df_new.csv')
full_seq_df = pd.read_csv('data/full_seq_df_new.csv')
sequence_df = pd.concat([sequence_df,full_seq_df])
sequence_df = sequence_df[sequence_df['label'] == 1]
sequence_df['full_seq'] = sequence_df['TCR_A_sequence'] + '.' + sequence_df['TCR_B_sequence'] + '.' + sequence_df['peptide'] + '.' + sequence_df['MHC Sequence'] + '.'

# Split data and tokenize
sequences = list(sequence_df['full_seq'])
train_sequences, test_sequences= train_test_split(sequences, test_size=0.25, shuffle=True)
train_tokenized = tokenizer(train_sequences)
test_tokenized = tokenizer(test_sequences)
# Create Hugging Face datasets
train_dataset = hf_datasets.Dataset.from_dict(train_tokenized)
test_dataset = hf_datasets.Dataset.from_dict(test_tokenized)

# Configure data collator and training arguments for MLM
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
training_args = Seq2SeqTrainingArguments(
    output_dir="my_esm_tcr_mlm_model",
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=2e-5,
    predict_with_generate = True,
    torch_empty_cache_steps = 1,
    gradient_accumulation_steps=4,
    num_train_epochs=5,
    weight_decay=0.01,
    push_to_hub=False,
    load_best_model_at_end=True,
)
# Initialize and train the Trainer for MLM
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()

print("Preparing for STAG-LLM training")

# Load and prepare structure data for STAG-LLM training
structure_df = pd.read_csv('data/final_dataset_modeled.csv')
structure_df['structure_name'] = structure_df['peptide']+'_'+structure_df['CDR3a']+'_'+structure_df['CDR3b']
structure_df.head()

# Get input embeddings and ESM encoder from the pre-trained model
input_embeeddings = model.get_input_embeddings()
model_esm = model.esm.encoder
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Create full sequence column and get unique sequences
structure_df['full_seq'] = structure_df['TCR_A_sequence'] + '.' + structure_df['TCR_B_sequence'] + '.' + structure_df['peptide'] + '.' + structure_df['MHC Sequence'] + '.'
seqs = list(set(structure_df['full_seq']))

# Tokenize sequences
toks = {}
for seq in tqdm(seqs):
    toks[seq] = tokenizer(seq)

structure_df['seq_tok'] = structure_df['full_seq'].apply(lambda x: torch.tensor(toks[x]['input_ids']))

del toks # Clear up memory

# Setup output directory for results
out_dir = 'test'
job_dir = False
while not job_dir:
    job_name = np.random.choice(adjs)+'-'+np.random.choice(dino_names)
    if not os.path.exists(out_dir+'/'+job_name):
        os.mkdir(out_dir+'/'+job_name)
        job_dir = True

def get_lr(optimizer_):
    for param_group in optimizer_.param_groups:
        return param_group['lr']

# Initialize the STAG-LLM model
my_model = LLM_transfer(model_esm,input_embeeddings)
my_model = my_model.to(device)

# Training parameters
bs = 50 # Batch size
scheduler_free_epochs = 5
graph_epochs = 10
llm_epochs = 15

# Optimizer and Loss function
optimizer = torch.optim.AdamW(my_model.parameters(), lr=0.0001, weight_decay=0.01)
s_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1-np.mean(structure_df['label']))/np.mean(structure_df['label'])))
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,mode='max',patience=0,threshold=1e-3,threshold_mode='rel',factor=0.5)

with open(out_dir+'/'+job_name+'/status.txt','a+') as f:
    f.write('graph dataset only ---- \n')

# set random seed for fair model comparison
# used the following seeds for experiments in paper
# 1,\ 2,\ 4,\ 8,\ 16,\ 32,\ 64,\ 128,\ 256,\ 512,\ 1024,\ 2048,\ 4096,\ 8192,\ 16384
np.random.seed(16384)

# Create train, validation, and test splits based on 'fold'
structure_df['fold'] = np.random.randint(0,5,size=structure_df.shape[0])
test_df = structure_df[structure_df['fold'] == 0]
val_df = structure_df[structure_df['fold'] == 1]
train_df = structure_df[structure_df['fold'] != 0]
train_df = train_df[train_df['fold'] != 1]

# train_df = train_df.sample(frac=0.75)
# uncomment to train with 75% of data
# change "frac" to test different training sizes

# Create datasets and data loaders
test_data = TCRpHLA_dataset(test_df)
val_data = TCRpHLA_dataset(val_df)
train_data = TCRpHLA_dataset(train_df)

test_loader = DataLoader(test_data,batch_size=bs,shuffle=False,drop_last=False,collate_fn=pad_collate)
val_loader = DataLoader(val_data,batch_size=bs,shuffle=False,drop_last=False,collate_fn=pad_collate)
train_loader = DataLoader(train_data,batch_size=bs,shuffle=True,drop_last=False,collate_fn=pad_collate)

best_pr = 0

print("Starting STAG-LLM training")
print('JOB NAME: ',job_name)

# Main training loop for STAG-LLM
for epoch in range(graph_epochs+llm_epochs):
    my_model.tuning_LLM = False

    total_loss = 0

    if epoch == scheduler_free_epochs:
        optimizer = torch.optim.AdamW(my_model.parameters(), lr=0.00004, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,mode='max',patience=0,threshold=1e-3,threshold_mode='rel',factor=0.5)
        torch.cuda.empty_cache()

    if epoch >= graph_epochs:
        print('Tuning LLM too!')
        my_model.tuning_LLM = True

    print('LR: ',get_lr(optimizer))

    my_model.train()
    outs = []
    ys = []
    train_mode = 0# Controls which branch of the model is used (seq, structure, or combined) # optional include peptide2HLA binding prediction or mixup augmentation
    for graphs, seq_toks, lens, labels, indicies in tqdm(train_loader):
        seq_toks = seq_toks.to(device)
        if train_mode != 2:
            out = my_model(seq_toks,graphs,lens,bs,True,False,False)
        else:
            out = my_model(seq_toks,None,lens,bs,True,False,False)

        train_mode = (train_mode + 1) % 3 # Cycle train_mode 0, 1, 2

        y = torch.tensor(labels).float().to(device)
        loss = s_criterion(out,y)
        loss.backward()
        total_loss += loss.item()

        optimizer.step()
        optimizer.zero_grad()

        outs = outs + out.detach().to('cpu').numpy().tolist()
        ys = ys + list(labels)
    roc_train = roc_auc_score(ys,outs)
    pr_train = average_precision_score(ys,outs)
    print(f'train-ROC: {roc_train:.4f}')
    print(f'train-PR: {pr_train:.4f}')

    my_model.eval()
    with torch.no_grad():
        outs = []
        ys = []

        # Validation loop
        for graphs, seq_toks, lens, labels, indicies in tqdm(val_loader):
            seq_toks = seq_toks.to(device)
            out = my_model(seq_toks,graphs,lens,bs,True,False,False)

            outs = outs + out.detach().to('cpu').numpy().tolist()
            ys = ys + list(labels)

        pr = average_precision_score(ys,outs)
        scheduler.step(pr)
        # If current PR is best, evaluate on test set and save model
        # Scores from the last test run are reported in "final_preds" df
        if pr > best_pr:
            best_pr = pr

            test_outs = []
            test_ys = []
            test_ids = []

            for graphs, seq_toks, lens, labels, indicies in tqdm(test_loader):
                seq_toks = seq_toks.to(device)
                out = my_model(seq_toks,graphs,lens,bs,True,False,False)

                test_outs = test_outs + out.detach().to('cpu').numpy().tolist()
                test_ys = test_ys + list(labels)
                test_ids = test_ids + list(indicies)

            print(f'test-ROC: {roc_auc_score(test_ys, test_outs):.4f}')
            print(f'test-PR: {average_precision_score(test_ys, test_outs):.4f}')

        # Save model state dictionary
        torch.save(my_model.cpu().state_dict(), os.path.join(out_dir, job_name, f'epoch{epoch}.pt'))
        my_model.to(device) # Move model back to device

        # Log epoch results
        with open(out_dir+'/'+job_name+'/status.txt','a+') as of:
            of.write('epoch: '+str(epoch)+'\n')
            of.write('val-ROC: '+str(roc_auc_score(ys,outs))+'\n')
            of.write('val-PR: '+str(average_precision_score(ys,outs))+'\n')
            of.write('training-loss: '+str(total_loss)+'\n')

# Save final predictions
final_preds = pd.DataFrame(columns=['index', 'label', 'pred'])
for k,t_out in enumerate(test_outs):
    final_preds = pd.concat([final_preds, pd.DataFrame.from_dict({'index': [test_ids[k]], 'label': [test_ys[k]], 'pred': [t_out]})])
final_preds.to_csv(os.path.join('test', job_name + '.csv'), index=False)
