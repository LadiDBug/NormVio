import os
import sys
import json
import random
import argparse
import unicodedata
import itertools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import sigmoid
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from time import time
from datetime import datetime
from torch import optim
from os.path import join, exists
from multiprocessing import Pool
from convokit import Corpus
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import BertTokenizer
from config import BERT_TYPE, BERT_CACHE, HIDDEN_SIZE, MAX_LENGTH, ENC_NUM_LAYER, DROPOUT_PROB, PAD_token, SOS_token, EOS_token, UNK_token, SEP_STRING
from config import BATCH_SIZE, VALID_BATCH_SIZE, CLIP, TF_RATIO, LR, EPOCHS, EARLY_STOPPING
from models import EncoderBERT, ContextEncoderRNN, SingleTargetClf, Predictor
from utils import import_jsonl, import_json, import_tsv, save_json, check_or_create_dir
# Create temp directory at the start
os.makedirs("temp", exist_ok=True)

#TODO cambiare nell'altro file il nome dei singoli modelli e salvarli in una unica cartella in modo da avere una cartella unica con tutti i modelli


def load_dataset(path, filter_removed=False):
    data = {}
    for split in ["test_n_communities_out","test_n_rules_out","test_stratified"]:
        data[split]= import_jsonl(join(path, split+".jsonl"))
    return data
        

# load a single model
def load_model(model_dir, device, num_classes):

    checkpoint_path = join(model_dir, "finetuned_model.pt")
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    encoder = EncoderBERT(device = device)
    if 'en' in checkpoint:
        encoder.load_state_dict(checkpoint['en'])
        print("EncoderBERT weights loaded.")
    else:
        raise KeyError("'en' (EncoderBERT state_dict) not found in checkpoint.")
    encoder.to(device)

    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=LR)
    if 'en_opt' in checkpoint:
        encoder_optimizer.load_state_dict(checkpoint['en_opt'])
        print("EncoderBERT optimizer state loaded.")
    else:
        print("Warning: 'en_opt' (EncoderBERT optimizer state) not found in checkpoint. Optimizer initialized fresh.")


    context_encoder = None
    context_encoder_optimizer = None
    if 'ctx' in checkpoint and checkpoint['ctx'] is not None:
        context_encoder = ContextEncoderRNN(
            hidden_size=HIDDEN_SIZE,
            n_layers=ENC_NUM_LAYER,
            dropout=DROPOUT_PROB,
            device=device
        )
        context_encoder.load_state_dict(checkpoint['ctx'])
        print("ContextEncoderRNN weights loaded.")
        context_encoder.to(device)

        context_encoder_optimizer = torch.optim.Adam(context_encoder.parameters(), lr=LR)
        if 'ctx_opt' in checkpoint and checkpoint['ctx_opt'] is not None:
            context_encoder_optimizer.load_state_dict(checkpoint['ctx_opt'])
            print("ContextEncoderRNN optimizer state loaded.")
        else:
            print("Warning: 'ctx_opt' (ContextEncoderRNN optimizer state) not found or is None in checkpoint. Optimizer initialized fresh.")
    elif 'ctx' in checkpoint and checkpoint['ctx'] is None:
        print("ContextEncoderRNN state_dict is None in checkpoint (likely use_context=False during training).")
    else:
        print("No 'ctx' (ContextEncoderRNN state_dict) in checkpoint. Assuming no context encoder was used.")


    attack_clf = SingleTargetClf(
        hidden_size=HIDDEN_SIZE,
        num_classes=num_classes, # Passato come argomento alla funzione
        dropout=DROPOUT_PROB,
        device=device
    )
    if 'atk_clf' in checkpoint:
        attack_clf.load_state_dict(checkpoint['atk_clf'])
        print("SingleTargetClf weights loaded.")
    else:
        raise KeyError("'atk_clf' (SingleTargetClf state_dict) not found in checkpoint.")
    attack_clf.to(device)

    attack_clf_optimizer = torch.optim.Adam(attack_clf.parameters(), lr=LR)
    if 'atk_clf_opt' in checkpoint:
        attack_clf_optimizer.load_state_dict(checkpoint['atk_clf_opt'])
        print("SingleTargetClf optimizer state loaded.")
    else:
        print("Warning: 'atk_clf_opt' (SingleTargetClf optimizer state) not found in checkpoint. Optimizer initialized fresh.")


    print(f"Individual models and optimizers loaded from {model_dir}.")
    return encoder, context_encoder, attack_clf, encoder_optimizer, context_encoder_optimizer, attack_clf_optimizer



# PART OF PROCESSING DATA. 
################################################################################################################################################à

def preprocess_data(df, df_rules, cat_idx_mapping, target_class_idx, min_context=2, max_context=4, use_context=False, append_subreddit=None, append_rule=False, is_test=False):
    pairs = []
    rule_counts = {}
    #df_rules_subreddit = {subreddit: sub_df for subreddit, sub_df in df_rules.groupby('subreddit')}
    data = list(map(preprocess_row_binary,[(row, target_class_idx, cat_idx_mapping, None, is_test, use_context, append_subreddit, min_context, max_context) for row in df]))
    
    pairs = [pair for pairs in data for pair in pairs]
    return pairs


def preprocess_row_binary(inp):
    row, target_class_idx, cat_idx_mapping, df_rules_subreddit, is_test, use_context, append_subreddit, min_context, max_context = inp
    append_rule = False # might change this in future. but for now we assume no rule text is given as input.

    context, final_comment, rule_texts, cats = row['context'], row['final_comment'], row['rule_texts'], row['cats']
    comment_id, conv_id, label, subreddit = row['comment_id'], row['conv_id'], row['bool_derail'], row['subreddit']
    pairs = []
    if use_context and len(context) < min_context:
        return pairs
    
    all_cat_idxs = [cat_idx_mapping[c] for cat in cats.split(SEP_STRING) for c in cat.split(",") if c!=""]
    # rule_texts, cats = rule_texts.split(SEP_STRING), cats.split(SEP_STRING)

    encoded_final_comment = encode(get_input_text(final_comment["tokens"], None, subreddit, append_rule, append_subreddit))
    encoded_context = None
    if use_context:
        if max_context is None:
            encoded_context = [encode(get_input_text(u["tokens"], None, subreddit, append_rule, append_subreddit)) for u in context]
        else:
            encoded_context = [encode(get_input_text(u["tokens"], None, subreddit, append_rule, append_subreddit)) for u in context[-max_context:]]
    
    # training pair: (final_comment, context, binary_label, rule_idx, conv_id+"||"comment_id+"||"+str(correct_rule_idx))
    data_sample_id = conv_id + "||" + comment_id + "||" + ",".join([str(c) for c in target_class_idx])
    if target_class_idx == [0]:
        binary_label = int(not label)
    else:
        if label is False:
            binary_label = 0
        else:
            binary_label = int(any([(c in all_cat_idxs) for c in target_class_idx]))
    pairs.append((encoded_final_comment, encoded_context, binary_label, target_class_idx, data_sample_id))
    return pairs



def encode(text, max_length=MAX_LENGTH):
    # simplify the problem space by considering only ASCII data
    cleaned_text = unicodeToAscii(text)

    # if the resulting string is empty, nothing else to do
    if not cleaned_text.strip():
        return []

    return tokenizer.encode(cleaned_text, add_special_tokens=True, truncation=True, max_length=max_length)


NUM_PROCESS = 22
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('DEVICE: ' , device)
tokenizer = BertTokenizer.from_pretrained(BERT_TYPE)

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def get_input_text(tokens, rule_text=None, subreddit_name=None, append_rule=False, append_subreddit=None):
    input_text = tokens
    if append_subreddit == "subreddit":
        input_text = f"r/{subreddit_name} {input_text}"
   # elif append_subreddit == "nsfw":
   #     input_text = f"r/{get_subreddit_type(subreddit_name)} {input_text}"
    if append_rule:
        input_text += " [SEP] "+rule_text
    return input_text

def get_convid_to_uttr(split_data):
    _convid_to_uttr = {}
    for d in split_data:
        _convid_to_uttr[d['conv_id']] = d['context'] + [d['final_comment']]
    return _convid_to_uttr


##################################################################################################################################################################

# FUNZIONI PER IL TESTING


def zeroPadding(l, fillvalue=PAD_token):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))

# Returns padded input sequence tensor and lengths
def inputVar(l):
    indexes_batch = l
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths

def dialogBatch2UtteranceBatch(dialog_batch):
    utt_tuples = []  # will store tuples of (utterance, original position in batch, original position in dialog)
    for batch_idx in range(len(dialog_batch)):
        dialog = dialog_batch[batch_idx]
        for dialog_idx in range(len(dialog)):
            utterance = dialog[dialog_idx]
            utt_tuples.append((utterance, batch_idx, dialog_idx))
    # sort the utterances in descending order of length, to remain consistent with pytorch padding requirements
    utt_tuples.sort(key=lambda x: len(x[0]), reverse=True)
    # return the utterances, original batch indices, and original dialog indices as separate lists
    utt_batch = [u[0] for u in utt_tuples]
    batch_indices = [u[1] for u in utt_tuples]
    dialog_indices = [u[2] for u in utt_tuples]
    return utt_batch, batch_indices, dialog_indices

# Returns all items for a given batch of pairs
def batch2TrainData(pair_batch, already_sorted=False):
    if not already_sorted:
        pair_batch.sort(key=lambda x: len(x[0]), reverse=True)
    input_batch, contexts, label_batch, id_batch = [], [], [], []
    for pair in pair_batch:
        final_comment, context, label, _, id = pair
        input_batch.append([final_comment])
        contexts.append(context)
        label_batch.append(label)
        id_batch.append(id)
    
    if contexts[0] is not None:
        input_batch = [context+comment for context, comment in zip(contexts, input_batch)]

    dialog_lengths = torch.tensor([len(x) for x in input_batch])
    input_utterances, batch_indices, dialog_indices = dialogBatch2UtteranceBatch(input_batch)
    inp, utt_lengths = inputVar(input_utterances)
    label_batch = torch.FloatTensor(label_batch) if label_batch[0] is not None else None
    return inp, dialog_lengths, utt_lengths, batch_indices, dialog_indices, label_batch, id_batch


def batchIterator(source_data, batch_size, shuffle=True):
    cur_idx = 0
    if shuffle:
        random.shuffle(source_data)
    while True:
        if cur_idx >= len(source_data):
            cur_idx = 0
            if shuffle:
                random.shuffle(source_data)
        batch = source_data[cur_idx:(cur_idx + batch_size)]
        # the true batch size may be smaller than the given batch size if there is not enough data left
        true_batch_size = len(batch)
        # ensure that the dialogs in this batch are sorted by length, as expected by the padding module
        if batch[0][1] != None:
            batch.sort(key=lambda x: len(x[1])+1, reverse=True)
        else:
            batch.sort(key=lambda x: len(x[0]), reverse=True)
        # for analysis purposes, get the source dialogs and labels associated with this batch
        batch_dialogs = [x[0] for x in batch]
        batch_labels = [x[2] for x in batch]

        # convert batch to tensors
        batch_tensors = batch2TrainData(batch, already_sorted=True)
        yield (batch_tensors, true_batch_size)
        cur_idx += batch_size

def evaluateBatch(encoder, context_encoder, predictor, input_batch, dialog_lengths, utt_lengths, batch_indices, dialog_indices, batch_size, device, thre=0.5,
                  max_length=MAX_LENGTH):
    # Set device options
    input_batch = input_batch.to(device)
    dialog_lengths = dialog_lengths.to(device)
    utt_lengths = utt_lengths.to(device)
    # Predict future attack using predictor
    scores = predictor(input_batch, dialog_lengths, utt_lengths, batch_indices, dialog_indices,
                       batch_size, max_length)
    if len(scores.shape) == 1:
        scores = scores.unsqueeze(0)
    
    _, predictions = torch.max(scores, 1)
    return predictions, scores


def evaluateDataset(dataset, encoder, context_encoder, predictor, batch_size):
    # create a batch iterator for the given data
    batch_iterator = batchIterator(dataset, batch_size, shuffle=False)
    # find out how many iterations we will need to cover the whole dataset
    n_iters = len(dataset) // batch_size + int(len(dataset) % batch_size > 0)
    output_df = {
        "id": [],
        "prediction": [],
        "score": [],
        'label': []
    }
    with torch.no_grad():
        # aggiunto tqdm
        for iteration in tqdm(range(1, n_iters + 1)):
            batch, true_batch_size = next(batch_iterator)
            # Extract fields from batch
            input_variable, dialog_lengths, utt_lengths, batch_indices, dialog_indices, labels, conv_ids = batch
            # run the model
            predictions, scores = evaluateBatch(encoder, context_encoder, predictor, input_variable,
                                                dialog_lengths, utt_lengths, batch_indices, dialog_indices,
                                                true_batch_size, device)

            # format the output as a dataframe (which we can later re-join with the corpus)
            for i in range(true_batch_size):
                conv_id = conv_ids[i]
                pred = predictions[i].item()
                label = labels[i].item()
                score = -1  # scores[i].item()
                output_df["id"].append(conv_id)
                output_df["prediction"].append(pred)
                output_df["score"].append(score)
                output_df["label"].append(label)
            # if iteration % 1000 == 0:
            #     print(f"Iteration: {iteration}/{n_iters} ({iteration/n_iters*100:.1f}%)")

    return pd.DataFrame(output_df).set_index("id")


# Utility fn to calculate val F1, used during training to check for best model
def get_best_val_f1(val_pairs, convid_to_uttr, encoder, context_encoder, predictor, batch_size, model_output_dir, test_name):
    forecasts_df = evaluateDataset(val_pairs, encoder, context_encoder, predictor, batch_size)
    
    os.makedirs("temp", exist_ok=True)
    forecasts_df.to_csv("temp/temp.tsv",sep="\t")

    predicted_corpus = []
    conversational_forecasts_df = []
    for idx, row in forecasts_df.iterrows():
        conv_id, utt_id, rule_id = idx.split("||")
        key = conv_id + "||" + rule_id
        label = int(row['label'])
        utterances = [utterance['tokens'] for utt_idx, utterance in enumerate(convid_to_uttr[conv_id])]
        predicted_corpus.append({'key':key,'conv_id': conv_id, 'rule_id': rule_id,'uttr_id':utt_id, 'pred':row['prediction'], 'label': label, 'final_comment':utterances[-1], 'context':utterances[:-1]})

    df_detection = pd.DataFrame(predicted_corpus)
    df_detection.to_csv("temp/temp2.tsv",sep="\t")
    output_path = os.path.join(model_output_dir, f"test_for_{test_name}.tsv")
    df_detection.to_csv(output_path, sep="\t", index=False)

    f1 = f1_score(df_detection['label'], df_detection['pred'], average='macro')
    acc = accuracy_score(df_detection['label'], df_detection['pred'])
    prec = precision_score(df_detection['label'], df_detection['pred'], average='macro')
    rec = recall_score(df_detection['label'], df_detection['pred'], average='macro')
    res = {'f1':f1, 'acc':acc, 'prec':prec, 'rec':rec}
    return res, df_detection

def evaluate(encoder, attack_clf, processed_data, convid_to_uttr, model_output_dir, valid_batch_size, test_name):

    encoder.eval()
    attack_clf.eval()

    predictor = Predictor(encoder, None, attack_clf)
    start = time()

    res_dev, df_detection = get_best_val_f1(processed_data[test_name], convid_to_uttr[test_name], encoder, None, predictor, valid_batch_size, model_output_dir, test_name)
    dev_f1, dev_acc = res_dev['f1'], res_dev['acc']
    print(f" f1: {dev_f1 * 100:.1f}, acc: {dev_acc*100:.1f} (took {(time()-start)/60:.1f}m)")
    
##################################################################################################################################################################

def main():

    # path vari
    test_path = path
    models_dir = path
    output_dir = path

    # path idx delle categorie
    cat_idx_mapping = import_json("data/mappings/cat10_to_idx.json")
    cat_idx_mapping['Safe'] = 0
    NUM_CLASSES = 2

    # target class è la lista delle categorie target_class = ["spam", "off-topic", "ecc..."]
    target_class = list(cat_idx_mapping.keys())
    
    # lista degli indici delle categorie ["1", "2", "7", "ecc..."]
    target_class_idx = [cat_idx_mapping[c] for c in target_class]
    data = load_dataset(test_path)

    # per ogni modello nella dir
    for model_sub_dir in os.listdir(models_dir):
        path_model = os.path.join(models_dir, model_sub_dir)
        
        # necessario per escludere summary.tsv
        if os.path.isfile(path_model):
            continue

         # Crea cartella di output corrispondente
        model_output_dir = os.path.join(output_dir, model_sub_dir)
        os.makedirs(model_output_dir, exist_ok=True)


        target_class_name = model_sub_dir
        if target_class_name not in cat_idx_mapping:
            print(f"Warning: {target_class_name} not in cat_idx_mapping")
            continue

        target_class_idx = [cat_idx_mapping[target_class_name]]
        
        # prendi dalla cartella il file finetuned_model.pt
        # carica il modello -> load_model
        encoder, context_encoder, attack_clf, _, _, _  = load_model(path_model, device, 2)
        print('Model loaded!')


        processed_data = {}
        for split in ['test_n_communities_out', 'test_n_rules_out', 'test_stratified']:
            processed_data[split] = preprocess_data(
                data[split], 
                None, 
                cat_idx_mapping, 
                target_class_idx, 
                1, 6, 
                False, 
                append_subreddit="subreddit", 
                append_rule=False, 
                is_test=True
            )
            
        convid_to_uttr = {}
        convid_to_uttr['test_n_communities_out'] = get_convid_to_uttr(data['test_n_communities_out'])
        convid_to_uttr['test_n_rules_out'] = get_convid_to_uttr(data['test_n_rules_out'])
        convid_to_uttr['test_stratified'] = get_convid_to_uttr(data['test_stratified'])

        for split in processed_data:
            evaluate(
                encoder, attack_clf, processed_data, convid_to_uttr, model_output_dir, valid_batch_size=12, test_name=split
            )
        # evaluate(encoder, attack_clf, processed_data, convid_to_uttr, model_output_dir, valid_batch_size=12, test_name='test_n_communities_out')
        # evaluate(encoder, attack_clf, processed_data, convid_to_uttr, model_output_dir, valid_batch_size=12, test_name='test_n_rules_out')
        # evaluate(encoder, attack_clf, processed_data, convid_to_uttr, model_output_dir, valid_batch_size=12, test_name='test_stratified')


        

if __name__ == "__main__":
    main()