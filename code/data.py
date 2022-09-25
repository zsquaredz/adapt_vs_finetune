import json
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW
import torch
from torch.utils.data import Dataset, DataLoader
import io
import torch.nn as nn
import datasets
import time
import logging
import argparse
from tqdm import tqdm
import math

class SummarizationDataset(Dataset):
    def __init__(self, tokenizer, split, args):  
        super().__init__()     
        self.args = args  
        self.src_lang = args.src_lang.split('_')[0] # zh_CN --> zh
        self.tgt_lang = args.tgt_lang.split('_')[0] # en_XX --> en
        self.documents = self.get_documents(split=split, src_lang=self.src_lang, tgt_lang=self.tgt_lang)
        self.summaries = self.get_summaries(split=split, src_lang=self.src_lang, tgt_lang=self.tgt_lang)
        self.tokenizer = tokenizer
        
    def get_documents(self, split, src_lang, tgt_lang):
        data = []
        path = self.args.data_path + split + '.' + src_lang + '-' + tgt_lang + '.source'
        with io.open(path, encoding='utf-8') as f:
            for line in f:
                data.append(self.args.task_prefix + line.strip()) 
                # T5 need a task prompt as prefix to the source text, for summarization it is "summarize: "
                # need to double check what the prefix is for multilingual T5
        return data
    
    def get_summaries(self, split, src_lang, tgt_lang):
        data = []
        path = self.args.data_path + split + '.' + src_lang + '-' + tgt_lang + '.target'
        with io.open(path, encoding='utf-8') as f:
            for line in f:
                data.append(line.strip())
        return data
  
    def __len__(self):
        return len(self.documents)
  
    def __getitem__(self, idx):
        document = self.documents[idx]
        summary = self.summaries[idx]
        
        return document, summary

class SummarizationDataset_XLSUM(Dataset):
    def __init__(self, tokenizer, split, args):  
        super().__init__()     
        self.args = args  
        self.src_lang = args.src_lang.split('_')[0] # zh_CN --> zh
        self.tgt_lang = args.tgt_lang.split('_')[0] # en_XX --> en
        self.documents = self.get_documents(split=split, src_lang=self.src_lang, tgt_lang=self.tgt_lang)
        self.summaries = self.get_summaries(split=split, src_lang=self.src_lang, tgt_lang=self.tgt_lang)
        self.tokenizer = tokenizer
        
    def get_documents(self, split, src_lang, tgt_lang):
        data = []
        path = self.args.data_path + src_lang + '-' + tgt_lang + '/' + split + '.' + src_lang + '-' + tgt_lang + '.source'
        with io.open(path, encoding='utf-8') as f:
            for line in f:
                data.append(self.args.task_prefix + line.strip()) 
                # T5 need a task prompt as prefix to the source text, for summarization it is "summarize: "
                # need to double check what the prefix is for multilingual T5
        return data
    
    def get_summaries(self, split, src_lang, tgt_lang):
        data = []
        path = self.args.data_path + src_lang + '-' + tgt_lang + '/' + split + '.' + src_lang + '-' + tgt_lang + '.target'
        with io.open(path, encoding='utf-8') as f:
            for line in f:
                data.append(line.strip())
        return data
  
    def __len__(self):
        return len(self.documents)
  
    def __getitem__(self, idx):
        document = self.documents[idx]
        summary = self.summaries[idx]
        
        batch_encoding = self.tokenizer(
                        document, 
                        padding='max_length',
                        max_length=self.args.max_input_len,
                        truncation=True,
                        return_tensors="pt")
            # Setup the tokenizer for targets
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                    summary,
                    padding='max_length',
                    max_length=self.args.max_output_len,
                    truncation=True,
                    return_tensors="pt")

            input_ids = batch_encoding['input_ids']
            attention_mask = batch_encoding['attention_mask']

            batch_encoding["labels"] = labels["input_ids"]
            out_ids = batch_encoding['labels']
            out_ids[out_ids[:, :] == self.tokenizer.pad_token_id] = -100

        return input_ids, attention_mask, out_ids
    
