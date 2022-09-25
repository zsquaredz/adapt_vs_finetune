# This script is adpated from repo below
# https://github.com/Adapter-Hub/adapter-transformers/blob/master/examples/pytorch/summarization/run_summarization.py
from transformers import MBartTokenizer, AdapterConfig
from torch.optim import AdamW
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import time
import logging
import argparse
from tqdm import tqdm
import math
from data import SummarizationDataset_XLSUM
from models import mBART_Summarizer_XLSUM
import os
from torch.utils.data.distributed import DistributedSampler
import datasets
import builtins
import re
import transformers
from accelerate import Accelerator

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO, datefmt='%b-%d-%Y %H:%M:%S')
logger = logging.getLogger(__name__)

def set_random_seeds(random_seed=0):
    torch.manual_seed(random_seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    # np.random.seed(random_seed)
    # random.seed(random_seed)

def parse_score(result):
    return {k: round(v.mid.fmeasure * 100, 4) for k, v in result.items()}

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def convert(inputs, convert_dict):
    counter = 1
    convert_list = []
    for summary in inputs:
        summary = summary.strip()
        summary_converted = []
        for token in summary.split():
            if token in convert_dict:
                summary_converted.append(convert_dict[token])
            else:
                convert_dict[token] = str(counter)
                summary_converted.append(convert_dict[token])
            counter += 1
        convert_list.append(' '.join(summary_converted))
    return convert_list

def train(model, iterator, optimizer, tokenizer, args, device):
    model.train()
    epoch_loss = 0

    with tqdm(total=len(iterator)) as t:
        for batch in iterator:
            model.zero_grad() # should be equalvalent with optimizer.zero_grad()
            
            input_ids, attention_mask, out_ids = batch['input_ids'], batch['attention_mask'], batch['labels']

            outputs = model(input_ids, attention_mask, out_ids)
        
            loss = outputs[0]
            accelerator.backward(loss)

            # torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

            optimizer.step()

            epoch_loss += loss.item()
            t.update()

    return epoch_loss / len(iterator)

def validation(model, iterator, tokenizer, args, device, tgt_lang):
    model.eval()
    epoch_loss = 0
    predictions = []
    references = []
    # with tqdm(total=len(iterator)) as t:
    with torch.no_grad():
        for batch in iterator:
            model.zero_grad() # should be equalvalent with optimizer.zero_grad()

            input_ids, attention_mask, out_ids = batch['input_ids'], batch['attention_mask'], batch['labels']

            if args.distributed:
                loss, prediction, reference = model.module.generate(input_ids, attention_mask, out_ids, tokenizer, tgt_lang)
            else:
                loss, prediction, reference = model.generate(input_ids, attention_mask, out_ids, tokenizer, tgt_lang)
            epoch_loss += loss.item()
            # t.update()
            predictions += prediction
            references += reference
        
        # write predictions (List[Str]) down to a file
        with open(args.prediction_file + '.val.'+ tgt_lang, 'w', encoding='utf-8') as f:
            for pred in predictions:
                f.write(pred.strip())
                f.write('\n')
        # write references (List[Str]) down to a file
        with open(args.reference_file + '.val.'+ tgt_lang, 'w', encoding='utf-8') as f:
            for ref in references:
                f.write(ref.strip())
                f.write('\n')

        if args.tgt_lang in ['zh_CN', 'ja_XX']:
            # need special treatment for chinese and japanese
            CONVERT_DICT={}
            pred_original = args.prediction_file + '.val'
            pred_char = args.prediction_file + '.val' + '.char'
            os.system("sed 's/./& /g' " + pred_original + " > " + pred_char)
            ref_original = args.reference_file + '.val'
            ref_char = args.reference_file + '.val' + '.char'
            os.system("sed 's/./& /g' " + ref_original + " > " + ref_char)
            rouge_new = datasets.load_metric('rouge')
            with open(pred_char, encoding='utf-8') as pred_f,\
                open(ref_char, encoding='utf-8') as ref_f:
                predictions_c = pred_f.readlines()
                references_c = ref_f.readlines()
            predictions_converted = convert(predictions_c, CONVERT_DICT)
            references_converted = convert(references_c, CONVERT_DICT)
            rouge_new.add_batch(predictions=predictions_converted, references=references_converted)
            results = rouge_new.compute()
            rouge_dict = parse_score(results)
        elif args.tgt_lang in ['en_XX']:
            # English are just normal
            if args.distributed:
                results = model.module.rouge.compute()
            else:
                results = model.rouge.compute()
            rouge_dict = parse_score(results)
        else:
            # other languages are fine without using char level, but need to use sub
            # metric_names = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
            CONVERT_DICT={}
            rouge_new = datasets.load_metric('rouge')
            predictions_converted = convert(predictions, CONVERT_DICT)
            references_converted = convert(references, CONVERT_DICT)
            rouge_new.add_batch(predictions=predictions_converted, references=references_converted)
            results = rouge_new.compute()
            rouge_dict = parse_score(results)
           
    return epoch_loss / len(iterator), rouge_dict['rouge1'], rouge_dict['rouge2'], rouge_dict['rougeL']

def test(model, iterator, tokenizer, args, device, tgt_lang):
    model.eval()
    predictions = []
    references = []
    # with tqdm(total=len(iterator)) as t:
    with torch.no_grad():
        for batch in iterator:
            model.zero_grad()  # should be equalvalent with optimizer.zero_grad()

            input_ids, attention_mask, out_ids = batch['input_ids'], batch['attention_mask'], batch['labels']
            input_ids.to(device)
            attention_mask.to(device)
            out_ids.to(device)

            prediction, reference = model.inference(input_ids, attention_mask, out_ids, tokenizer, tgt_lang)
            predictions += prediction
            references += reference
            # t.update()
    
        # write predictions (List[Str]) down to a file
        with open(args.prediction_file + '.' + tgt_lang, 'w', encoding='utf-8') as f:
            for pred in predictions:
                f.write(pred.strip())
                f.write('\n')
        # write references (List[Str]) down to a file
        with open(args.reference_file + '.' + tgt_lang, 'w', encoding='utf-8') as f:
            for ref in references:
                f.write(ref.strip())
                f.write('\n')

        if args.tgt_lang in ['zh_CN', 'ja_XX']:
            # need special treatment for chinese and japanese  
            CONVERT_DICT={}         
            pred_original = args.prediction_file
            pred_char = args.prediction_file + '.char'
            os.system("sed 's/./& /g' " + pred_original + " > " + pred_char)
            ref_original = args.reference_file
            ref_char = args.reference_file + '.char'
            os.system("sed 's/./& /g' " + ref_original + " > " + ref_char)
            rouge_new = datasets.load_metric('rouge')
            with open(pred_char, encoding='utf-8') as pred_f,\
                open(ref_char, encoding='utf-8') as ref_f:
                predictions_c = pred_f.readlines()
                references_c = ref_f.readlines()
            predictions_converted = convert(predictions_c, CONVERT_DICT)
            references_converted = convert(references_c, CONVERT_DICT)
            rouge_new.add_batch(predictions=predictions_converted, references=references_converted)
            results = rouge_new.compute()
            rouge_dict = parse_score(results)
        elif args.tgt_lang in ['en_XX']:
            # English are just normal
            results = model.rouge.compute()
            rouge_dict = parse_score(results)
        else:
            # other languages are fine without char level, but need to sub
            # metric_names = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
            CONVERT_DICT={}
            rouge_new = datasets.load_metric('rouge')
            predictions_converted = convert(predictions, CONVERT_DICT)
            references_converted = convert(references, CONVERT_DICT)
            rouge_new.add_batch(predictions=predictions_converted, references=references_converted)
            results = rouge_new.compute()
            rouge_dict = parse_score(results)

    return rouge_dict['rouge1'], rouge_dict['rouge2'], rouge_dict['rougeL']

def collate_fn(batch):
    batch = list(zip(*batch))
    return batch[0], batch[1]

def get_parameter_names(model, forbidden_layer_types):
    """
    Returns the names of the model parameters that are not inside a forbidden layer.
    """
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result

def preprocess_function_ko(examples):
    inputs, targets = [], []
    for i in range(len(examples['text'])):
        if examples['text'][i] is not None and examples['summary'][i] is not None:
            inputs.append(examples['text'][i])
            targets.append(examples['summary'][i])
    
    model_inputs = tokenizer_ko(
                        inputs, 
                        padding='max_length',
                        max_length=args.max_input_len,
                        truncation=True)
    with tokenizer_ko.as_target_tokenizer():
        labels = tokenizer_ko(
                targets,
                padding='max_length',
                max_length=args.max_output_len,
                truncation=True)
    labels['input_ids'] = [
        [(l if l != tokenizer_ko.pad_token_id else -100) for l in label] for label in labels['input_ids']
    ]
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

def preprocess_function_ko1(examples):
    inputs, targets = [], []
    for i in range(len(examples['text'])):
        if examples['text'][i] is not None and examples['summary'][i] is not None:
            inputs.append(examples['text'][i])
            targets.append(examples['summary'][i])
    
    model_inputs = tokenizer_ko(
                        inputs, 
                        padding='max_length',
                        max_length=args.max_input_len,
                        truncation=True)
    with tokenizer_ko.as_target_tokenizer():
        labels = tokenizer_ko(
                targets,
                padding='max_length',
                max_length=args.max_output_len,
                truncation=True)
    
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

def preprocess_function_si(examples):
    inputs, targets = [], []
    for i in range(len(examples['text'])):
        if examples['text'][i] is not None and examples['summary'][i] is not None:
            inputs.append(examples['text'][i])
            targets.append(examples['summary'][i])
    
    model_inputs = tokenizer_si(
                        inputs, 
                        padding='max_length',
                        max_length=args.max_input_len,
                        truncation=True)
    with tokenizer_si.as_target_tokenizer():
        labels = tokenizer_si(
                targets,
                padding='max_length',
                max_length=args.max_output_len,
                truncation=True)
    labels['input_ids'] = [
        [(l if l != tokenizer_si.pad_token_id else -100) for l in label] for label in labels['input_ids']
    ]
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

def preprocess_function_si1(examples):
    inputs, targets = [], []
    for i in range(len(examples['text'])):
        if examples['text'][i] is not None and examples['summary'][i] is not None:
            inputs.append(examples['text'][i])
            targets.append(examples['summary'][i])
    
    model_inputs = tokenizer_si(
                        inputs, 
                        padding='max_length',
                        max_length=args.max_input_len,
                        truncation=True)
    with tokenizer_si.as_target_tokenizer():
        labels = tokenizer_si(
                targets,
                padding='max_length',
                max_length=args.max_output_len,
                truncation=True)
    
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

def preprocess_function_ne(examples):
    inputs, targets = [], []
    for i in range(len(examples['text'])):
        if examples['text'][i] is not None and examples['summary'][i] is not None:
            inputs.append(examples['text'][i])
            targets.append(examples['summary'][i])
    
    model_inputs = tokenizer_ne(
                        inputs, 
                        padding='max_length',
                        max_length=args.max_input_len,
                        truncation=True)
    with tokenizer_ne.as_target_tokenizer():
        labels = tokenizer_ne(
                targets,
                padding='max_length',
                max_length=args.max_output_len,
                truncation=True)
    labels['input_ids'] = [
        [(l if l != tokenizer_ne.pad_token_id else -100) for l in label] for label in labels['input_ids']
    ]
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

def preprocess_function_ne1(examples):
    inputs, targets = [], []
    for i in range(len(examples['text'])):
        if examples['text'][i] is not None and examples['summary'][i] is not None:
            inputs.append(examples['text'][i])
            targets.append(examples['summary'][i])
    
    model_inputs = tokenizer_ne(
                        inputs, 
                        padding='max_length',
                        max_length=args.max_input_len,
                        truncation=True)
    with tokenizer_ne.as_target_tokenizer():
        labels = tokenizer_ne(
                targets,
                padding='max_length',
                max_length=args.max_output_len,
                truncation=True)
    
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

def preprocess_function_fr(examples):
    inputs, targets = [], []
    for i in range(len(examples['text'])):
        if examples['text'][i] is not None and examples['summary'][i] is not None:
            inputs.append(examples['text'][i])
            targets.append(examples['summary'][i])
    
    model_inputs = tokenizer_fr(
                        inputs, 
                        padding='max_length',
                        max_length=args.max_input_len,
                        truncation=True)
    with tokenizer_fr.as_target_tokenizer():
        labels = tokenizer_fr(
                targets,
                padding='max_length',
                max_length=args.max_output_len,
                truncation=True)
    labels['input_ids'] = [
        [(l if l != tokenizer_fr.pad_token_id else -100) for l in label] for label in labels['input_ids']
    ]
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

def preprocess_function_fr1(examples):
    inputs, targets = [], []
    for i in range(len(examples['text'])):
        if examples['text'][i] is not None and examples['summary'][i] is not None:
            inputs.append(examples['text'][i])
            targets.append(examples['summary'][i])
    
    model_inputs = tokenizer_fr(
                        inputs, 
                        padding='max_length',
                        max_length=args.max_input_len,
                        truncation=True)
    with tokenizer_fr.as_target_tokenizer():
        labels = tokenizer_fr(
                targets,
                padding='max_length',
                max_length=args.max_output_len,
                truncation=True)
    
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

def preprocess_function_gu(examples):
    inputs, targets = [], []
    for i in range(len(examples['text'])):
        if examples['text'][i] is not None and examples['summary'][i] is not None:
            inputs.append(examples['text'][i])
            targets.append(examples['summary'][i])
    
    model_inputs = tokenizer_gu(
                        inputs, 
                        padding='max_length',
                        max_length=args.max_input_len,
                        truncation=True)
    with tokenizer_gu.as_target_tokenizer():
        labels = tokenizer_gu(
                targets,
                padding='max_length',
                max_length=args.max_output_len,
                truncation=True)
    labels['input_ids'] = [
        [(l if l != tokenizer_gu.pad_token_id else -100) for l in label] for label in labels['input_ids']
    ]
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

def preprocess_function_gu1(examples):
    inputs, targets = [], []
    for i in range(len(examples['text'])):
        if examples['text'][i] is not None and examples['summary'][i] is not None:
            inputs.append(examples['text'][i])
            targets.append(examples['summary'][i])
    
    model_inputs = tokenizer_gu(
                        inputs, 
                        padding='max_length',
                        max_length=args.max_input_len,
                        truncation=True)
    with tokenizer_gu.as_target_tokenizer():
        labels = tokenizer_gu(
                targets,
                padding='max_length',
                max_length=args.max_output_len,
                truncation=True)
    
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

def run(args):
    # We need to use seeds to make sure that the models initialized in different processes are the same
    set_random_seeds(random_seed=args.seed)
    device = accelerator.device
    if args.distributed:
        args.rank = int(os.environ['RANK'])
    if args.rank != 0:
        def print_pass(*args):
            pass

        builtins.print = print_pass
        logger = logging.getLogger()
        logger.disabled = True
        logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.WARN,
                            datefmt='%b-%d-%Y %H:%M:%S')
    

    logging.info('Loading tokenizer and dataset')
    train_data_ko = datasets.load_dataset("csebuetnlp/xlsum", 'korean', split='train')
    val_data_ko = datasets.load_dataset("csebuetnlp/xlsum", 'korean', split='validation')
    train_data_ko = train_data_ko.map(preprocess_function_ko, batched=True)
    val_data_ko = val_data_ko.map(preprocess_function_ko1, batched=True)
    train_data_ko.set_format(type='torch', columns=['input_ids','attention_mask', 'labels'])
    val_data_ko.set_format(type='torch', columns=['input_ids','attention_mask', 'labels'])

    # args.src_lang, args.tgt_lang = 'si_LK', 'si_LK'
    train_data_si = datasets.load_dataset("csebuetnlp/xlsum", 'sinhala', split='train')
    val_data_si = datasets.load_dataset("csebuetnlp/xlsum", 'sinhala', split='validation')
    train_data_si = train_data_si.map(preprocess_function_si, batched=True)
    val_data_si = val_data_si.map(preprocess_function_si1, batched=True)
    train_data_si.set_format(type='torch', columns=['input_ids','attention_mask', 'labels'])
    val_data_si.set_format(type='torch', columns=['input_ids','attention_mask', 'labels'])

    # args.src_lang, args.tgt_lang = 'ne_NP', 'ne_NP'
    train_data_ne = datasets.load_dataset("csebuetnlp/xlsum", 'nepali', split='train')
    val_data_ne = datasets.load_dataset("csebuetnlp/xlsum", 'nepali', split='validation')
    train_data_ne = train_data_ne.map(preprocess_function_ne, batched=True)
    val_data_ne = val_data_ne.map(preprocess_function_ne1, batched=True)
    train_data_ne.set_format(type='torch', columns=['input_ids','attention_mask', 'labels'])
    val_data_ne.set_format(type='torch', columns=['input_ids','attention_mask', 'labels'])

    # args.src_lang, args.tgt_lang = 'fr_XX', 'fr_XX'
    train_data_fr = datasets.load_dataset("csebuetnlp/xlsum", 'french', split='train')
    val_data_fr = datasets.load_dataset("csebuetnlp/xlsum", 'french', split='validation')
    train_data_fr = train_data_fr.map(preprocess_function_fr, batched=True)
    val_data_fr = val_data_fr.map(preprocess_function_fr1, batched=True)
    train_data_fr.set_format(type='torch', columns=['input_ids','attention_mask', 'labels'])
    val_data_fr.set_format(type='torch', columns=['input_ids','attention_mask', 'labels'])

    # args.src_lang, args.tgt_lang = 'gu_IN', 'gu_IN'
    train_data_gu = datasets.load_dataset("csebuetnlp/xlsum", 'gujarati', split='train')
    val_data_gu = datasets.load_dataset("csebuetnlp/xlsum", 'gujarati', split='validation')
    train_data_gu = train_data_gu.map(preprocess_function_gu, batched=True)
    val_data_gu = val_data_gu.map(preprocess_function_gu1, batched=True)
    train_data_gu.set_format(type='torch', columns=['input_ids','attention_mask', 'labels'])
    val_data_gu.set_format(type='torch', columns=['input_ids','attention_mask', 'labels'])
    # test_data = SummarizationDataset(tokenizer, 'test', args)
    train_data = datasets.concatenate_datasets([train_data_ko,train_data_si,train_data_ne,train_data_fr,train_data_gu])
    model = mBART_Summarizer_XLSUM(args, None)
    model.to(device)

    # Setup adapters
    task_name = args.task_name
    pretrained_model = model.model
    if args.train_adapter:
        # check if adapter already exists, otherwise add it
        if task_name not in pretrained_model.config.adapters:
            # resolve the adapter config
            # see more here https://github.com/Adapter-Hub/adapter-transformers/blob/master/src/transformers/adapters/configuration.py#L485
            adapter_config = AdapterConfig.load(
                args.adapter_config,
                non_linearity=args.adapter_non_linearity,
                reduction_factor=args.adapter_reduction_factor,
            )
            # add a new adapter
            pretrained_model.add_adapter(task_name, config=adapter_config)
        # Freeze all model weights except of those of this adapter
        pretrained_model.train_adapter([task_name])
        # Set the adapters to be used in every forward pass
        pretrained_model.set_active_adapters(task_name)

    logging.info('Model built')
  
    base_params = []
    base_param_size = 0
    adapt_params = []
    adapt_param_size = 0
    for name, param in model.named_parameters():
        if task_name in name or 'layernorm' in name.lower() or 'adapter' in name.lower():
            adapt_params.append(param)
            adapt_param_size += param.view(-1).size()[0]
        else:
            # param.requires_grad = False # this should be done already by the line model.train_adapter([task_name])
            base_params.append(param)
            base_param_size += param.view(-1).size()[0]

    logging.info('base params: ' + str(base_param_size))
    logging.info('adapter params: ' + str(adapt_param_size))

    # below optimizer setup copied from 
    # https://github.com/Adapter-Hub/adapter-transformers/blob/master/src/transformers/adapters/trainer.py
    decay_parameters = get_parameter_names(model, [nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    if hasattr(pretrained_model, "config") and hasattr(pretrained_model.config, "adapters"):
        match_str = r"adapter_fusion_layer\..*\.value"
        decay_parameters = [name for name in decay_parameters if not re.match(match_str, name)]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon)

    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                    num_workers=args.num_workers)
    val_dataloader_ko = DataLoader(val_data_ko, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    val_dataloader_si = DataLoader(val_data_si, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    val_dataloader_ne = DataLoader(val_data_ne, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    val_dataloader_fr = DataLoader(val_data_fr, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    val_dataloader_gu = DataLoader(val_data_gu, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    model, optimizer, train_dataloader, val_dataloader_ko, val_dataloader_si, val_dataloader_ne, val_dataloader_fr, val_dataloader_gu = accelerator.prepare(model, optimizer, train_dataloader, val_dataloader_ko,val_dataloader_si,val_dataloader_ne,val_dataloader_fr,val_dataloader_gu)
    logging.info('Dataset loaded')

    # best_valid_loss = float('inf')
    best_valid_rouge = float('-inf')

    logging.info('Starting the training loop')

    for epoch in range(args.epochs):
        start_time = time.time()
        # # fix sampling seed such that each gpu gets different part of dataset
        # if args.distributed:
        #     train_dataloader.sampler.set_epoch(epoch)

        train_loss = train(model, train_dataloader, optimizer, None, args, device)
        if accelerator.is_local_main_process:
            logging.info('Starting to validate')
            temp_rouge_sum = 0
            valid_loss, val_r1, val_r2, val_rl = validation(model, val_dataloader_ko, tokenizer_ko, args, device, 'ko_KR')
            print(f'\t Ko: Val. R-1: {val_r1} | Val. R-2: {val_r2} | Val. R-L: {val_rl}')
            temp_rouge_sum += (val_r1 + val_r2 + val_rl)/3

            valid_loss, val_r1, val_r2, val_rl = validation(model, val_dataloader_si, tokenizer_si, args, device, 'si_LK')
            print(f'\t Si: Val. R-1: {val_r1} | Val. R-2: {val_r2} | Val. R-L: {val_rl}')
            temp_rouge_sum += (val_r1 + val_r2 + val_rl)/3

            valid_loss, val_r1, val_r2, val_rl = validation(model, val_dataloader_ne, tokenizer_ne, args, device, 'ne_NP')
            print(f'\t Ne: Val. R-1: {val_r1} | Val. R-2: {val_r2} | Val. R-L: {val_rl}')
            temp_rouge_sum += (val_r1 + val_r2 + val_rl)/3

            valid_loss, val_r1, val_r2, val_rl = validation(model, val_dataloader_fr, tokenizer_fr, args, device, 'fr_XX')
            print(f'\t Fr: Val. R-1: {val_r1} | Val. R-2: {val_r2} | Val. R-L: {val_rl}')
            temp_rouge_sum += (val_r1 + val_r2 + val_rl)/3

            valid_loss, val_r1, val_r2, val_rl = validation(model, val_dataloader_gu, tokenizer_gu, args, device, 'gu_IN')
            print(f'\t Gu: Val. R-1: {val_r1} | Val. R-2: {val_r2} | Val. R-L: {val_rl}')
            temp_rouge_sum += (val_r1 + val_r2 + val_rl)/3
            logging.info('Validation done')
            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            val_r_avg = (val_r1 + val_r2 + val_rl)/3
            # if valid_loss < best_valid_loss:
            if val_r_avg > best_valid_rouge:
                # best_valid_loss = valid_loss
                best_valid_rouge = val_r_avg
                checkpoint_dict = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }
                # torch.save(checkpoint_dict, args.output_dir + args.checkpoint + '_epoch' + str(epoch+1) +'.pt')
                torch.save(checkpoint_dict, args.output_dir + args.checkpoint+'.pt')

                # save adapter
                if args.save_adapter and args.train_adapter:
                    if args.distributed:
                        model.module.model.save_adapter(args.output_dir + args.checkpoint + '_adapter_ckpt/', args.task_name)
                    else:
                        model.model.save_adapter(args.output_dir + args.checkpoint + '_adapter_ckpt/', args.task_name)

                # # load model
                # model = AutoAdapterModel.from_pretrained('./path/to/model/directory/')
                # model.load_adapter('./path/to/adapter/directory/')

                logging.info('checkpoint saved')

            print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
            print(f'\t Val. R-1: {val_r1} | Val. R-2: {val_r2} | Val. R-L: {val_rl}')

def getParllelNetworkStateDict(state_dict):
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict

def inference(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = accelerator.device
    logging.info('Initializing to just do inference')

    logging.info('Loading tokenizer and dataset')
    test_data_ko = datasets.load_dataset("csebuetnlp/xlsum", 'korean', split='test')
    test_data_ko = test_data_ko.map(preprocess_function_ko1, batched=True)
    test_data_ko.set_format(type='torch', columns=['input_ids','attention_mask', 'labels'])

    test_data_si = datasets.load_dataset("csebuetnlp/xlsum", 'sinhala', split='test')
    test_data_si = test_data_si.map(preprocess_function_si1, batched=True)
    test_data_si.set_format(type='torch', columns=['input_ids','attention_mask', 'labels'])

    test_data_ne = datasets.load_dataset("csebuetnlp/xlsum", 'nepali', split='test')
    test_data_ne = test_data_ne.map(preprocess_function_ne1, batched=True)
    test_data_ne.set_format(type='torch', columns=['input_ids','attention_mask', 'labels'])

    test_data_fr = datasets.load_dataset("csebuetnlp/xlsum", 'french', split='test')
    test_data_fr = test_data_fr.map(preprocess_function_fr1, batched=True)
    test_data_fr.set_format(type='torch', columns=['input_ids','attention_mask', 'labels'])

    test_data_gu = datasets.load_dataset("csebuetnlp/xlsum", 'gujarati', split='test')
    test_data_gu = test_data_gu.map(preprocess_function_gu1, batched=True)
    test_data_gu.set_format(type='torch', columns=['input_ids','attention_mask', 'labels'])

    test_dataloader_ko = DataLoader(test_data_ko, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_dataloader_si = DataLoader(test_data_si, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_dataloader_ne = DataLoader(test_data_ne, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_dataloader_fr = DataLoader(test_data_fr, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_dataloader_gu = DataLoader(test_data_gu, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    logging.info('Dataset loaded')

    model = mBART_Summarizer_XLSUM(args, None)
    model.to(device)
    # Setup adapters
    task_name = args.task_name
    pretrained_model = model.model
    if args.train_adapter:
        # check if adapter already exists, otherwise add it
        if task_name not in pretrained_model.config.adapters:
            # resolve the adapter config
            # see more here https://github.com/Adapter-Hub/adapter-transformers/blob/master/src/transformers/adapters/configuration.py#L485
            adapter_config = AdapterConfig.load(
                args.adapter_config,
                non_linearity=args.adapter_non_linearity,
                reduction_factor=args.adapter_reduction_factor,
            )
            # add a new adapter
            pretrained_model.add_adapter(task_name, config=adapter_config)
        # Freeze all model weights except of those of this adapter
        pretrained_model.train_adapter([task_name])
        # Set the adapters to be used in every forward pass
        pretrained_model.set_active_adapters(task_name)
    model.to(device)
    logging.info('Model built')

    # load weights into model
    ckpt = torch.load(args.output_dir + args.checkpoint)
    state_dict = getParllelNetworkStateDict(ckpt['state_dict'])
    model.load_state_dict(state_dict)
    logging.info('Model checkpoint loaded')

    model, test_dataloader_ko, test_dataloader_si, test_dataloader_ne, test_dataloader_fr, test_dataloader_gu = accelerator.prepare(model, 
                                                                                                                                        test_dataloader_ko, 
                                                                                                                                        test_dataloader_si,
                                                                                                                                        test_dataloader_ne,
                                                                                                                                        test_dataloader_fr,
                                                                                                                                        test_dataloader_gu)

    start_time = time.time()
    logging.info('Starting to inference')
    r1, r2, rl = test(model, test_dataloader_ko, tokenizer_ko, args, device, 'ko_KR')
    print(f'\t Ko: R-1: {r1} | R-2: {r2} | R-L: {rl}')

    r1, r2, rl = test(model, test_dataloader_si, tokenizer_si, args, device, 'si_LK')
    print(f'\t Si: R-1: {r1} | R-2: {r2} | R-L: {rl}')

    r1, r2, rl = test(model, test_dataloader_ne, tokenizer_ne, args, device, 'ne_NP')
    print(f'\t Ne: R-1: {r1} | R-2: {r2} | R-L: {rl}')

    r1, r2, rl = test(model, test_dataloader_fr, tokenizer_fr, args, device, 'fr_XX')
    print(f'\t Fr: R-1: {r1} | R-2: {r2} | R-L: {rl}')

    r1, r2, rl = test(model, test_dataloader_gu, tokenizer_gu, args, device, 'gu_IN')
    print(f'\t Gu: R-1: {r1} | R-2: {r2} | R-L: {rl}')

    logging.info('Inference done')
    end_time = time.time()
    mins, secs = epoch_time(start_time, end_time)
    print(f'Time: {mins}m {secs}s')

def inference_using_pretrain(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info('Initializing to just do inference')

    logging.info('Loading tokenizer and dataset')
    tokenizer = MBartTokenizer.from_pretrained(args.model_type, src_lang=args.src_lang, tgt_lang=args.tgt_lang)
    test_data = SummarizationDataset(tokenizer, 'test', args)
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    logging.info('Dataset loaded')

    model = mBART_Summarizer(args, tokenizer)
    model.to(device)
    logging.info('Model built')

    # # load weights into model
    # ckpt = torch.load(args.output_dir + args.checkpoint)
    # state_dict = (ckpt['state_dict'])
    # model.load_state_dict(state_dict)
    # logging.info('Model checkpoint loaded')

    start_time = time.time()
    logging.info('Starting to inference')
    r1, r2, rl = test(model, test_dataloader, tokenizer, args, device)
    logging.info('Inference done')
    end_time = time.time()
    mins, secs = epoch_time(start_time, end_time)
    print(f'Time: {mins}m {secs}s')
    print(f'\t R-1: {r1} | R-2: {r2} | R-L: {rl}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="summarization")
    parser.add_argument("--seed", type=int, default=1234, help="Seed")
    parser.add_argument("--lr", type=float, default=0.00005, help="Maximum learning rate")
    parser.add_argument("--warmup", type=int, default=1000, help="Number of warmup steps")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of data loader workers")
    parser.add_argument("--output_dir", type=str, default='./saved_models/test', help="Location of output dir")
    parser.add_argument("--task_prefix", type=str, default='', help="Task prefix for T5 model, for other models, leave empty")
    parser.add_argument("--src_lang", type=str, default='zh_CN', help="Language of src, use Fairseq language code")
    parser.add_argument("--tgt_lang", type=str, default='en_XX', help="Language of tgt, use Fairseq language code")
    parser.add_argument("--max_input_len", type=int, default=512, help="maximum num of wordpieces in the document")
    parser.add_argument("--max_output_len", type=int, default=256, help="maximum num of wordpieces in the summary")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--beam_size", type=int, default=5, help="Beam size")
    parser.add_argument("--inference_min_len", type=int, default=10, help="minimum num of tokens in the summary at inference")
    parser.add_argument("--inference_max_len", type=int, default=200, help="maximum num of tokens in the summary at inference")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="Repetition penalty at generation step. 1.0 means no penalty")
    parser.add_argument("--length_penalty", type=float, default=1.0, help="Exponential penalty to length at generation step. 1.0 means no penalty")
    parser.add_argument("--no_repeat_ngram_size", type=int, default=3, help="No repeat Ngram size")
    parser.add_argument("--grad_accum", type=int, default=1, help="number of gradient accumulation steps")
    parser.add_argument("--fp16", action='store_true', help="Use fp16 ")
    parser.add_argument("--mixed_precision",  type=str, default='no', help="Use mixed precision training, choose from 'no','fp16','bf16'")
    # parser.add_argument('--grad_ckpt', action='store_true', help='Enable gradient checkpointing to save memory')
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8, help="Adam epsilon")
    parser.add_argument("--data_path", type=str, default='./data/',
                        help="Location of dataset directory")
    parser.add_argument("--task", type=str, default='train', help="train | val | test")
    parser.add_argument('--model_type', default='"facebook/mbart-large-cc25"', type=str,
                        help="The model to use, e.g. t5-small.")
    parser.add_argument("--checkpoint", type=str, default='test.ckpt',
                        help="Location of checkpoint path, when used for testing, should be the specific checkpoint")
    parser.add_argument("--prediction_file", type=str, default='predictions.txt',
                        help="Location of predictions path")
    parser.add_argument("--reference_file", type=str, default='references.txt',
                        help="Location of references path")

    ######### args for distributed computing ###########
    # parser.add_argument('--world_size', default=-1, type=int,
    #                     help='Number of total GPU used for distributed training. gpus/node * num_nodes')
    # parser.add_argument('--rank', default=-1, type=int,
    #                     help='Node rank for distributed training')
    # parser.add_argument('--gpu', default=None, type=int)
    # parser.add_argument("--local_rank", default=-1, type=int)
    # parser.add_argument("--ngpu", type=int, default=-1, help="Number of GPUs")
    # parser.add_argument("--nnode", type=int, default=1, help="Number of GPU Nodes")
    parser.add_argument("--distributed", action='store_true', help="whether to use distributed")
    parser.add_argument("--continue_training", action='store_true', help="continue training")
    parser.add_argument("--exp_id", type=str, default='', help="a unique experiment identifier.")

    ######### args for adatpers ###########
    parser.add_argument("--train_adapter", action='store_true', help="whether to use Adapter")
    parser.add_argument("--save_adapter", action='store_true', help="whether to save Adapter checkpoint")
    parser.add_argument("--task_name", type=str, default='summarization', help="Name for the task adapter")
    parser.add_argument("--adapter_config", type=str, default='houlsby', help="Adapter configuration: houlsby, pfeiffer, parallel, compacter, etc.")
    parser.add_argument("--adapter_non_linearity", type=str, default='swish', help="Non-linearity fn for adapter: swish, gelu, relu, sigmoid, tanh, etc.")
    parser.add_argument("--adapter_reduction_factor", type=int, default=16, help="The ratio between a model's layer hidden dimension and the bottleneck dimension")
    args = parser.parse_args()
    
    
    FAIRSEQ_LANGUAGE_CODES = ["ar_AR", "cs_CZ", "de_DE", "en_XX", "es_XX", "et_EE", "fi_FI", "fr_XX", "gu_IN", 
                            "hi_IN", "it_IT", "ja_XX", "kk_KZ", "ko_KR", "lt_LT", "lv_LV", "my_MM", "ne_NP", 
                            "nl_XX", "ro_RO", "ru_RU", "si_LK", "tr_TR", "vi_VN", "zh_CN"]
    assert args.src_lang in FAIRSEQ_LANGUAGE_CODES
    assert args.tgt_lang in FAIRSEQ_LANGUAGE_CODES

    tokenizer_ko = MBartTokenizer.from_pretrained(args.model_type, src_lang='ko_KR', tgt_lang='ko_KR')
    tokenizer_si = MBartTokenizer.from_pretrained(args.model_type, src_lang='si_LK', tgt_lang='si_LK')
    tokenizer_ne = MBartTokenizer.from_pretrained(args.model_type, src_lang='ne_NP', tgt_lang='ne_NP')
    tokenizer_fr = MBartTokenizer.from_pretrained(args.model_type, src_lang='fr_XX', tgt_lang='fr_XX')
    tokenizer_gu = MBartTokenizer.from_pretrained(args.model_type, src_lang='gu_IN', tgt_lang='gu_IN')

    accelerator = Accelerator(fp16=args.fp16, mixed_precision=args.mixed_precision)
    logger.info(accelerator.state)

    # # Setup logging, we only want one process per machine to log things on the screen.
    # # accelerator.is_local_main_process is only True for one process per machine.
    # logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    # if accelerator.is_local_main_process:
    #     datasets.utils.logging.set_verbosity_warning()
    #     transformers.utils.logging.set_verbosity_info()
    # else:
    #     datasets.utils.logging.set_verbosity_error()
    #     transformers.utils.logging.set_verbosity_error()
    logging.info(args)

    mode = args.task
    if mode == 'train':
        logging.info('Initializing to train')
        run(args)
    elif mode == 'retrain':
        logging.info('Restarting to train using '+args.checkpoint)
        # run_from_checkpoint(verification_model, optimizer, args.num_epoch, args.checkpoint)
    elif mode == 'eval':
        logging.info('Starting to eval using '+args.checkpoint)
        # eval_using_checkpoint(verification_model, args.checkpoint)
    elif mode == 'test':
        logging.info('Starting to test using '+args.checkpoint)
        inference(args)
        # inference_using_pretrain(args)
    else:
        print('wrong mode')



