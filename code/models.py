from transformers import MBartForConditionalGeneration as mBARTOriginal
from transformers import BartForConditionalGeneration as BARTOriginal
import torch
import torch.nn as nn
import datasets
import math
from transformers.adapters import BartAdapterModel, MBartAdapterModel


def gelu_new(x):
    """
    Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
    Also see https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class mBART_Summarizer(nn.Module):
    def __init__(self, params, tokenizer) -> None:
        super().__init__()
        self.args = params
        self.tokenizer = tokenizer

        self.model = mBARTOriginal.from_pretrained(self.args.model_type)
        if self.args.continue_training:
            # load pre-trained weights into model
            ckpt = torch.load(self.args.output_dir + self.args.checkpoint)
            self.model.load_state_dict(ckpt['state_dict'])

        self.rouge = datasets.load_metric('rouge')

    def forward(self, input_ids, attention_mask, out_ids):
        # out_ids = batch_encoding['labels']
        # out_ids[out_ids[:, :] == self.tokenizer.pad_token_id] = -100
        # out_mask = torch.where(out_ids==0, 0, 1).unsqueeze(-1) # batch_size, out_len
        # the T5 model should creates its won decoder_attention_mask
        # https://huggingface.co/transformers/model_doc/t5.html#t5forconditionalgeneration

        # dec_inp_ids = model._shift_right(out_ids)
        # # no need above for T5, please see: https://huggingface.co/transformers/glossary.html#decoder-input-ids

        return self.model(input_ids=input_ids,
                          attention_mask=attention_mask,
                          labels=out_ids)

    def generate(self, input_ids, attention_mask, out_ids):
        generated_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.args.inference_max_len,
            min_length=self.args.inference_min_len,
            num_beams=self.args.beam_size,
            no_repeat_ngram_size=self.args.no_repeat_ngram_size,
            repetition_penalty=self.args.repetition_penalty,
            length_penalty=self.args.length_penalty,
            early_stopping=True,
            decoder_start_token_id=self.tokenizer.lang_code_to_id[self.args.tgt_lang]
        )

        # Convert predicted and gold token ids to strings
        predictions = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        references = self.tokenizer.batch_decode(out_ids, skip_special_tokens=True)

        self.rouge.add_batch(predictions=predictions, references=references)
        loss = self(input_ids, attention_mask, out_ids)[0]
        return loss, predictions, references

    def inference(self, input_ids, attention_mask, out_ids):
        # the only diff with generate() is that this one writes down predictions
        generated_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.args.inference_max_len,
            min_length=self.args.inference_min_len,
            num_beams=self.args.beam_size,
            no_repeat_ngram_size=self.args.no_repeat_ngram_size,
            repetition_penalty=self.args.repetition_penalty,
            length_penalty=self.args.length_penalty,
            early_stopping=True,
            decoder_start_token_id=self.tokenizer.lang_code_to_id[self.args.tgt_lang]
        )

        # Convert predicted and gold token ids to strings
        predictions = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        references = self.tokenizer.batch_decode(out_ids, skip_special_tokens=True)

        self.rouge.add_batch(predictions=predictions, references=references)

        return predictions, references

class mBART_Summarizer_XLSUM(nn.Module):
    def __init__(self, params, tokenizer) -> None:
        super().__init__()
        self.args = params
        # self.tokenizer = tokenizer 
        if self.args.continue_training:
            self.model = mBARTOriginal.from_pretrained(self.args.output_dir +
                                                       self.args.checkpoint + '.pt')
        else:
            self.model = mBARTOriginal.from_pretrained(self.args.model_type)
        self.rouge = datasets.load_metric('rouge')

    def forward(self, input_ids, attention_mask, out_ids):
        # out_ids = batch_encoding['labels']
        # out_ids[out_ids[:, :] == self.tokenizer.pad_token_id] = -100
        # out_mask = torch.where(out_ids==0, 0, 1).unsqueeze(-1) # batch_size, out_len
        # the T5 model should creates its won decoder_attention_mask
        # https://huggingface.co/transformers/model_doc/t5.html#t5forconditionalgeneration

        # dec_inp_ids = model._shift_right(out_ids)
        # # no need above for T5, please see: https://huggingface.co/transformers/glossary.html#decoder-input-ids

        return self.model(input_ids=input_ids,
                          attention_mask=attention_mask,
                          labels=out_ids)

    def generate(self, input_ids, attention_mask, out_ids, tokenizer, tgt_lang):
        generated_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.args.inference_max_len,
            min_length=self.args.inference_min_len,
            num_beams=self.args.beam_size,
            no_repeat_ngram_size=self.args.no_repeat_ngram_size,
            repetition_penalty=self.args.repetition_penalty,
            length_penalty=self.args.length_penalty,
            early_stopping=True,
            decoder_start_token_id=tokenizer.lang_code_to_id[tgt_lang]
        )

        # Convert predicted and gold token ids to strings
        predictions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        # references = predictions
        references = tokenizer.batch_decode(out_ids, skip_special_tokens=True)

        self.rouge.add_batch(predictions=predictions, references=references)
        loss = self(input_ids, attention_mask, out_ids)[0]
        return loss, predictions, references

    def inference(self, input_ids, attention_mask, out_ids, tokenizer, tgt_lang):
        # the only diff with generate() is that this one writes down predictions
        generated_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.args.inference_max_len,
            min_length=self.args.inference_min_len,
            num_beams=self.args.beam_size,
            no_repeat_ngram_size=self.args.no_repeat_ngram_size,
            repetition_penalty=self.args.repetition_penalty,
            length_penalty=self.args.length_penalty,
            early_stopping=True,
            decoder_start_token_id=tokenizer.lang_code_to_id[tgt_lang]
        )

        # Convert predicted and gold token ids to strings
        predictions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        references = tokenizer.batch_decode(out_ids, skip_special_tokens=True)

        self.rouge.add_batch(predictions=predictions, references=references)

        return predictions, references

class mBART_Adapter_Summarizer(nn.Module):
    def __init__(self, params, tokenizer) -> None:
        super().__init__()
        self.args = params
        self.tokenizer = tokenizer
        self.model = MBartAdapterModel.from_pretrained(self.args.model_type)
        self.model.add_seq2seq_lm_head('summarization')
        self.rouge = datasets.load_metric('rouge')

    def forward(self, input_ids, attention_mask, out_ids):
        # out_ids = batch_encoding['labels']
        # out_ids[out_ids[:, :] == self.tokenizer.pad_token_id] = -100
        # out_mask = torch.where(out_ids==0, 0, 1).unsqueeze(-1) # batch_size, out_len
        # the T5 model should creates its won decoder_attention_mask
        # https://huggingface.co/transformers/model_doc/t5.html#t5forconditionalgeneration

        # dec_inp_ids = model._shift_right(out_ids)
        # # no need above for T5, please see: https://huggingface.co/transformers/glossary.html#decoder-input-ids

        return self.model(input_ids=input_ids,
                          attention_mask=attention_mask,
                          labels=out_ids)

    def generate(self, input_ids, attention_mask, out_ids):
        generated_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.args.inference_max_len,
            min_length=self.args.inference_min_len,
            num_beams=self.args.beam_size,
            no_repeat_ngram_size=self.args.no_repeat_ngram_size,
            repetition_penalty=self.args.repetition_penalty,
            length_penalty=self.args.length_penalty,
            early_stopping=True,
            decoder_start_token_id=self.tokenizer.lang_code_to_id[self.args.tgt_lang]
        )

        # Convert predicted and gold token ids to strings
        predictions = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        references = self.tokenizer.batch_decode(out_ids, skip_special_tokens=True)

        self.rouge.add_batch(predictions=predictions, references=references)
        loss = self(input_ids, attention_mask, out_ids)[0]
        return loss, predictions, references

    def inference(self, input_ids, attention_mask, out_ids):
        # the only diff with generate() is that this one writes down predictions
        generated_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.args.inference_max_len,
            min_length=self.args.inference_min_len,
            num_beams=self.args.beam_size,
            no_repeat_ngram_size=self.args.no_repeat_ngram_size,
            repetition_penalty=self.args.repetition_penalty,
            length_penalty=self.args.length_penalty,
            early_stopping=True,
            decoder_start_token_id=self.tokenizer.lang_code_to_id[self.args.tgt_lang]
        )

        # Convert predicted and gold token ids to strings
        predictions = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        references = self.tokenizer.batch_decode(out_ids, skip_special_tokens=True)

        self.rouge.add_batch(predictions=predictions, references=references)

        return predictions, references

class Adapter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.adapter_down_project = nn.Linear(config.hidden_size, config.adapter_size)
        self.adapter_act_fn = gelu_new
        self.adapter_up_project = nn.Linear(config.adapter_size, config.hidden_size)

    def forward(self, x_BxIxH):
        y_BxIxh = self.adapter_down_project(x_BxIxH)
        y_BxIxh = self.adapter_act_fn(y_BxIxh)
        y_BxIxH = self.adapter_up_project(y_BxIxh)
        return x_BxIxH + y_BxIxH

    # This is copied from the BertPreTrainedModel class to make this a self containing class.
    @staticmethod
    def init_bert_weights(module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # std defaults to 0.02, this might need to be changed
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

class BART_Summarizer(nn.Module):
    def __init__(self, params, tokenizer) -> None:
        super().__init__()
        self.args = params
        self.tokenizer = tokenizer

        self.model = BARTOriginal.from_pretrained(self.args.model_type)
        if self.args.continue_training:
            # load pre-trained weights into model
            ckpt = torch.load(self.args.output_dir + self.args.checkpoint)
            self.model.load_state_dict(ckpt['state_dict'])

        self.rouge = datasets.load_metric('rouge')

    def forward(self, input_ids, attention_mask, out_ids):
        # out_ids = batch_encoding['labels']
        # out_ids[out_ids[:, :] == self.tokenizer.pad_token_id] = -100
        # out_mask = torch.where(out_ids==0, 0, 1).unsqueeze(-1) # batch_size, out_len
        # the T5 model should creates its won decoder_attention_mask
        # https://huggingface.co/transformers/model_doc/t5.html#t5forconditionalgeneration

        # dec_inp_ids = model._shift_right(out_ids)
        # # no need above for T5, please see: https://huggingface.co/transformers/glossary.html#decoder-input-ids

        return self.model(input_ids=input_ids,
                          attention_mask=attention_mask,
                          labels=out_ids)

    def generate(self, input_ids, attention_mask, out_ids):
        generated_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.args.inference_max_len,
            min_length=self.args.inference_min_len,
            num_beams=self.args.beam_size,
            no_repeat_ngram_size=self.args.no_repeat_ngram_size,
            repetition_penalty=self.args.repetition_penalty,
            length_penalty=self.args.length_penalty,
            early_stopping=True,
            # decoder_start_token_id=self.tokenizer.lang_code_to_id[self.args.tgt_lang]
        )

        # Convert predicted and gold token ids to strings
        predictions = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        references = self.tokenizer.batch_decode(out_ids, skip_special_tokens=True)

        self.rouge.add_batch(predictions=predictions, references=references)
        loss = self(input_ids, attention_mask, out_ids)[0]
        return loss, predictions, references

    def inference(self, input_ids, attention_mask, out_ids):
        # the only diff with generate() is that this one writes down predictions
        generated_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.args.inference_max_len,
            min_length=self.args.inference_min_len,
            num_beams=self.args.beam_size,
            no_repeat_ngram_size=self.args.no_repeat_ngram_size,
            repetition_penalty=self.args.repetition_penalty,
            length_penalty=self.args.length_penalty,
            early_stopping=True,
            # decoder_start_token_id=self.tokenizer.lang_code_to_id[self.args.tgt_lang]
        )

        # Convert predicted and gold token ids to strings
        predictions = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        references = self.tokenizer.batch_decode(out_ids, skip_special_tokens=True)

        self.rouge.add_batch(predictions=predictions, references=references)

        return predictions, references
