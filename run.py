from __future__ import absolute_import, division, print_function
import argparse
import logging
import os
import random
import math
import copy
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from autocsc import *


logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                    datefmt="%m/%d/%Y %H:%M:%S",
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):
    def __init__(self, guid, src, trg):
        self.guid = guid
        self.src = src
        self.trg = trg


class InputFeatures(object):
    def __init__(self, src_ids, attention_mask, trg_ids):
        self.src_ids = src_ids
        self.attention_mask = attention_mask
        self.trg_ids = trg_ids


class DataProcessor:
    """
    Processor for the data set.
    a) in a .tsv format, i.e. src\ttrg;
    b) separate Chinese characters from each other by spaces;
    c) without headlines.
    """

    def get_train_examples(self, data_dir, filename):
        return self._create_examples(self._read_csv(os.path.join(data_dir, filename)), "train")

    def get_dev_examples(self, data_dir, filename):
        return self._create_examples(self._read_csv(os.path.join(data_dir, filename)), "dev")

    def get_test_examples(self, data_dir, filename):
        return self._create_examples(self._read_csv(os.path.join(data_dir, filename)), "test")

    @staticmethod
    def _read_csv(input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            lines = []
            for line in f:
                src, trg = line.strip().split("\t")
                lines.append((src.split(), trg.split()))
            return lines

    @staticmethod
    def _create_examples(lines, set_type):
        examples = []
        for i, (src, trg) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            if len(src) == len(trg):
                examples.append(InputExample(guid=guid, src=src, trg=trg))
        return examples


def convert_examples_to_features(examples, max_seq_length, tokenizer):
    features = []
    for i, example in enumerate(examples):
        encoded_inputs = tokenizer(example.src,
                                   max_length=max_seq_length,
                                   padding="max_length",
                                   truncation=True,
                                   return_token_type_ids=True,
                                   is_split_into_words=True)

        trg_ids = tokenizer(example.trg,
                            max_length=max_seq_length,
                            padding="max_length",
                            truncation=True,
                            return_token_type_ids=True,
                            is_split_into_words=True)["input_ids"]

        src_ids = encoded_inputs["input_ids"]
        attention_mask = encoded_inputs["attention_mask"]

        assert len(src_ids) == max_seq_length
        assert len(attention_mask) == max_seq_length
        assert len(trg_ids) == max_seq_length

        if i < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("src_tokens: %s" % " ".join(example.src))
            logger.info("trg_tokens: %s" % " ".join(example.trg))
            logger.info("src_ids: %s" % " ".join([str(x) for x in src_ids]))
            logger.info("trg_ids: %s" % " ".join([str(x) for x in trg_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))

        features.append(
                InputFeatures(src_ids=src_ids,
                              attention_mask=attention_mask,
                              trg_ids=trg_ids)
        )
    return features


class Metrics:
    @staticmethod
    def compute(src_sents, trg_sents, prd_sents):
        def difference(src, trg):
            ret = copy.deepcopy(src)
            for i, (src_char, trg_char) in enumerate(zip(src, trg)):
                if src_char!= trg_char:
                    ret[i] = "(" + src_char + "->" + trg_char + ")"

            return "".join(ret)

        pos_sents, neg_sents, tp_sents, fp_sents, fn_sents, prd_pos_sents, prd_neg_sents = [], [], [], [], [], [], []
        for s, t, p in zip(src_sents, trg_sents, prd_sents):
            if s != t:
                pos_sents.append(difference(s, t))
                if p == t:
                    tp_sents.append(difference(s, t))
                if p == s:
                    fn_sents.append(difference(s, t))

            else:
                neg_sents.append(difference(s, t))
                if p != t:
                    fp_sents.append(difference(t, p))

            if s != p:
                prd_pos_sents.append(difference(s, p))
            if s == p:
                prd_neg_sents.append(difference(s, p))

        p = 1.0 * len(tp_sents) / len(prd_pos_sents)
        r = 1.0 * len(tp_sents) / len(pos_sents)
        f1 = 2.0 * (p * r) / (p + r + 1e-12)
        fpr = 1.0 * (len(fp_sents) + 1e-12) / (len(neg_sents) + 1e-12)

        return p, r, f1, fpr, tp_sents, fp_sents, fn_sents


def mask_tokens(inputs, tokenizer, noise_probability=0.2):
    inputs = inputs.clone()
    probability_matrix = torch.full(inputs.shape, noise_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in inputs.tolist()
    ]
    special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)

    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    inputs[masked_indices] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    return inputs


def mask_tokens_only_neg(inputs, labels, tokenizer, noise_probability=0.2):
    inputs = inputs.clone()
    labels = labels.clone()
    probability_matrix = torch.full(inputs.shape, noise_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in inputs.tolist()
    ]
    special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
    neq_tokens_mask = (inputs != labels).cpu()

    probability_matrix.masked_fill_(special_tokens_mask + neq_tokens_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    inputs[masked_indices] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    return inputs


def main():
    parser = argparse.ArgumentParser()

    # Data config
    parser.add_argument("--data_dir", type=str, default="../data/csc/",
                        help="Directory to contain the input data for all tasks.")
    parser.add_argument("--train_on", type=str, default="",
                        help="Specify a training set.")
    parser.add_argument("--eval_on", type=str, default="",
                        help="Specify a dev set.")
    parser.add_argument("--test_on", type=str, default="",
                        help="Specify a test set.")
    parser.add_argument("--load_model_path", type=str, default="bert-base-chinese",
                        help="Pre-trained model path to load.")
    parser.add_argument("--model_type", type=str, default="finetune",
                        help="Model architecture to load.")
    parser.add_argument("--cache_dir", type=str, default="../cache/",
                        help="Directory to store the pre-trained language models downloaded from s3.")
    parser.add_argument("--output_dir", type=str, default="model/",
                        help="Directory to output predictions and checkpoints.")
    parser.add_argument("--load_state_dict", type=str, default="",
                        help="Trained model weights to load for evaluation if needed.")

    # Training config
    parser.add_argument("--do_train", action="store_true",
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true",
                        help="Whether to evaluate on the dev set.")
    parser.add_argument("--do_test", action="store_true",
                        help="Whether to evaluate on the test set.")
    parser.add_argument("--use_slow_tokenizer", action="store_true",
                        help="A slow tokenizer will be used if passed.")
    parser.add_argument("--do_lower_case", action="store_true",
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--max_seq_length", type=int, default=64,
                        help="Maximum total input sequence length after word-piece tokenization.")
    parser.add_argument("--train_batch_size", type=int, default=128,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", type=int, default=512,
                        help="Total batch size for evaluation.")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", type=float, default=3.0,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_train_steps", type=int, default=None,
                        help="Total number of training steps to perform. If provided, overrides training epochs.")
    parser.add_argument("--weight_decay", type=float, default=0.,
                        help="L2 weight decay for training.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward pass.")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Whether not to use CUDA when available.")
    parser.add_argument("--fp16", action="store_true",
                        help="Whether to use mixed precision.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for initialization.")
    parser.add_argument("--save_steps", type=int, default=100,
                        help="How many steps to save the checkpoint once.")
    parser.add_argument("--mft", action="store_true",
                        help="Training with masked-fine-tuning.")

    args = parser.parse_args()

    AutoCSC = {
            "finetune": AutoCSCfinetune,
            "softmasked": AutoCSCSoftMasked,
            "mdcspell": AutoCSCMDCSpell,
        }

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, "Unsupported", args.fp16))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.do_train:
        torch.save(args, os.path.join(args.output_dir, "train_args.bin"))

    processor = DataProcessor()

    cache_dir = args.cache_dir
    tokenizer = AutoTokenizer.from_pretrained(args.load_model_path,
                                              do_lower_case=args.do_lower_case,
                                              cache_dir=cache_dir,
                                              use_fast=not args.use_slow_tokenizer,
                                              add_prefix_space=True)

    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir, args.train_on)
        train_features = convert_examples_to_features(train_examples, args.max_seq_length, tokenizer)

        all_input_ids = torch.tensor([f.src_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.attention_mask for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.trg_ids for f in train_features], dtype=torch.long)

        train_data = TensorDataset(all_input_ids, all_input_mask, all_label_ids)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        if args.max_train_steps is None:
            args.max_train_steps = int(args.num_train_epochs * num_update_steps_per_epoch)
        else:
            args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

        model = AutoCSC[args.model_type].from_pretrained(args.load_model_path,
                                                         cache_dir=cache_dir)
        model.to(device)
        if args.load_state_dict:
            model.load_state_dict(torch.load(args.load_state_dict))
        if n_gpu > 1:
            model = torch.nn.DataParallel(model)

        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0
            }
        ]

        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

        scaler = None
        if args.fp16:
            from torch.cuda.amp import autocast, GradScaler

            scaler = GradScaler()
        
        if args.do_eval:
            eval_examples = processor.get_dev_examples(args.data_dir, args.eval_on)
            eval_features = convert_examples_to_features(eval_examples, args.max_seq_length, tokenizer)

            all_input_ids = torch.tensor([f.src_ids for f in eval_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.attention_mask for f in eval_features], dtype=torch.long)
            all_label_ids = torch.tensor([f.trg_ids for f in eval_features], dtype=torch.long)

            eval_data = TensorDataset(all_input_ids, all_input_mask, all_label_ids)
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    if args.do_train:
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", args.max_train_steps)

        global_step = 0
        best_result = list()
        wrap = False
        progress_bar = tqdm(range(args.max_train_steps))
        for _ in progress_bar:
            train_loss = 0
            num_train_examples = 0
            if wrap: break
            for step, batch in enumerate(train_dataloader):
                model.train()
                batch = tuple(t.to(device) for t in batch)
                src_ids, attention_mask, trg_ids = batch
                if args.mft:
                    src_ids = mask_tokens_only_neg(src_ids, trg_ids, tokenizer)

                if args.fp16:
                    with autocast():
                        outputs = model(src_ids=src_ids,
                                        attention_mask=attention_mask,
                                        trg_ids=trg_ids)
                else:
                    outputs = model(src_ids=src_ids,
                                    attention_mask=attention_mask,
                                    trg_ids=trg_ids)
                loss = outputs["loss"]

                if n_gpu > 1:
                    loss = loss.mean()
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                if args.fp16:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                train_loss += loss.item()
                num_train_examples += src_ids.size(0)
                if (step + 1) % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    if args.fp16:
                        scaler.unscale_(optimizer)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
                    progress_bar.update(1)

                if args.do_eval and global_step % args.save_steps == 0:
                    logger.info("***** Running evaluation *****")
                    logger.info("  Num examples = %d", len(eval_examples))
                    logger.info("  Batch size = %d", args.eval_batch_size)

                    def decode(input_ids):
                        return tokenizer.convert_ids_to_tokens(input_ids, skip_special_tokens=True)

                    model.eval()
                    all_inputs, all_labels, all_predictions = [], [], []
                    for batch in tqdm(eval_dataloader):
                        batch = tuple(t.to(device) for t in batch)
                        src_ids, attention_mask, trg_ids = batch
                        with torch.no_grad():
                            outputs = model(src_ids=src_ids,
                                            attention_mask=attention_mask,
                                            trg_ids=trg_ids)
                            prd_ids = outputs["predict_ids"].tolist()

                        for s, t, p in zip(src_ids, trg_ids, prd_ids):
                            all_inputs += [decode(s)]
                            all_labels += [decode(t)]
                            all_predictions += [decode(p)]
    
                    loss = train_loss / global_step
                    p, r, f1, fpr, tp, fp, fn = Metrics.compute(all_inputs, all_labels, all_predictions)
    
                    output_tp_file = os.path.join(args.output_dir, "sents.tp")
                    with open(output_tp_file, "w") as writer:
                        for line in tp:
                            writer.write(line + "\n")
                    output_fp_file = os.path.join(args.output_dir, "sents.fp")
                    with open(output_fp_file, "w") as writer:
                        for line in fp:
                            writer.write(line + "\n")
                    output_fn_file = os.path.join(args.output_dir, "sents.fn")
                    with open(output_fn_file, "w") as writer:
                        for line in fn:
                            writer.write(line + "\n")

                    result = {
                        "global_step": global_step,
                        "loss": loss,
                        "eval_p": p * 100,
                        "eval_r": r * 100,
                        "eval_f1": f1 * 100,
                        "eval_fpr": fpr * 100,
                    }
                    model_to_save = model.module if hasattr(model, "module") else model
                    output_model_file = os.path.join(args.output_dir, "step-%s_f1-%.2f.bin" % (str(global_step), result["eval_f1"]))
                    torch.save(model_to_save.state_dict(), output_model_file)
                    best_result.append((result["eval_f1"], output_model_file))
                    best_result.sort(key=lambda x: x[0], reverse=True)
                    if len(best_result) > 3:
                        _, model_to_remove = best_result.pop()
                        os.remove(model_to_remove)

                    output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
                    with open(output_eval_file, "a") as writer:
                        logger.info("***** Eval results *****")
                        writer.write(
                            "Global step = %s | eval precision = %.2f | eval recall = %.2f | eval f1 = %.2f | eval fp rate = %.2f\n"
                            % (str(result["global_step"]),
                            result["eval_p"],
                            result["eval_r"],
                            result["eval_f1"],
                            result["eval_fpr"]))
                        for key in sorted(result.keys()):
                            logger.info("Global step: %s,  %s = %s", str(global_step), key, str(result[key]))

                if global_step >= args.max_train_steps:
                    wrap = True
                    break

    if args.do_test:
        eval_examples = processor.get_test_examples(args.data_dir, args.test_on)
        eval_features = convert_examples_to_features(eval_examples, args.max_seq_length, tokenizer)

        all_input_ids = torch.tensor([f.src_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.attention_mask for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.trg_ids for f in eval_features], dtype=torch.long)

        eval_data = TensorDataset(all_input_ids, all_input_mask, all_label_ids)
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        model = AutoCSC[args.model_type].from_pretrained(args.load_model_path,
                                                         cache_dir=cache_dir)
        model.to(device)
        if args.load_state_dict:
            model.load_state_dict(torch.load(args.load_state_dict))
        if n_gpu > 1:
            model = torch.nn.DataParallel(model)

        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)

        def decode(input_ids):
            return tokenizer.convert_ids_to_tokens(input_ids, skip_special_tokens=True)

        model.eval()
        eval_loss = 0
        eval_steps = 0
        all_inputs, all_labels, all_predictions = [], [], []
        for batch in tqdm(eval_dataloader):
            batch = tuple(t.to(device) for t in batch)
            src_ids, attention_mask, trg_ids = batch
            with torch.no_grad():
                outputs = model(src_ids=src_ids,
                                attention_mask=attention_mask,
                                trg_ids=trg_ids)
                tmp_eval_loss = outputs["loss"]
                prd_ids = outputs["predict_ids"].tolist()

            eval_loss += tmp_eval_loss.mean().item()
            for s, t, p in zip(src_ids, trg_ids, prd_ids):
                all_inputs += [decode(s)]
                all_labels += [decode(t)]
                all_predictions += [decode(p)]
            eval_steps += 1

        eval_loss = eval_loss / eval_steps
        p, r, f1, fpr, tp, fp, fn = Metrics.compute(all_inputs, all_labels, all_predictions)

        output_tp_file = os.path.join(args.output_dir, "sents.tp")
        with open(output_tp_file, "w") as writer:
            for line in tp:
                writer.write(line + "\n")
        output_fp_file = os.path.join(args.output_dir, "sents.fp")
        with open(output_fp_file, "w") as writer:
            for line in fp:
                writer.write(line + "\n")
        output_fn_file = os.path.join(args.output_dir, "sents.fn")
        with open(output_fn_file, "w") as writer:
            for line in fn:
                writer.write(line + "\n")

        result = {
            "eval_loss": eval_loss,
            "eval_p": p * 100,
            "eval_r": r * 100,
            "eval_f1": f1 * 100,
            "eval_fpr": fpr * 100,
        }

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "a") as writer:
            logger.info("***** Eval results *****")
            writer.write(
                "Global step = %s | eval precision = %.2f | eval recall = %.2f | eval f1 = %.2f | eval fp rate = %.2f\n"
                % (str(-1),
                result["eval_p"],
                result["eval_r"],
                result["eval_f1"],
                result["eval_fpr"]))
        for key in sorted(result.keys()):
            logger.info("Global step: %s,  %s = %s", str(-1), key, str(result[key]))


if __name__ == "__main__":
    main()
