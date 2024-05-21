from __future__ import absolute_import, division, print_function
import argparse
import json
import logging
import os
import random
import copy
import math
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
import sklearn.metrics as mtc
from scipy.stats import spearmanr
from tqdm import tqdm, trange
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import get_scheduler
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from accelerate import Accelerator, DeepSpeedPlugin

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                    datefmt="%m/%d/%Y %H:%M:%S",
                    level=logging.INFO)
logger = logging.getLogger(__name__)

detection_instructions = [
    "用<>标注出句子中的错别字",
]

class InputExample(object):
    def __init__(self, guid, context_prefix, context_suffix, src, trg, src_dev, instruction_type='correct'):
        '''
        instruction type: {detect, correct}
        '''
        self.guid = guid
        self.context_prefix = context_prefix
        self.context_suffix = context_suffix
        self.src = src
        self.trg = trg
        self.src_dev=src_dev,
        self.instruction_type = instruction_type


class InputFeatures(object):
    def __init__(self, input_ids, attention_mask, labels, src_ref):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels
        self.src_ref = src_ref


class DataProcessor:

    def get_train_examples(self, input_file, demon=False):
        return self._create_examples(
            self._read_jsonl(os.path.join(input_file)), "train", demonstration=demon)

    def get_dev_examples(self, input_file, demon=False):
        return self._create_examples(
            self._read_jsonl(os.path.join(input_file)), "dev", demonstration=demon)
    
    @staticmethod
    def _create_examples(lines, set_type, demonstration=False):
        examples = []
        for (i, line) in enumerate(lines):
            instruction_type = 'correct'
            if line["instruction"] in detection_instructions:
                instruction_type = 'detect'
            guid = "%s-%s" % (set_type, i)
            context_prefix = "Instruction:\n{instruction}\n\nInput:\n".format_map(line)
            start = line['input']
            context_suffix = "\n\nResponse:\n"
            end = line["response"]
            src_dev = context_prefix + start + context_suffix

            examples.append(
                InputExample(guid=guid, context_prefix=context_prefix, context_suffix=context_suffix, \
                             src=start, trg=end, src_dev=src_dev, instruction_type=instruction_type))

        return examples

    @classmethod
    def _read_jsonl(cls, input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            lines = []
            for line in f:
                lines.append(json.loads(line.strip()))
            return lines


class DataProcessorForRephrasing(DataProcessor):
    @staticmethod
    def convert_examples_to_features(examples, max_seq_length, tokenizer, verbose=True, model_type="baichuan"):
        '''
        set max_seq_length to 256 for ecspell, avoid truncating the input
        for baichuan:
        self.user_tokens = 195
        self.assistant_tokens = 196
        mask_token = 190
        self.ignore_index = -100

        for qwen:
        no pad, eos, bos、eos、unk、pad、mask、sep.
        Here we set pad_token as '<|endoftext|>' with id 151643.
        '''
        features = []
        for i, example in enumerate(examples):
            #example.context_prefix + example.src + example.context_suffix
            prefix_ids = tokenizer(example.context_prefix,
                                   max_length=max_seq_length // 2 - 2,
                                   truncation=True,
                                   is_split_into_words=False,
                                   add_special_tokens=False).input_ids
            src_ids = tokenizer(example.src,
                                max_length=max_seq_length // 2 - 2,
                                truncation=True,
                                is_split_into_words=False,
                                add_special_tokens=False).input_ids
            suffix_ids = tokenizer(example.context_suffix,
                                   max_length=max_seq_length // 2 - 2,
                                   truncation=True,
                                   is_split_into_words=False,
                                   add_special_tokens=False).input_ids
            trg_ids = tokenizer(example.trg,
                                max_length=max_seq_length // 2 - 2,
                                truncation=True,
                                is_split_into_words=False,
                                add_special_tokens=False).input_ids
           # if 0:
            src_all = prefix_ids + src_ids + suffix_ids
            if model_type == "baichuan":
                input_ids = [195] + src_all + [196] + trg_ids + [tokenizer.eos_token_id]
                if example.instruction_type == 'detect':
                    src_ref = [195] + [-100 for _ in src_all] + [196] + [-100 for _ in trg_ids] + [tokenizer.eos_token_id]
                else:
                    src_ref = [195] + [-100 for _ in prefix_ids] + src_ids + [-100 for _ in suffix_ids] + [196] \
                        + [-100 for _ in trg_ids] + [tokenizer.eos_token_id]
                label_ids = [-100] + [-100 for _ in src_all] + [-100] + trg_ids + [tokenizer.eos_token_id]
            else:
                assert model_type == "qwen"
                input_ids = src_all + trg_ids + [tokenizer.eos_token_id]
                if example.instruction_type == 'detect':
                    src_ref = [-100 for _ in src_all] + [-100 for _ in trg_ids]
                else:
                    src_ref =  [-100 for _ in prefix_ids] + src_ids + [-100 for _ in suffix_ids] \
                        + [-100 for _ in trg_ids] + [tokenizer.eos_token_id]
                label_ids = [-100 for _ in src_all] + trg_ids + [tokenizer.eos_token_id]
            '''
            else:
                start = [tokenizer.bos_token_id] + tokenizer.convert_tokens_to_ids(tokenizer.tokenize("Human: "))
                end = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("Assistant: "))
                input_ids = start + src_ids + end + trg_ids + [tokenizer.eos_token_id]
                label_ids = [-100] * len(start) + [-100 for _ in src_ids] + [-100] * len(end) + trg_ids + [tokenizer.eos_token_id]
            '''
            attention_mask = [1] * len(input_ids)

            offset_length = max_seq_length - len(input_ids)
            if offset_length > 0:
                input_ids = input_ids + [tokenizer.pad_token_id] * offset_length
                src_ref = src_ref + [tokenizer.pad_token_id] * offset_length
                attention_mask = attention_mask + [0] * offset_length
                label_ids = label_ids + [-100] * offset_length
            input_ids, src_ref, attention_mask, label_ids = input_ids[:max_seq_length], src_ref[:max_seq_length], attention_mask[:max_seq_length], label_ids[:max_seq_length]

            assert len(input_ids) == max_seq_length
            assert len(attention_mask) == max_seq_length
            assert len(label_ids) == max_seq_length
            assert len(src_ref) == max_seq_length

            if verbose and i < 5:
                logger.info("*** Example ***")
                logger.info("guid: %s" % example.guid)
                logger.info("src_tokens: %s" % " ".join(tokenizer.decode(input_ids)))
                logger.info("src_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info("src_ref: %s" % " ".join([str(x) for x in src_ref]))
                logger.info("trg_ids: %s" % " ".join([str(x) for x in label_ids]))
                logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))

            features.append(
                    InputFeatures(input_ids=input_ids,
                                  src_ref = src_ref,
                                  attention_mask=attention_mask,
                                  labels=label_ids)
            )
        return features
    
class Metrics:
    @staticmethod
    def csc_compute(src_sents, trg_sents, prd_sents, simple=False):
        def difference(src, trg, simple=False):
            if not simple:
                ret = copy.deepcopy(src)
                for i, (src_char, trg_char) in enumerate(zip(src, trg)):
                    if src_char!= trg_char:
                        ret[i] = "(" + src_char + "->" + trg_char + ")"
            else:
                return "->".join(["".join(src), "".join(trg)])
            return "".join(ret)

        pos_sents, neg_sents, tp_sents, fp_sents, fn_sents, prd_pos_sents, prd_neg_sents, wp_sents = [], [], [], [], [], [], [], []
        for s, t, p in zip(src_sents, trg_sents, prd_sents):
            # For positive examples
            if s != t:
                pos_sents.append(difference(s, t, simple))
                #print(difference(s, t))
                if p == t:
                    tp_sents.append(difference(s, t,simple))
                if p == s:
                    fn_sents.append(difference(s, t, simple))
                if (p!=t and p!=s):
                    wp_sents.append(difference(s,p, simple))
            # For negative examples
            else:
                neg_sents.append(difference(s, t, simple))
                if p != t:
                    fp_sents.append(difference(t, p, simple))
            # For predictions
            if s != p:
                prd_pos_sents.append(difference(s, p, simple))
            if s == p:
                prd_neg_sents.append(difference(s, p, simple))
        if len(pos_sents)==0:
            p=0
            r=0
            f1=0
            wpr=0
        else:
            p = 1.0 * len(tp_sents) / len(prd_pos_sents)
            r = 1.0 * len(tp_sents) / len(pos_sents)
            f1 = 2.0 * (p * r) / (p + r + 1e-12)
            wpr = 1.0 * len(wp_sents) / len(pos_sents)
        fpr = 1.0 * (len(fp_sents) + 1e-12) / (len(neg_sents) + 1e-12)

        return p, r, f1, fpr, wpr, tp_sents, fp_sents, fn_sents, wp_sents


def mask_tokens(inputs, src_ref, tokenizer, noise_probability=0.3):
    '''
    we accpet masking the error tokens in the input
    input_ids:  prefix + src + suffix + trg  + pad
    src_ref: -100... + src + -100... + ...-100... + pad
    '''
    device = src_ref.device
    inputs = inputs.clone()
    probability_matrix = torch.full(inputs.shape, noise_probability).to(device)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in inputs.tolist()
    ]
    special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool).to(device)
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

    ## do not mask target part and the error tokens in src part
    probability_matrix.masked_fill_(inputs!=src_ref, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    inputs[masked_indices] = 190

    return inputs



def main():
    parser = argparse.ArgumentParser()

    # Data config
    parser.add_argument("--data_dir", type=str, default="data/",
                        help="Directory to contain the input data for all tasks.")
    parser.add_argument("--load_model_path", type=str, default="meta-llama/Llama-2-7b-hf",
                        help="Pre-trained language model to load.")
    parser.add_argument("--cache_dir", type=str, default="../../cache/",
                        help="Directory to store the pre-trained language models downloaded from s3.")
    parser.add_argument("--output_dir", type=str, default="model/",
                        help="Directory to output predictions and checkpoints.")
    parser.add_argument("--load_ckpt", type=str, default="",
                        help="Checkpoint to load for trianing or evaluation.")

    # Training config
    parser.add_argument("--do_train", action="store_true",
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true",
                        help="Whether to evaluate on the dev set.")
    parser.add_argument("--train_on", type=str, default="",
                        help="Choose a training set.")
    parser.add_argument("--eval_on", type=str, default="",
                        help="Choose a dev set.")
    parser.add_argument("--noise_probability", type=float, default=0.2,
                        help="Mask rate for masked-fine-tuning.")
    parser.add_argument("--mft", action="store_true",
                        help="Training with masked-fine-tuning.")
    parser.add_argument("--do_lower_case", action="store_true",
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--max_seq_length", type=int, default=128,
                        help="Maximum total input sequence length after word-piece tokenization.")
    parser.add_argument("--train_batch_size", type=int, default=128,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", type=int, default=256,
                        help="Total batch size for evaluation.")
    parser.add_argument("--learning_rate", type=float, default=3e-5,
                        help="Peak learning rate for optimization.")
    parser.add_argument("--num_train_epochs", type=float, default=3.0,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_train_steps", type=int, default=None,
                        help="Total number of training steps to perform (overrides training epochs).")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine",
                        help="Scheduler type for learning rate warmup.")
    parser.add_argument("--warmup_proportion", type=float, default=0.06,
                        help="Proportion of training to perform learning rate warmup for.")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="L2 weight decay for training.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward pass.")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Whether not to use CUDA when available.")
    parser.add_argument("--fp16", action="store_true",
                        help="Whether to use mixed precision.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for initialization.")
    parser.add_argument("--lora", action="store_true",
                        help="Whether to use low rank adaption.")
    parser.add_argument("--deepspeed", action="store_true",
                        help="Whether to use DeepSpeed.")

    parser.add_argument("--demon", action="store_true",)
    parser.add_argument("--model_type", type=str, default="baichuan", choices=["baichuan", "qwen"],)

    args = parser.parse_args()


    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
    logger.info("device: {}, n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, "-accelerate", args.fp16))

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

    processor = DataProcessorForRephrasing()

    cache_dir = args.cache_dir
    if args.model_type=='baichuan':
        tokenizer = AutoTokenizer.from_pretrained(args.load_model_path,
                                                trust_remote_code=True,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=cache_dir,
                                                )
        if getattr(tokenizer, "pad_token_id") is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
    else:
        assert args.model_type=='qwen'
        tokenizer = AutoTokenizer.from_pretrained(args.load_model_path,
                                            trust_remote_code=True,
                                            do_lower_case = args.do_lower_case,
                                            cache_dir=cache_dir,
                                            pad_token='<|endoftext|>',
                                            )
        tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids('<|endoftext|>')


    if args.do_train:
        if args.deepspeed:
            with open("deepspeed_configs/zero3_config.json") as f:
                ds_config = json.load(f)
            ds_plugin = DeepSpeedPlugin(hf_ds_config=ds_config)
            ds_plugin.hf_ds_config.config["train_micro_batch_size_per_gpu"] = args.train_batch_size

            accelerator = Accelerator(cpu=args.no_cuda, mixed_precision="fp16" if args.fp16 else "no", deepspeed_plugin=ds_plugin)
        else:
            accelerator = Accelerator(cpu=args.no_cuda, mixed_precision="fp16" if args.fp16 else "no")
        device = accelerator.device

        train_examples = processor.get_train_examples(os.path.join(args.data_dir, args.train_on), demon=args.demon)
        train_features = processor.convert_examples_to_features(train_examples, args.max_seq_length, tokenizer, model_type=args.model_type)

        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_src_ref = torch.tensor([f.src_ref for f in train_features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in train_features], dtype=torch.long)
        all_labels = torch.tensor([f.labels for f in train_features], dtype=torch.long)

        train_data = TensorDataset(all_input_ids, all_src_ref, all_attention_mask, all_labels)
        train_dataloader = DataLoader(train_data, shuffle=True, batch_size=args.train_batch_size)
        train_dataloader = accelerator.prepare(train_dataloader)

        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        if args.max_train_steps is None:
            args.max_train_steps = int(args.num_train_epochs * num_update_steps_per_epoch)
        else:
            args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

        model = AutoModelForCausalLM.from_pretrained(args.load_model_path,
                                                     cache_dir=cache_dir,
                                                     trust_remote_code=True)

        if args.lora:
            if args.load_ckpt:
                model = PeftModel.from_pretrained(model, args.load_ckpt, is_trainable=True)
            else:
                if args.model_type=="baichuan":
                    # peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=32, lora_dropout=0.1, target_modules=["W_pack"])
                    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=32, lora_dropout=0.1, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"])
                else:
                    assert args.model_type=="qwen"
                    peft_config = LoraConfig(
                        r=64,
                        lora_alpha=16,
                        target_modules=["c_attn", "c_proj", "w1", "w2"],
                        lora_dropout=0.05,
                        bias="none",
                        task_type="CAUSAL_LM",
                    )
                model = get_peft_model(model, peft_config)
                model.print_trainable_parameters()

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
        scheduler = get_scheduler(name=args.lr_scheduler_type,
                                  optimizer=optimizer,
                                  num_warmup_steps=30,
                                  num_training_steps=args.max_train_steps)

        model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

        if args.do_eval:
            eval_examples = processor.get_dev_examples(os.path.join(args.data_dir, args.eval_on))
            eval_features = processor.convert_examples_to_features(eval_examples, args.max_seq_length, tokenizer, model_type=args.model_type)

            all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
            all_attention_mask = torch.tensor([f.attention_mask for f in eval_features], dtype=torch.long)
            all_labels = torch.tensor([f.labels for f in eval_features], dtype=torch.long)

            eval_data = TensorDataset(all_input_ids, all_attention_mask, all_labels)
            eval_dataloader = DataLoader(eval_data, shuffle=False, batch_size=args.eval_batch_size)
            eval_dataloader = accelerator.prepare(eval_dataloader)

        #model, optimizer, scheduler, train_dataloader, eval_dataloader = accelerator.prepare(model, optimizer, scheduler, train_dataloader, eval_dataloader)

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size * accelerator.num_processes)
        logger.info("  Num steps = %d", args.max_train_steps)

        global_step = 0
        best_epoch = 0
        best_result = 0.0
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            model.train()
            train_loss = 0
            num_train_examples = 0
            train_steps = 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration", leave=True)):
                batch = tuple(t.to(device) for t in batch)
                input_ids, src_ref, attention_mask, labels = batch
                if args.mft:
                    input_ids = mask_tokens(input_ids, src_ref, tokenizer, args.noise_probability)
                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                labels=labels)
                loss = outputs.loss

                if n_gpu > 1:
                    loss = loss.mean()
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                accelerator.backward(loss)

                train_loss += loss.item()
                num_train_examples += input_ids.size(0)
                train_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1

                if global_step >= args.max_train_steps:
                    break

            model_to_save = model.module if hasattr(model, "module") else model
            output_model_file = os.path.join(args.output_dir, "checkpoint_ep-{}".format(epoch))
            if accelerator.is_local_main_process and (epoch + 1) % 1 == 0:
                model_to_save.save_pretrained(output_model_file)

            if args.do_eval:
                logger.info("***** Running evaluation *****")
                logger.info("  Num examples = %d", len(eval_examples))
                logger.info("  Batch size = %d", args.eval_batch_size)

                model.eval()
                all_predictions, all_labels = [], []
                for batch in tqdm(eval_dataloader, desc="Evaluation"):
                    batch = tuple(t.to(device) for t in batch)
                    input_ids, attention_mask, labels = batch
                    with torch.no_grad():
                        outputs = model(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        labels=labels)
                        logits = outputs[1]
                        shift_logits = logits[..., :-1, :].contiguous()
                        shift_labels = labels[..., 1:].contiguous()

                    shift_predictions, shift_labels = accelerator.gather_for_metrics((shift_logits.argmax(dim=-1), shift_labels))
                    predictions, labels = shift_predictions[torch.where(shift_labels > 0)].to("cpu").numpy(), shift_labels[torch.where(shift_labels > 0)].to("cpu").numpy()
                    all_predictions.extend(predictions.squeeze().tolist())
                    all_labels.extend(labels.squeeze().tolist())

                output_predict_file = os.path.join(args.output_dir, "predict_results.txt")

                def decode_acc():
                    slacc = tlacc = 0
                    n = m = 0
                    tmp_prediction = []
                    tmp_label = []
                    with open(output_predict_file, "w") as writer:
                        for p, l in zip(all_predictions, all_labels):
                            if l == tokenizer.eos_token_id:
                                writer.write(" -> ".join([tokenizer.decode(tmp_label), tokenizer.decode(tmp_prediction)]) + "\n")
                                if tmp_prediction == tmp_label:
                                    slacc += 1
                                n += 1
                                tlacc += sum([int(c == d) for c, d in zip(tmp_prediction, tmp_label)])
                                m += len(tmp_label)
                                del tmp_prediction[:]
                                del tmp_label[:]
                            else:
                                tmp_prediction += [p]
                                tmp_label += [l]
                    return slacc / n, tlacc / m

                train_epoch_loss = train_loss / len(train_dataloader)
                train_ppl = math.exp(train_epoch_loss)
                acc1, acc2 = decode_acc()

                result = {
                    "global_step": global_step,
                    "train_ppl": train_ppl,
                    "eval_acc": acc1 * 100,
                    "eval_token_acc": acc2 * 100
                }
                if result["eval_acc"] > best_result:
                    best_epoch = epoch
                    best_result = result["eval_acc"]

                if accelerator.is_local_main_process:
                    output_eval_file = os.path.join(args.output_dir, "eval_results.txt")

                    def printf():
                        with open(output_eval_file, "a") as writer:
                            writer.write(
                                "Epoch %s: global step = %s | train ppl = %.3f | setence acc = %.2f | token acc = %.2f\n"
                                % (str(epoch),
                                str(result["global_step"]),
                                result["train_ppl"],
                                result["eval_acc"],
                                result["eval_token_acc"]))

                    printf()
                    for key in sorted(result.keys()):
                        logger.info("Epoch: %s,  %s = %s", str(epoch), key, str(result[key]))
                    logger.info("Best epoch: %s, result:  %s", str(best_epoch), str(best_result))


if __name__ == "__main__":
    main()
