from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import json
import copy
from tqdm import tqdm
import logging
import os
import argparse

class Metrics:
    @staticmethod
    def compute(src_sents, trg_sents, prd_sents, simple=False):
        def difference(src, trg, simple=False, note=None):
            if not simple:
                ret = copy.deepcopy(src)
                for i, (src_char, trg_char) in enumerate(zip(src, trg)):
                    if src_char!= trg_char:
                        ret[i] = "(" + src_char + "->" + trg_char + ")"
            else:
                if note is not None:
                    return "->\n".join(["".join(src), "".join(trg)])+"\n"+"".join(note)+"\n"
                else:
                    return "->\n".join(["".join(src), "".join(trg)])+"\n"
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
                    wp_sents.append(difference(s,p, simple, note=t))
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/ecspell",
                        help="Directory to contain the input data for all tasks.")
    parser.add_argument("--load_model_path", type=str, default="../../cache/models--baichuan-inc--Baichuan2-7B-Base/snapshots/66875f9e5d56275ab7a961fe12f1af3e84ac3feb",
                        help="Pre-trained language model to load.")
    parser.add_argument("--cache_dir", type=str, default="../../cache/",
                        help="Directory to store the pre-trained language models downloaded from s3.")
    parser.add_argument("--output_dir", type=str, default="model/med/det_natural_confus_3",
                        help="Directory to output predictions and checkpoints.")
    parser.add_argument("--load_ckpt", type=str, default="",
                        help="Checkpoint to load for trianing or evaluation.")
    parser.add_argument("--seed", type=int, default=42,)
    parser.add_argument("--no_cuda", action="store_true",
                        help="Whether not to use CUDA when available.")

    parser.add_argument("--test_on", type=str, default="test_med_det_rephrase.jsonl", help="test on which dataset")
    parser.add_argument("--model_type", type=str, default="baichuan", choices=["baichuan", "qwen"],)
    parser.add_argument("--result_file", type=str, default="eval_results.txt",)
    parser.add_argument("--max_seq_length", type=int, default=128,)
    parser.add_argument("--eval_batch_size", type=int, default=8,)

    parser.add_argument("--response_file", type=str, default=None,)

    args = parser.parse_args()
    # Set up logging
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                    datefmt="%m/%d/%Y %H:%M:%S",
                    level=logging.INFO)
    logger = logging.getLogger(__name__)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
    logger.info("device: {} n_gpu: {}".format(
        device, n_gpu))

    # create output directory if not exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # load evaluation data
    processor = DataProcessor()
    examples = processor.get_dev_examples(os.path.join(args.data_dir, args.test_on))
    all_inputs, all_labels, all_predictions = [], [], []
    for example in examples:
        all_inputs.append(example.context_prefix+example.src+example.context_suffix)
        all_labels.append(example.trg)
    for i in range(5):
        logger.info("src: %s \n" % all_inputs[i])
        logger.info("trg: %s \n" % all_labels[i])

    cache_dir = args.cache_dir
    # load tokenizer
    if args.model_type=='baichuan':
        # right padding for baichaun !
        tokenizer = AutoTokenizer.from_pretrained(args.load_model_path,
                                                trust_remote_code=True,
                                                cache_dir=cache_dir,
                                                )
        if getattr(tokenizer, "pad_token_id") is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
    else:
        assert args.model_type=='qwen'
        tokenizer = AutoTokenizer.from_pretrained(args.load_model_path,
                                            trust_remote_code=True,
                                            cache_dir=cache_dir,
                                            pad_token='<|endoftext|>',
                                            )
        tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids('<|endoftext|>')
    # load model
    predict_model = AutoModelForCausalLM.from_pretrained(args.load_model_path,
                                                     cache_dir="../../cache",
                                                     trust_remote_code=True)
    if args.load_ckpt:
        load_ckpt = args.load_ckpt
        predict_model = PeftModel.from_pretrained(predict_model, load_ckpt)
    predict_model.to(device)
    predict_model.eval()

    # generate
    logger.info("***** Generation *****")
    logger.info("  Num examples = %d", len(all_inputs))

    def decode(token_ids, model_type="baichuan"):
        return tokenizer.batch_decode(token_ids, skip_special_tokens=True)
            
    batch_size = args.eval_batch_size
    for i in tqdm(range(0, len(all_inputs), batch_size), desc="Testing"):
        e = min(len(all_inputs)-1, i+batch_size)
        inputs = tokenizer(all_inputs[i: e], return_tensors="pt",is_split_into_words=False, padding=True, max_length=args.max_seq_length)
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)
        with torch.no_grad():
            prd_ids = predict_model.generate(input_ids=input_ids,
                                            attention_mask=attention_mask,
                                            max_new_tokens=128,
                                            eos_token_id=tokenizer.eos_token_id)
        preds = decode(prd_ids)
        #print(preds)
        all_predictions += preds
    # compute metrics
    all_inputs_, all_labels_, all_predictions_ = all_inputs, all_labels, all_predictions
    all_inputs, all_labels, all_predictions = [], [], []
    for input, label, prediction in zip(all_inputs_, all_labels_, all_predictions_):
        all_inputs+=["".join(input).split('\n\n')[-2].split('\n')[-1]]
        #all_inputs+=["这句话不存在错别字\n"]
        all_labels+=["".join(label).split('\n')[-1]]
        #.split('\n')[-1].split("最终结果：")[-1]
        all_predictions+=["".join(prediction).split('\n')[-1]]
    all_inputs = [list(input) for input in all_inputs]
    all_labels = [list(label) for label in all_labels]
    all_predictions = [list(prediction) for prediction in all_predictions]
    for i in range(0,3):
        logger.info("input: %s " % " ".join([str(x) for x in all_inputs[i]]))
        logger.info("label: %s " % " ".join([str(x) for x in all_labels[i]]))
        logger.info("prediction: %s " % " ".join([str(x) for x in all_predictions[i]]))

    p, r, f1, fpr, wpr, tp_sents, fp_sents, fn_sents, wp_sents = Metrics.compute(all_inputs, all_labels, all_predictions,simple=True)

    if args.response_file:
        output_file = os.path.join(args.output_dir, args.response_file)
        with open(output_file, "w") as writer:
            for input, label, prediction in zip(all_inputs, all_labels, all_predictions):
                writer.write("input: " + " ".join(input) + "\t")
                writer.write("label: " + " ".join(label) + "\t")
                writer.write("prediction: " + " ".join(prediction) + "\t")
                if prediction==label:
                    writer.write("correct\n")
                else:
                    writer.write("wrong\n")

    result = {
        "eval_p": p*100,
        "eval_r": r*100,
        "eval_f1": f1*100,
        "eval_fpr": fpr*100,
    }
    output_eval_file = os.path.join(args.output_dir, args.result_file)

    def printf():
        with open(output_eval_file, "a") as writer:
            writer.write("-------\n")
            writer.write("test model: {} on dataset: {}.\n".format(args.load_model_path, args.test_on))
            writer.write(
                "Epoch %s: p = %.3f | r = %.3f | f1 = %.2f | fpr = %.2f\n"
                % (str(-1),
                result["eval_p"],
                result["eval_r"],
                result["eval_f1"],
                result["eval_fpr"]))

    printf()

    output_wp_file = os.path.join(args.output_dir,"wp.txt") 
    with open(output_wp_file, "w") as writer:
        for line in wp_sents:
            writer.write(line + "\n")
    output_fp_file = os.path.join(args.output_dir,"fp.txt") 
    with open(output_fp_file, "w") as writer:
        for line in fp_sents:
            writer.write(line + "\n")
    output_fn_file = os.path.join(args.output_dir,"fn.txt") 
    with open(output_fn_file, "w") as writer:
        for line in fn_sents:
            writer.write(line + "\n")
if __name__ == '__main__':
    main()


