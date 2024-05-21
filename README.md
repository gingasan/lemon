# Rethinking Masked Language Modeling for Chinese Spelling Correction

This is the official repo for the ACL 2023 paper [Rethinking Masked Language Modeling for Chinese Spelling Correction](https://aclanthology.org/2023.acl-long.600.pdf); AAAI 2024 paper [Chinese Spelling Correction as Rephraing Language Model](https://arxiv.org/pdf/2308.08796.pdf).



Fine-tuning results on some of benchmarks:

|                         | EC-LAW   | EC-MED   | EC-ODW   | MCSC     |
| ----------------------- | -------- | -------- | -------- | -------- |
| BERT                    | 39.8     | 22.3     | 25.0     | 70.7     |
| MDCSpell-**Masked-FT**  | 80.6     | 69.6     | 66.9     | 78.5     |
| Baichuan2-**Masked-FT** | 86.0     | 73.2     | 82.6     | 75.5     |
| **ReLM**                | **95.6** | **89.9** | **92.3** | **83.2** |



==New==

**ReLM**

*ReLM* pre-trained model is released. It is a rephrasing language model trained based on bert-base-chinese and 34 million monolingual data.

The main idea is illustrated in the figure below. We concatenate the input and a sequence of mask tokens of the same length as the input, and train the model to rephrase the entire sentence by infilling additional slots, instead of character-to-character tagging. We also apply the masked-fine-tuning technique during training, which masks a proportion of characters in the source sentence. We will not mask source sentence in evaluation stage.

![](figs/relm.png)

[relm-m0.3.bin](https://drive.google.com/file/d/10vvkG_jzNK-CjIwlSvizhE1IOpnn9OqN/view?usp=share_link)

Different from BERT-MFT, ReLM is a pure language model, which optimizes the rephrasing language modeling objective instead of sequence tagging. 

```python
from autocsc import AutoCSCReLM

model = AutoCSCReLM.from_pretrained("bert-base-chinese",
                                    state_dict=torch.load("relm-m0.3.bin"),
                                    cache_dir="cache")
```



**Monolingual data**

We share our used training data for LEMON. It contains 34 million monolingual sentences and we synthesize sentence pairs based on our confusion set in `confus`.

[monolingual-wiki-news-l64](https://drive.google.com/file/d/144ui9mkHEK1xLNZXB1WP-EjmydorwkYg/view?usp=share_link)

We split the data into 343 sub-files with 100,000 sentences for each. The total size of the .zip file is 1.5G.

Our code supports multiple GPUs now:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --multi_gpu run.py \
  --do_train \
  --do_eval \
  --fp16 \
  --mft
```



## LEMOM

*LEMON (large-scale multi-domain dataset with natural spelling errors)* is a novel benchmark released with our paper. All test sets are in `lemon_v2`.

**Note: This dataset can only be used for academic research, it cannot be used for commercial purposes.**

The other test sets we use in the paper are in `sighan_ecspell`.

The confusion sets are in `confus`.



**Trained weights**

In our paper, we train BERT for 30,000 steps, with the learning rate 5e-5 and batch size 8192. The backbone model is [bert-base-chinese](https://huggingface.co/bert-base-chinese). We share our trained model weights to facilitate future research. We welcome researchers to develop better ones based on our models.

[BERT-finetune-MFT](https://drive.google.com/file/d/1nKWX0G5e-xzx7D66MzcAFOK-5CSr0_yH/view?usp=share_link)

[BERT-finetune-MFT-CreAT-maskany](https://drive.google.com/file/d/1g7mxIQMLloxpPSJcW65KU4uZmbVN985c/view?usp=share_link)

[BERT-SoftMasked-MFT](https://drive.google.com/file/d/1HBLw4IM4JCz3g7P6YedTsPU_1DBQhv8m/view?usp=share_link)



## AutoCSC

We implement some architectures in recent CSC papers in `autocsc.py`.

* [Spelling Error Correction with Soft-Masked BERT](https://aclanthology.org/2020.acl-main.82.pdf)

* [MDCSpell: A Multi-task Detector-Corrector Framework for Chinese Spelling Correction](https://aclanthology.org/2022.findings-acl.98.pdf)

For instance (Soft-Masked BERT):

```python
from autocsc import AutoCSCSoftMasked

# Load the model, similar to huggingface transformers.
model = AutoCSCSoftMasked.from_pretrained("bert-base-chinese",
                                          cache_dir="cache")

# Go forward step.
outputs = model(src_ids=src_ids,
                attention_mask=attention_mask,
                trg_ids=trg_ids)
loss = outputs["loss"]
prd_ids = outputs["predict_ids"].tolist()
```

If you have new models or suggestions for promoting our implementations, feel free to email me.



Running (set `--mft` for **Masked-FT**):

```bash
CUDA_VISIBLE_DEVICES=0 python run.py \
  --do_train \
  --do_eval \
  --train_on xxx.txt \
  --eval_on xx.txt \
  --output_dir mft \
  --max_train_steps 10000 \
  --fp16 \
  --model_type mdcspell \
  --mft
```



Directly testing on LEMON (including SIGHAN):

```bash
CUDA_VISIBLE_DEVICES=0 python run.py \
  --test_on_lemon ../data/lemon \
  --output_dir relm \
  --model_type relm \
  --load_state_dict relm-m0.3.bin
```