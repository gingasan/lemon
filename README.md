# Rethinking Masked Language Modeling for Chinese Spelling Correction

This is the official repo for the ACL 2023 paper [Rethinking Masked Language Modeling for Chinese Spelling Correction](https://arxiv.org/abs/2305.17721).



==New==

**New work**

*ReLM* is now released. It is a rephrasing language model trained based on bert-base-chinese and 34 million monolingual data.

[relm-m0.2.bin](https://drive.google.com/file/d/1vjPhPO0fTYZS80dTE5ba4vhwDlSnSfA4/view?usp=share_link)

Different from BERT-MFT, ReLM is a pure language model, which optimizes the rephrasing training objective instead of sequence tagging. The details are in our paper [Chinese Spelling Correction as Rephraing Language Model]().



**Monolingual data**

We share our used training data for LEMON. It contains 34 million monolingual sentences and we synthesize sentence pairs based on our confusion set in `confus`.

[monolingual-wiki-news-l64](https://drive.google.com/file/d/144ui9mkHEK1xLNZXB1WP-EjmydorwkYg/view?usp=share_link)

We split the data into 343 sub-files with 100,000 sentences for each. The total size of the .zip file is 1.5G.



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

For instance (SoftMasked BERT):

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

