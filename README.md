# Rethinking Masked Language Modeling for Chinese Spelling Correction

This is the official repo for the ACL 2023 paper Rethinking Masked Language Modeling for Chinese Spelling Correction.



## LEMOM

LEMON (large-scale multi-domain dataset with natural spelling errors) is a novel benchmark released with our paper. All test sets are in `lemon_v2`.

**Note: This dataset can only be used for academic research, it cannot be used for commercial purposes.**

The other test sets we use in the paper are in `sighan_ecspell`.

The confusion sets are in `confus`.



**Trained weights**

In our paper, we train BERT for 30,000 steps, with the learning rate in 5e- 5 and batch size 8192. We share our trained model weights to facilitate future research. We welcome researchers to develop better ones based on our models.

[BERT-MFT]()

[BERT-AT-MFT]()

[BERT-SoftMasked-MFT]()

(Coming soon)



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

