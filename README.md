# Rethinking Masked Language Modeling for Chinese Spelling Correction

This is the official repo for the ACL 2023 paper Rethinking Masked Language Modeling for Chinese Spelling Correction.



## LEMOM

LEMON (large-scale multi-domain dataset with natural spelling errors) is a novel benchmark released with our paper. All test sets are in `lemon`. (coming soon)

The other test sets we use in the paper are in `sighan_ecspell`.

The confusion sets are in `confus`.



## AutoCSC

We implement some architectures in recent CSC papers in `autocsc.py`.

For instance (SoftMasked BERT):

```python
from autocsc import AutoCSCSoftMasked

# To load the model, similar to huggingface transformers.
model = AutoCSCSoftMasked.from_pretrained("bert-base-chinese",
                                          cache_dir="cache")

# To go forward step.
outputs = model(src_ids=src_ids,
                attention_mask=attention_mask,
                trg_ids=trg_ids)
loss = outputs["loss"]
prd_ids = outputs["predict_ids"].tolist()
```

If you have new models or suggestions for promoting our implementations, feel free to email me.

