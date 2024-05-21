import torch
import torch.nn as nn
from copy import deepcopy
from transformers import BertPreTrainedModel, BertModel, BertForMaskedLM


class AutoCSCReLM(BertForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.vocab_size
        self.softmax = nn.Softmax(-1)

    def forward(self, src_ids, trg_ids, attention_mask, output_attentions=False):
        labels = trg_ids.clone()
        labels[(src_ids == trg_ids)] = -100

        outputs = self.bert(
            input_ids=src_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        if output_attentions:
            attentions = outputs[-1]
        else:
            attentions = None

        sequence_output = outputs[0]
        logits = self.cls(sequence_output)

        loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        probs = self.softmax(logits)
        _, predict_ids = torch.max(probs, -1)

        return {
            "loss": loss,
            "predict_ids": predict_ids,
            "attentions": attentions,
        }


class PromptEmbeddings(nn.Module):
    def __init__(self, hidden_size, num_virtual_tokens=10):
        super().__init__()
        self.num_virtual_tokens = num_virtual_tokens
        self.hidden_size = hidden_size

        self.embedding = torch.nn.Embedding(num_virtual_tokens, hidden_size)
        layers = [
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_size, self.hidden_size)
        ]
        self.mlp_head = torch.nn.Sequential(*layers)

    def forward(self, input_ids):
        input_embeds = self.embedding(input_ids)
        output_embeds = self.mlp_head(input_embeds)

        return output_embeds


class AutoCSCReLMPrompt(BertForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.vocab_size

        self.pe = PromptEmbeddings(config.hidden_size)
        self.softmax = nn.Softmax(-1)

        self.post_init()

    def forward(self, src_ids, trg_ids, attention_mask):
        batch_size = src_ids.size(0)
        prefix_attention_mask = torch.ones(batch_size, self.pe.num_virtual_tokens).to(attention_mask.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        inputs_embeds = self.bert.embeddings(src_ids)
        indices = torch.arange(self.pe.num_virtual_tokens).unsqueeze(0).expand(batch_size, -1).to(src_ids.device)
        prompts = self.pe(indices)
        prompts = prompts.to(inputs_embeds.dtype)
        inputs_embeds = torch.cat((prompts, inputs_embeds), dim=1)

        labels = trg_ids.clone()
        labels[(src_ids == trg_ids)] = -100
        labels = torch.cat((torch.full_like(indices, -100).to(indices.dtype), labels), dim=1)

        outputs = self.bert(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )

        sequence_output = outputs[0]
        logits = self.cls(sequence_output)

        loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        probs = self.softmax(logits)
        _, predict_ids = torch.max(probs, -1)

        return {
            "loss": loss,
            "predict_ids": predict_ids,
        }


class AutoCSCfinetune(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.vocab_size

        self.bert = BertModel(config, add_pooling_layer=False)
        self.generate_linear = nn.Linear(config.hidden_size, self.num_labels)
        self.softmax = nn.Softmax(-1)

        self.post_init()

    def forward(self, src_ids, trg_ids, attention_mask, output_attentions=False):
        outputs = self.bert(
            src_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )

        sequence_output = outputs[0]

        if output_attentions:
            attentions = outputs[-1]
        else:
            attentions = None

        logits = self.generate_linear(sequence_output)
        probs = self.softmax(logits)
        _, predict_ids = torch.max(probs, -1)
        predict_ids = predict_ids.masked_fill(attention_mask == 0, 0)

        loss_fct = nn.CrossEntropyLoss(ignore_index=0) # ignore padding
        loss = loss_fct(logits.view(-1, self.num_labels), trg_ids.view(-1))

        return {
            "loss": loss,
            "predict_ids": predict_ids,
            "attentions": attentions,
        }


class AutoCSCSoftMasked(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.vocab_size

        self.bert = BertModel(config, add_pooling_layer=False)
        self.generate_linear = nn.Linear(config.hidden_size, self.num_labels)

        self.rnn = nn.GRU(config.hidden_size, 256, batch_first=True,
                          bidirectional=True, num_layers=2, dropout=0.2)

        self.mask_embed = nn.Linear(1, config.hidden_size, bias=False)
        self.copy_linear = nn.Linear(256 * 2, 1)

        self.softmax = nn.Softmax(-1)
        self.sigmoid = nn.Sigmoid()
        
        self.post_init()

    def forward(self, src_ids, trg_ids, attention_mask):
        embedding_output = self.bert.embeddings(src_ids)
        detect_embed = self.bert.embeddings.word_embeddings(src_ids)

        rnn_hidden_states, _ = self.rnn(detect_embed)
        copy_logits = self.copy_linear(rnn_hidden_states)
        copy_probs = self.sigmoid(copy_logits)
        embedding_output = copy_probs * embedding_output + self.mask_embed(1 - copy_probs)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        sequence_output = self.bert.encoder(embedding_output, extended_attention_mask)[0]
        sequence_output = sequence_output + embedding_output

        logits = self.generate_linear(sequence_output)
        probs = self.softmax(logits)
        _, predict_ids = torch.max(probs, -1)
        predict_ids = predict_ids.masked_fill(attention_mask == 0, 0)

        loss_fct = nn.CrossEntropyLoss(ignore_index=0) # ignore padding
        loss = loss_fct(logits.view(-1, self.num_labels), trg_ids.view(-1))

        b_loss_fct = nn.BCEWithLogitsLoss(reduction="none")
        b_logits_loss = b_loss_fct(copy_logits.view(-1), (src_ids == trg_ids).float().view(-1))
        b_loss = (b_logits_loss * attention_mask.view(-1)).mean()
        loss = 0.8 * loss + (1 - 0.8) * b_loss

        return {
            "loss": loss,
            "predict_ids": predict_ids,
        }


class AutoCSCMDCSpell(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.vocab_size

        self.bert = BertModel(config, add_pooling_layer=False)
        self.generate_linear = nn.Linear(config.hidden_size, self.num_labels)
        self.softmax = nn.Softmax(-1)

        self.detect_layers = deepcopy(self.bert.encoder.layer[:2])
        self.detect_sigmoid = nn.Sequential(nn.Linear(config.hidden_size, 1), nn.Sigmoid())
        self.emb = self.bert.embeddings
        
        self.post_init()

    def forward(self, src_ids, trg_ids, attention_mask):
        outputs = self.bert(
            src_ids,
            attention_mask=attention_mask,
        )

        sequence_output = outputs[0]

        position_ids = self.emb.position_ids[:, : src_ids.shape[1]]
        token_type_ids = torch.zeros(*src_ids.shape, dtype=torch.long, device=position_ids.device)
        detect_output = self.emb.word_embeddings(src_ids) + self.emb.position_embeddings(position_ids) + self.emb.token_type_embeddings(token_type_ids)
        for layer in self.detect_layers:
            detect_output = layer(detect_output)[0]
        detect_probs = self.detect_sigmoid(detect_output)

        logits = self.generate_linear(sequence_output + detect_output)
        probs = self.softmax(logits)
        _, predict_ids = torch.max(probs, -1)
        predict_ids = predict_ids.masked_fill(attention_mask == 0, 0)

        loss_fct = nn.CrossEntropyLoss(ignore_index=0) # ignore padding
        loss = loss_fct(logits.view(-1, self.num_labels), trg_ids.view(-1))

        detect_labels = (src_ids != trg_ids).float()
        detect_loss_fct = nn.BCEWithLogitsLoss(size_average=True)
        detect_loss = detect_loss_fct(detect_probs.squeeze(-1) * attention_mask, detect_labels)
        loss = 0.85 * loss + 0.15 * detect_loss

        return {
            "loss": loss,
            "predict_ids": predict_ids,
        }


class AutoCSCCRASpell(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.vocab_size

        self.bert = BertModel(config, add_pooling_layer=False)
        self.generate_linear = nn.Linear(config.hidden_size, self.num_labels)
        self.softmax = nn.Softmax(-1)

        self.copy = nn.Sequential(nn.Linear(config.hidden_size, 384), nn.GELU(), nn.Dropout(0.1), nn.LayerNorm(384), nn.Linear(384, 1), nn.Sigmoid())
        
        self.post_init()

    def forward(self, src_ids, trg_ids, attention_mask, noisy_src_ids=None):
        outputs = self.bert(
            src_ids,
            attention_mask=attention_mask,
        )

        sequence_output = outputs[0]

        logits = self.generate_linear(sequence_output)
        probs = self.softmax(logits)

        copy_probs = self.copy(sequence_output)
        helper_tensor = torch.ones([1, self.num_labels], dtype=torch.float32, device=copy_probs.device)
        copy_probs = torch.matmul(copy_probs, helper_tensor)
        src_one_hot_labels = torch.nn.functional.one_hot(src_ids, num_classes=self.num_labels).float()
        probs = copy_probs * src_one_hot_labels + (1.0 - copy_probs) * probs
        probs = torch.clip(probs, 1e-10, 1.0 - 1e-7)

        _, predict_ids = torch.max(probs, -1)
        predict_ids = predict_ids.masked_fill(attention_mask == 0, 0)

        log_probs = torch.log(probs)
        loss = nn.NLLLoss(ignore_index=0)(log_probs.view(-1, self.num_labels), trg_ids.view(-1))

        probs_n = None
        if noisy_src_ids is not None:
            bert_output_n = self.bert(noisy_src_ids, attention_mask)[0]
            logits_n = self.generate_linear(bert_output_n)
            probs_n = self.softmax(logits_n)
            log_probs_n = torch.log(probs_n)

            kl_fct = nn.KLDivLoss(reduction="none", log_target=True)
            kl_loss = 0.5 * (kl_fct(log_probs, log_probs_n) + kl_fct(log_probs_n, log_probs))
            kl_loss = torch.sum(kl_loss * attention_mask.unsqueeze(-1)) / torch.sum(attention_mask)
            loss = 0.95 * loss + 0.05 * kl_loss

        return {
            "loss": loss,
            "predict_ids": predict_ids
        }
