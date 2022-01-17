import string
import torch
import torch.nn as nn

from transformers import BertPreTrainedModel, BertModel, BertTokenizerFast, AutoModel, AutoTokenizer, PreTrainedModel
from colbert.parameters import DEVICE


class ColBERT(PreTrainedModel):
    # from BertPretrainedModel, might be useful
    base_model_prefix = "colbert"
    _keys_to_ignore_on_load_missing = [r"position_ids", r"encoder_colbert"]

    def __init__(self, config, query_maxlen, doc_maxlen, mask_punctuation, dim=128, similarity_metric='cosine'):

        super(ColBERT, self).__init__(config)

        self.query_maxlen = query_maxlen
        self.doc_maxlen = doc_maxlen
        self.similarity_metric = similarity_metric
        self.dim = dim

        self.mask_punctuation = mask_punctuation
        self.skiplist = {}

        if self.mask_punctuation:
            self.tokenizer = AutoTokenizer.from_pretrained(config.name_or_path)
            self.skiplist = {w: True
                             for symbol in string.punctuation
                             for w in [symbol, self.tokenizer.encode(symbol, add_special_tokens=False)[0]]}

        self.bert = None
        self.roberta = None
        self.electra = None
        self.distilbert = None
        self.encoder_colbert = None
        if config.model_type == 'bert':
            self.bert = AutoModel.from_config(config)
            self.encoder_colbert = self.bert
        elif config.model_type in ['roberta', 'xlm-roberta']:
            self.roberta = AutoModel.from_config(config)
            self.encoder_colbert = self.roberta
        elif config.model_type == 'electra':
            self.electra = AutoModel.from_config(config)
            self.encoder_colbert = self.electra
        elif config.model_type == 'distilbert':
            self.distilbert = AutoModel.from_config(config)
            self.encoder_colbert = self.distilbert
        else:
            print("Please add lines to support this model type.")
            raise NotImplementedError
        self.linear = nn.Linear(config.hidden_size, dim, bias=False)

        # self._init_weights()

    # from BertPretrainedModel
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, Q, D):
        return self.score(self.query(*Q), self.doc(*D))

    def query(self, input_ids, attention_mask):
        input_ids, attention_mask = input_ids.to(DEVICE), attention_mask.to(DEVICE)
        Q = self.encoder_colbert(input_ids, attention_mask=attention_mask)[0]
        Q = self.linear(Q)

        return torch.nn.functional.normalize(Q, p=2, dim=2)

    # def query_list(self, input_ids, attention_mask):
    #     input_ids, attention_mask = input_ids.to(DEVICE), attention_mask.to(DEVICE)
    #     Q = self.encoder_colbert(input_ids, attention_mask=attention_mask)[0]
    #     Q = self.linear(Q)
    #     Q = torch.nn.functional.normalize(Q, p=2, dim=2)
    #     Q = Q.cpu().to(dtype=torch.float16)
    #     # could consider trimming down later
    #     return [q for idx, q in enumerate(Q)]
    #
    def doc(self, input_ids, attention_mask, keep_dims=True):
        input_ids, attention_mask = input_ids.to(DEVICE), attention_mask.to(DEVICE)
        D = self.encoder_colbert(input_ids, attention_mask=attention_mask)[0]
        D = self.linear(D)

        mask = torch.tensor(self.mask(input_ids), device=DEVICE).unsqueeze(2).float()
        D = D * mask

        D = torch.nn.functional.normalize(D, p=2, dim=2)

        if not keep_dims:
            D, mask = D.cpu().to(dtype=torch.float16), mask.cpu().bool().squeeze(-1)
            D = [d[mask[idx]] for idx, d in enumerate(D)]

        return D

    def score(self, Q, D):
        if self.similarity_metric == 'cosine':
            return (Q @ D.permute(0, 2, 1)).max(2).values.sum(1)

        assert self.similarity_metric == 'l2'
        return (-1.0 * ((Q.unsqueeze(2) - D.unsqueeze(1))**2).sum(-1)).max(-1).values.sum(-1)

    def mask(self, input_ids):
        mask = [[(x not in self.skiplist) and (x != 0) for x in d] for d in input_ids.cpu().tolist()]
        return mask
