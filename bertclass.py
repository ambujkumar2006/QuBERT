# import torch
# import torch.nn as nn
# from transformers import BertModel


# class BERTLayer(nn.Module):

#     def __init__(self):
#         super().__init__()

#         self.bert = BertModel.from_pretrained(
#             "bert-base-uncased"
#         )

#     def forward(self, input_ids, attention_mask):

#         output = self.bert(
#             input_ids=input_ids,
#             attention_mask=attention_mask
#         )

#         return output.last_hidden_state

import torch
import torch.nn as nn
from transformers import BertModel


class BERTLayer(nn.Module):
    # 1. Accept a freeze argument (default to False)
    def __init__(self, freeze=False):
        super().__init__()

        self.bert = BertModel.from_pretrained(
            "bert-base-uncased"
        )
        
        # 2. Freeze the weights if requested
        if freeze:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):

        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        return output.last_hidden_state
