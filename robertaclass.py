import torch
import torch.nn as nn
from transformers import RobertaModel


class RoBERTaLayer(nn.Module):

    def __init__(
        self,
        model_name="roberta-base"
    ):
        super().__init__()

        self.roberta = RobertaModel.from_pretrained(
            model_name
        )

    def forward(
        self,
        input_ids,
        attention_mask
    ):

        output = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        return output.last_hidden_state