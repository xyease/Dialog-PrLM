from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import BertModel, BertPreTrainedModel
import torch.nn.functional as F
from enum import Enum
from bert_insert_model import BertInsertion
from bert_delete_model_3 import BertDeletion
from bert_replace_model_2 import BertReplace
from torch.utils.data import Dataset


# class Flag(Enum):
#     insert = "insert"
#     delete = "delete"
#     relace = "replace"


class BertWIKIpretrainmodel(BertPreTrainedModel):
    def __init__(self, config):
        super(BertWIKIpretrainmodel, self).__init__(config)
        self.config = config
        # self.num_labels = config.num_labels
        self.bert = BertModel(config)


        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.bertinsert = BertInsertion(self.config)
        self.bertdelete = BertDeletion(self.config)
        self.bertreplace = BertReplace(self.config)

        self.init_weights()

    def forward(
        self,
        task,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sep_positions=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]  # batch x seglen x dim

        if task == "insert":
            logits, loss = self.bertinsert(sequence_output, sep_positions, labels)
        elif task == "delete_v3":
            logits, loss = self.bertdelete(sequence_output, sep_positions, labels)
        else:
            logits, loss = self.bertreplace(sequence_output, sep_positions, labels)

        return logits, loss