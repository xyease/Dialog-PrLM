from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import BertModel, BertPreTrainedModel
import torch.nn.functional as F
from enum import Enum
from bert_multitask_insert_model import BertInsertion
from bert_multitask_delete_model_3 import BertDeletion
from bert_multitask_replace_model_2 import BertReplace
from bert_multitask_RSmatching_model_2 import RSmatching_model
# from bert_postnspmlm_model import BertForPreTraining
from torch.utils.data import Dataset


# class Flag(Enum):
#     insert = "insert"
#     delete = "delete"
#     relace = "replace"


class Bertmultitaskmodel(BertPreTrainedModel):
    def __init__(self, config):
        super(Bertmultitaskmodel, self).__init__(config)
        self.config = config
        # self.num_labels = config.num_labels
        self.bert = BertModel(config)


        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        # self.bertnspmlm = BertForPreTraining(self.config)
        self.bertinsert = BertInsertion(self.config)
        self.bertdelete = BertDeletion(self.config)
        self.bertreplace = BertReplace(self.config)
        self.bertreselect = RSmatching_model(self.config)

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
        sot_positions=None,
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
            loss, pred_labels, true_labels = self.bertinsert(sequence_output, sot_positions, labels)
            return loss, pred_labels, true_labels
        elif task == "delete":
            loss, pred_labels, true_labels = self.bertdelete(sequence_output, sot_positions, labels)
            return loss, pred_labels, true_labels
        elif task == "replace":
            loss, pred_labels, true_labels = self.bertreplace(sequence_output, sot_positions, labels)
            return loss, pred_labels, true_labels
        elif task == "reselect":
            loss, logits = self.bertreselect(sequence_output, sot_positions, labels)
            return loss, logits, None
        else:
            raise AssertionError("undefined task")
