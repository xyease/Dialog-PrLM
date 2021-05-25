import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import BertPreTrainedModel, BertModel
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable


class RSmatching_model(nn.Module):
    def __init__(self, config):
        super(RSmatching_model, self).__init__()
        self.config=config
        # self.bert = BertModel(config)
        self.num_labels = config.num_labels
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # print(device)
        # self.mydevice = device
        # self.transformer_utt = TransformerBlock(device=device, input_size=self.config.hidden_size)
        # self.cos = nn.CosineSimilarity(dim=2, eps=1e-6)
        self.utt_gru_acc = nn.GRU(input_size=self.config.hidden_size, hidden_size=self.config.hidden_size, batch_first=True)
        self.classifier = nn.Linear(in_features=self.config.hidden_size, out_features=self.num_labels)
        # self.loss_func=BCELoss()
        # self.init_weights()

    def forward(self, sequence_output, sot_positions, labels=None):
        self.utt_gru_acc.flatten_parameters()
        # , _ = sequence_output.size()  # batch x seq_len
        # n = sot_pos.size()[1] # dialogue utterance + response num
        # sequence_output, pooled_output = self.bert(input_ids=input_ids,
        #                                            attention_mask=attention_mask,
        #                                            token_type_ids=token_type_ids)
        b, _, d = sequence_output.size()
        # b,dim
        match_vec = torch.zeros(b, d).to(torch.cuda.current_device())

        for batch_idx, sot_pos in enumerate(sot_positions):
            sot_pos_nonzero = sot_pos.nonzero().view(-1)
            b_utt_res = sequence_output[batch_idx, sot_pos_nonzero, :]  # sot_num x dim

            # b_len = true_len[bind]
            # b_sot_pos = sot_pos[bind][:b_len]
            #
            # b_utt_res = torch.index_select(seq, 0, b_sot_pos)  # b_len x dim
            b_utt_res = b_utt_res.unsqueeze(0)  # 1 x sot_num x dim
            b_hidden, _ = self.utt_gru_acc(b_utt_res)  # 1 x sot_num x dim
            match_vec[batch_idx] = b_hidden[-1, -1, :]

        match_vec = self.dropout(match_vec)  # batch x dim

        logits = self.classifier(match_vec)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            logits = F.softmax(logits, dim=1)
            return loss, logits
        else:
            logits = F.softmax(logits, dim=1)
            return _, logits
