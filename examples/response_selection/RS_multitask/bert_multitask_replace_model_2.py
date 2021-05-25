import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch

class BertReplace(nn.Module):
    def __init__(self, config):
        super(BertReplace, self).__init__()
        # self.cos = nn.CosineSimilarity(dim=2, eps=1e-6)
        self.config = config
        self.classification = nn.Sequential(
            nn.Dropout(self.config.hidden_dropout_prob),
            nn.Linear(self.config.hidden_size, 1)
        )

        # self.mydevice = mydevice
        self._criterion = nn.CrossEntropyLoss()

    def forward(self, sequence_output, sot_positions, labels):
        """

        Args:
            sequence_output: batch x seq_len x dim
            sep_positions: batch x sent_num
            labels: batch

        Returns:

        """
        # b, s, d = sequence_output.size()
        # _, n = sep_positions.size()

        rep_losses = []
        pred_labels = []
        true_labels = []

        for batch_idx, sot_pos in enumerate(sot_positions):
            if labels[batch_idx] == -1:
                continue

            sot_pos_nonzero = sot_pos.nonzero().view(-1)
            batch_sot_output = sequence_output[batch_idx, sot_pos_nonzero, :]  # sot_num x dim
            logits = self.classification(batch_sot_output).view(-1)  # sot_num

            label = labels[batch_idx]

            if labels is not None:
                rep_loss = self._criterion(logits.unsqueeze(0), label.unsqueeze(0))
                rep_losses.append(rep_loss)
                true_labels.append(label)

            pred_label = torch.max(logits, dim=-1)[1]
            pred_labels.append(pred_label)

        if labels is not None:
            if len(rep_losses) == 0:
                replace_loss = torch.tensor(0).float().to(torch.cuda.current_device())
                pred_labels = torch.tensor([]).to(torch.cuda.current_device())
                true_labels = torch.tensor([]).to(torch.cuda.current_device())
                return replace_loss, pred_labels, true_labels
                # return None, None, None
            else:
                replace_loss = torch.mean(torch.stack(rep_losses, dim=0), dim=-1)
                pred_labels = torch.stack(pred_labels, dim=0)
                true_labels = torch.stack(true_labels, dim=0)
                return replace_loss, pred_labels, true_labels
        else:
            if len(pred_labels) == 0:
                pred_labels = torch.tensor([]).to(torch.cuda.current_device())
                return None, pred_labels, None
            else:
                return None, torch.stack(pred_labels, dim=0), None


        # ## select sep token
        # expand_sep_pos = sep_positions.unsqueeze(2).expand(b, n, d)  # batch x sent_num x dim
        # utt_seps = sequence_output.gather(1, expand_sep_pos)  # batch x sent_num x dim
        # # utt_seps = utt_seps[:, 1:-1, :]  # batch x choice_num x dim
        #
        # logits = self.classification(utt_seps).view(b, n)
        # # labels = labels - 1
        #
        # # speaker1 = utt_seps[:, 0, :] #batch x dim
        # # remain_utt = utt_seps[:, 1:, :] #batch x (choice_num-1) x dim
        # # speaker1 = speaker1.unsqueeze(1).expand(b, remain_utt.size()[1], d).contiguous() #batch x (choice_num-1) x dim
        # # sim_scores = self.cos(remain_utt, speaker1) #batch x (choice_num-1)
        #
        # if labels is not None:
        #     loss_fct = CrossEntropyLoss()
        #     loss = loss_fct(logits, target=labels)
        #     return loss, loss, logits
        # else:
        #     return None, None, logits



