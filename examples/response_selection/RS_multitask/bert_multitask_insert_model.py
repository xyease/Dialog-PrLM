import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch
import logging
# testlogger = logging.getLogger(__name__)
# testlogger.setLevel(level=logging.INFO)
# handler = logging.FileHandler("../../result/ubuntu/testlog.txt")
# handler.setLevel(logging.INFO)
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
# handler.setFormatter(formatter)
# testlogger.addHandler(handler)


class BertInsertion(nn.Module):
    def __init__(self, config):
        super(BertInsertion, self).__init__()
        self.config = config
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        # self.mydevice = mydevice
        self._criterion = nn.CrossEntropyLoss()

    def forward(self, sequence_output, sot_positions, labels):
        """

        Args:
            sequence_output: batch x seq_len x dim
            sep_positions: batch x choice_num
            labels: batch

        Returns:

        """
        # testlogger.info("start insert forward" + str(torch.cuda.current_device()))
        b, s, d = sequence_output.size()
        # _, n = sep_positions.size()

        ins_losses = []
        pred_labels = []
        true_labels = []
        for batch_idx, sot_pos in enumerate(sot_positions):
            if labels[batch_idx] == -1:
                # testlogger.info("omit batch" + str(batch_idx))
                continue

            sot_pos_nonzero = sot_pos.nonzero().view(-1)
            batch_sot_output = sequence_output[batch_idx, sot_pos_nonzero, :]  #sot_num x dim
            speaker1 = batch_sot_output[0]  # dim
            remain_utt = batch_sot_output[1:, :]  # (sot_num-1) x dim
            speaker1 = speaker1.unsqueeze(0).expand(remain_utt.size()[0], d).contiguous()  # (sot_num-1) x dim
            sim_score = self.cos(remain_utt, speaker1)  # (sot_num-1)
            label = labels[batch_idx]

            if labels is not None:
                ins_loss = self._criterion(sim_score.unsqueeze(0), label.unsqueeze(0))
                ins_losses.append(ins_loss)
                true_labels.append(label)

            pred_label = torch.max(sim_score, dim=-1)[1]
            pred_labels.append(pred_label)
            # testlogger.info("success process batch" + str(batch_idx)+ " " + str(len(ins_losses)))
        # test = torch.tensor(0.0).to(self.mydevice)

        if labels is not None:
            if len(ins_losses) == 0:
                insertion_loss = torch.tensor(0).float().to(torch.cuda.current_device())
                pred_labels = torch.tensor([]).to(torch.cuda.current_device())
                true_labels = torch.tensor([]).to(torch.cuda.current_device())
                return insertion_loss, pred_labels, true_labels
            else:
                insertion_loss = torch.mean(torch.stack(ins_losses, dim=0), dim=-1)
                pred_labels = torch.stack(pred_labels, dim=0)
                true_labels = torch.stack(true_labels, dim=0)
                # return test, None, None
                return insertion_loss, pred_labels, true_labels
        else:
            if len(pred_labels) == 0:
                pred_labels = torch.tensor([]).to(torch.cuda.current_device())
                return None, pred_labels, None
            else:
                return None, torch.stack(pred_labels, dim=0), None

        # ## select sep token
        # expand_sep_pos = sep_positions.unsqueeze(2).expand(b, n, d) # batch x choice_num x dim
        # utt_seps = sequence_output.gather(1, expand_sep_pos) #batch x choice_num x dim
        #
        # speaker1 = utt_seps[:, 0, :] #batch x dim
        # remain_utt = utt_seps[:, 1:, :] #batch x (choice_num-1) x dim
        # speaker1 = speaker1.unsqueeze(1).expand(b, remain_utt.size()[1], d).contiguous() #batch x (choice_num-1) x dim
        # sim_scores = self.cos(remain_utt, speaker1) #batch x (choice_num-1)
        # if labels is not None:
        #     loss_fct = CrossEntropyLoss()
        #     loss = loss_fct(sim_scores.view(b, n-1), target=labels)
        #     return loss, loss, sim_scores
        # else:
        #     return None, None, sim_scores



