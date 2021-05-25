import torch.nn as nn
from torch.nn import CrossEntropyLoss

class BertInsertion(nn.Module):
    def __init__(self, config):
        super(BertInsertion, self).__init__()
        self.config = config
        self.cos = nn.CosineSimilarity(dim=2, eps=1e-6)

    def forward(self, sequence_output, sep_positions, labels):
        """

        Args:
            sequence_output: batch x seq_len x dim
            sep_positions: batch x choice_num
            labels: batch

        Returns:

        """
        b, s, d = sequence_output.size()
        _, n = sep_positions.size()

        ## select sep token
        expand_sep_pos = sep_positions.unsqueeze(2).expand(b, n, d) # batch x choice_num x dim
        utt_seps = sequence_output.gather(1, expand_sep_pos) #batch x choice_num x dim

        speaker1 = utt_seps[:, 0, :] #batch x dim
        remain_utt = utt_seps[:, 1:, :] #batch x (choice_num-1) x dim
        speaker1 = speaker1.unsqueeze(1).expand(b, remain_utt.size()[1], d).contiguous() #batch x (choice_num-1) x dim
        sim_scores = self.cos(remain_utt, speaker1) #batch x (choice_num-1)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(sim_scores.view(b, n-1), target=labels)
            return sim_scores, loss
        else:
            return sim_scores, None



