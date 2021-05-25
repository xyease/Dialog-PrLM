import logging
import numpy as np
from tqdm import tqdm
logger = logging.getLogger(__name__)
from tqdm import tqdm
import random
import torch
from torch.utils.data import Dataset


class MultiTaskDataset(Dataset):

  def __init__(
      self,
      examples,label_list, args, tokenizer
  ):
    super().__init__()

    self.examples = examples
    self.label_list = label_list
    self.args = args
    self.tokenizer = tokenizer


  def __len__(self):
    return len(self.examples)


  def __getitem__(self, index):
    curr_feature = convert_examples_to_features(index, self.examples, self.label_list,
                                                self.args, self.tokenizer)
    for task, task_feature in curr_feature.items():
        for datakey in task_feature.keys():
            curr_feature[task][datakey] = torch.tensor(curr_feature[task][datakey]).long()
    return curr_feature




class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()


class MyDataProcessorUtt(DataProcessor):
    def __init__(self):
        super(DataProcessor, self).__init__()

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(data_dir,"train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(data_dir, "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(data_dir, "test")

    def _create_examples(self, data_dir, set_type):
        """Creates examples for the texamples_res_lab = []
        # examples_utt = []raining and dev sets."""
        examples=[]
        #
        i = 0
        with open(data_dir, 'r', encoding='utf-8') as rf:
            for line in rf:
                line = line.strip()
                if line:
                    guid = "%s-%s" % (set_type, i)
                    label = line.split("\t")[0]
                    res = line.split("\t")[-1]
                    context = line.split("\t")[1:-1]
                    context = [x for x in context if x]
                    examples.append(
                        InputExample(guid=guid, text_a=context, text_b=res, label=label))
                    i += 1
        return examples

    def get_labels(self):
        """See base class."""
        return ["0", "1"]


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a=None, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


# class InputFeatures(object):
#     """A single set of features of data."""
#
#     def __init__(self, input_ids, input_mask, segment_ids, label_id, sot_pos,true_len):
#         self.input_ids = input_ids
#         self.input_mask = input_mask
#         self.segment_ids = segment_ids
#         self.label_id = label_id
#         self.sot_pos = sot_pos
#         self.true_len = true_len


def convert_examples_to_features(ex_index, examples, label_list, args, tokenizer):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label: i for i, label in enumerate(label_list)}
    # pbar = tqdm(total=len(examples), desc="converting examples to features")
    # features = []
    # for (ex_index, example) in enumerate(examples):
    #     if ex_index % 10000 == 0:
    #         logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        ### process response selection task
    example = examples[ex_index]
    reselect_feature = convert_reselect_features(ex_index, example, label_map, args, tokenizer)
    insert_feature = convert_insert_features(example, args, tokenizer)
    delete_feature = convert_delete_features(example, args, tokenizer)

    while True:
        rep_index = random.randint(0, len(examples) - 1)
        example_rep = examples[rep_index]
        if int(example_rep.label):
            rep_utts = example_rep.text_a + [example_rep.text_b]
        else:
            rep_utts = example_rep.text_a
        if len(rep_utts) < 2 or rep_utts[0] == example.text_a[0]:
            continue
        else:
            break

    rep_utt = rep_utts[random.randint(0, len(rep_utts) - 1)]
    replace_feature = convert_replace_features(example, rep_utt, args, tokenizer)

    feature = {"reselect": reselect_feature,
               "insert": insert_feature,
               "delete": delete_feature,
               "replace": replace_feature
               }
        # features.append(feature)
        # pbar.update(1)
    # pbar.close()
    return feature


def convert_reselect_features(ex_index, example, label_map, args, tokenizer,
                                 cls_token='[CLS]',
                                 cls_token_segment_id=1,
                                 sep_token='[SEP]',
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 sequence_b_segment_id=1,
                                 mask_padding_with_zero=True,
                                 sot_token="[SOT]"):
    tokens_a = []
    for text in example.text_a:
        tokens_a.extend([sot_token] + tokenizer.tokenize(text))

    tokens_b = tokenizer.tokenize(example.text_b)
    tokens_b = [sot_token] + tokens_b
    special_tokens_count = 3
    tokens_a, tokens_b = reselect_truncate_seq_pair(tokens_a, tokens_b, args.max_seq_length - special_tokens_count)
    tokens = tokens_a + [sep_token]
    segment_ids = [sequence_a_segment_id] * len(tokens)

    if tokens_b:
        tokens += tokens_b + [sep_token]
        segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

    tokens = [cls_token] + tokens
    segment_ids = [cls_token_segment_id] + segment_ids

    sot_positions = []
    for index, tk in enumerate(tokens):
        if tk == sot_token:
            sot_positions.append(1)
        else:
            sot_positions.append(0)
    true_len = len(sot_positions)
    sot_positions = sot_positions + ([0] * (args.max_seq_length - true_len))

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

    padding_length = args.max_seq_length - len(input_ids)

    input_ids = input_ids + ([pad_token] * padding_length)
    input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
    segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

    assert len(input_ids) == args.max_seq_length
    assert len(input_mask) == args.max_seq_length
    assert len(segment_ids) == args.max_seq_length
    assert len(sot_positions) == args.max_seq_length

    if (example.label):
        label_id = label_map[example.label]
    else:
        label_id = None

    if ex_index < 5:
        print("*** Example ***")
        print("guid: %s" % (example.guid))
        print("tokens: %s" % " ".join(
            [str(x) for x in tokens]))
        print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        print("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        print("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))

    reselect_feature = {
        "input_ids": input_ids,
        "attention_mask": input_mask,
        "token_type_ids": segment_ids,
        "labels": label_id,
        "sot_positions": sot_positions
    }

    return reselect_feature


def reselect_truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop(0)
        else:
            tokens_b.pop()
    if tokens_a[0] != "[SOT]":
        tokens_a.pop(0)
        tokens_a = ["[SOT]"] + tokens_a
    return tokens_a, tokens_b


def convert_insert_features(example, args, tokenizer, cls_token='[CLS]',
                            cls_token_segment_id=1, sep_token='[SEP]', pad_token=0,
                            pad_token_segment_id=0, sequence_a_segment_id=0,
                            mask_padding_with_zero=True, sot_token="[SOT]"):
    utts = example.text_a.copy()
    if int(example.label):
        utts += [example.text_b]

    if len(utts) > args.insert_max_num:
        randomseed = random.randint(0, len(utts)-args.insert_max_num)
        utts = utts[randomseed: randomseed + args.insert_max_num]
        assert len(utts) == args.insert_max_num

    if len(utts) <= 2:
        insert_feature = {
            "input_ids": [0] * args.max_seq_length,
            "attention_mask": [0] * args.max_seq_length,
            "token_type_ids": [0] * args.max_seq_length,
            "labels": -1,
            "sot_positions": [0] * args.max_seq_length
        }
        return insert_feature

    randomseed = random.randint(1, len(utts) - 1)
    temp = utts.pop(1)
    utts.insert(randomseed, temp)

    label_id = randomseed - 1
    max_len = args.max_seq_length - len(utts) - 1 - 1  # remove SOTS sep and cls

    tokens_list = []
    for utt in utts:
        tokens_list.append(tokenizer.tokenize(utt))
    task_truncate_seq_pair(tokens_list, max_len)

    tokens_a = []
    for tokens in tokens_list:
        tokens_a += ([sot_token] + tokens)

    tokens = tokens_a + [sep_token]
    segment_ids = [sequence_a_segment_id] * len(tokens)
    tokens = [cls_token] + tokens
    segment_ids = [cls_token_segment_id] + segment_ids

    sot_positions = []
    for index, tk in enumerate(tokens):
        if tk == sot_token:
            sot_positions.append(1)
        else:
            sot_positions.append(0)
    true_len = len(sot_positions)
    sot_positions = sot_positions + ([0] * (args.max_seq_length - true_len))
    assert sum(sot_positions) == len(utts)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

    padding_length = args.max_seq_length - len(input_ids)

    input_ids = input_ids + ([pad_token] * padding_length)
    input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
    segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

    assert len(input_ids) == args.max_seq_length
    assert len(input_mask) == args.max_seq_length
    assert len(segment_ids) == args.max_seq_length
    assert len(sot_positions) == args.max_seq_length


    insert_feature = {
        "input_ids": input_ids,
        "attention_mask": input_mask,
        "token_type_ids": segment_ids,
        "labels": label_id,
        "sot_positions": sot_positions
    }

    return insert_feature


def convert_delete_features(example, args, tokenizer, cls_token='[CLS]',
                            cls_token_segment_id=1, sep_token='[SEP]', pad_token=0,
                            pad_token_segment_id=0, sequence_a_segment_id=0,
                            sequence_b_segment_id=1, mask_padding_with_zero=True, sot_token="[SOT]"):
    utts = example.text_a.copy()
    if int(example.label):
        utts += [example.text_b]

    if len(utts) > args.delete_max_num:
        randomseed = random.randint(0, len(utts)-args.delete_max_num)
        utts = utts[randomseed: randomseed + args.delete_max_num]
        assert len(utts) == args.delete_max_num

    if len(utts) <= 2:
        delete_feature = {
            "input_ids": [0] * args.max_seq_length,
            "attention_mask": [0] * args.max_seq_length,
            "token_type_ids": [0] * args.max_seq_length,
            "labels": -1,
            "sot_positions": [0] * args.max_seq_length
        }
        return delete_feature

    randomseed = random.randint(0, len(utts) - 2)
    temp = utts.pop(randomseed)
    label_id = randomseed
    utts = utts + [temp]
    max_len = args.max_seq_length - len(utts) - 3  # remove SOTS sep and cls

    tokens_list = []
    for utt in utts:
        tokens_list.append(tokenizer.tokenize(utt))
    task_truncate_seq_pair(tokens_list, max_len)

    tokens_b = [sot_token] + tokens_list[-1] + [sep_token]
    tokens_list = tokens_list[:-1]

    tokens_a = []
    for tokens in tokens_list:
        tokens_a += ([sot_token] + tokens)

    tokens = tokens_a + [sep_token]
    segment_ids = [sequence_a_segment_id] * len(tokens)
    tokens = [cls_token] + tokens
    segment_ids = [cls_token_segment_id] + segment_ids
    tokens += tokens_b
    segment_ids += [sequence_b_segment_id] * len(tokens_b)

    sot_positions = []
    for index, tk in enumerate(tokens):
        if tk == sot_token:
            sot_positions.append(1)
        else:
            sot_positions.append(0)
    true_len = len(sot_positions)
    sot_positions = sot_positions + ([0] * (args.max_seq_length - true_len))
    assert sum(sot_positions) == len(utts)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
    padding_length = args.max_seq_length - len(input_ids)
    input_ids = input_ids + ([pad_token] * padding_length)
    input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
    segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

    assert len(input_ids) == args.max_seq_length
    assert len(input_mask) == args.max_seq_length
    assert len(segment_ids) == args.max_seq_length
    assert len(sot_positions) == args.max_seq_length


    delete_feature = {
        "input_ids": input_ids,
        "attention_mask": input_mask,
        "token_type_ids": segment_ids,
        "labels": label_id,
        "sot_positions": sot_positions
    }

    return delete_feature


def convert_replace_features(example, rep_utt, args, tokenizer, cls_token='[CLS]',
                            cls_token_segment_id=1, sep_token='[SEP]', pad_token=0,
                            pad_token_segment_id=0, sequence_a_segment_id=0,
                            mask_padding_with_zero=True, sot_token="[SOT]"):
    utts = example.text_a.copy()
    if int(example.label):
        utts += [example.text_b]

    if len(utts) > args.replace_max_num:
        randomseed = random.randint(0, len(utts)-args.replace_max_num)
        utts = utts[randomseed: randomseed + args.replace_max_num]
        assert len(utts) == args.replace_max_num

    if len(utts) <= 2:
        replace_feature = {
            "input_ids": [0] * args.max_seq_length,
            "attention_mask": [0] * args.max_seq_length,
            "token_type_ids": [0] * args.max_seq_length,
            "labels": -1,
            "sot_positions": [0] * args.max_seq_length
        }
        return replace_feature

    randomseed = random.randint(0, len(utts) - 1)
    utts[randomseed] = rep_utt

    label_id = randomseed
    max_len = args.max_seq_length - len(utts) - 1 - 1  # remove SOTS sep and cls

    tokens_list = []
    for utt in utts:
        tokens_list.append(tokenizer.tokenize(utt))
    task_truncate_seq_pair(tokens_list, max_len)

    tokens_a = []
    for tokens in tokens_list:
        tokens_a += ([sot_token] + tokens)

    tokens = tokens_a + [sep_token]
    segment_ids = [sequence_a_segment_id] * len(tokens)
    tokens = [cls_token] + tokens
    segment_ids = [cls_token_segment_id] + segment_ids

    sot_positions = []
    for index, tk in enumerate(tokens):
        if tk == sot_token:
            sot_positions.append(1)
        else:
            sot_positions.append(0)
    true_len = len(sot_positions)
    sot_positions = sot_positions + ([0] * (args.max_seq_length - true_len))
    assert sum(sot_positions) == len(utts)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

    padding_length = args.max_seq_length - len(input_ids)

    input_ids = input_ids + ([pad_token] * padding_length)
    input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
    segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

    assert len(input_ids) == args.max_seq_length
    assert len(input_mask) == args.max_seq_length
    assert len(segment_ids) == args.max_seq_length
    assert len(sot_positions) == args.max_seq_length


    replace_feature = {
        "input_ids": input_ids,
        "attention_mask": input_mask,
        "token_type_ids": segment_ids,
        "labels": label_id,
        "sot_positions": sot_positions
    }

    return replace_feature

def task_truncate_seq_pair(tokens_list, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    def get_total_len():
        total_len = 0
        utt_len_list = []
        for tokens in tokens_list:
            total_len += len(tokens)
            utt_len_list.append(len(tokens))
        return total_len, utt_len_list

    while True:
        total_length, utt_len_list = get_total_len()
        if total_length <= max_length:
            break
        max_len_index = utt_len_list.index(max(utt_len_list))
        tokens_list[max_len_index].pop()
        # return tokens_list


class Metrics(object):

    def __init__(self, score_file_path:str):
        super(Metrics, self).__init__()
        self.score_file_path = score_file_path
        self.segment = 10

    def __read_socre_file(self, score_file_path):
        sessions = []
        one_sess = []
        with open(score_file_path, 'r',encoding='utf-8') as infile:
            i = 0
            for line in infile.readlines():
                i += 1
                tokens = line.strip().split('\t')
                one_sess.append((float(tokens[0]), int(tokens[1])))
                if i % self.segment == 0:
                    one_sess_tmp = np.array(one_sess)
                    if one_sess_tmp[:, 1].sum() > 0:
                        sessions.append(one_sess)
                    one_sess = []
        return sessions


    def __mean_average_precision(self, sort_data):
        #to do
        count_1 = 0
        sum_precision = 0
        for index in range(len(sort_data)):
            if sort_data[index][1] == 1:
                count_1 += 1
                sum_precision += 1.0 * count_1 / (index+1)
        return sum_precision / count_1


    def __mean_reciprocal_rank(self, sort_data):
        sort_lable = [s_d[1] for s_d in sort_data]
        assert 1 in sort_lable
        return 1.0 / (1 + sort_lable.index(1))

    def __precision_at_position_1(self, sort_data):
        if sort_data[0][1] == 1:
            return 1
        else:
            return 0

    def __recall_at_position_k_in_10(self, sort_data, k):
        sort_label = [s_d[1] for s_d in sort_data]
        select_label = sort_label[:k]
        return 1.0 * select_label.count(1) / sort_label.count(1)


    def evaluation_one_session(self, data):
        '''
        :param data: one conversion session, which layout is [(score1, label1), (score2, label2), ..., (score10, label10)].
        :return: all kinds of metrics used in paper.
        '''
        np.random.shuffle(data)
        sort_data = sorted(data, key=lambda x: x[0], reverse=True)
        m_a_p = self.__mean_average_precision(sort_data)
        m_r_r = self.__mean_reciprocal_rank(sort_data)
        p_1   = self.__precision_at_position_1(sort_data)
        r_1   = self.__recall_at_position_k_in_10(sort_data, 1)
        r_2   = self.__recall_at_position_k_in_10(sort_data, 2)
        r_5   = self.__recall_at_position_k_in_10(sort_data, 5)
        return m_a_p, m_r_r, p_1, r_1, r_2, r_5


    def evaluate_all_metrics(self):
        sum_m_a_p = 0
        sum_m_r_r = 0
        sum_p_1 = 0
        sum_r_1 = 0
        sum_r_2 = 0
        sum_r_5 = 0

        sessions = self.__read_socre_file(self.score_file_path)
        total_s = len(sessions)
        for session in sessions:
            m_a_p, m_r_r, p_1, r_1, r_2, r_5 = self.evaluation_one_session(session)
            sum_m_a_p += m_a_p
            sum_m_r_r += m_r_r
            sum_p_1 += p_1
            sum_r_1 += r_1
            sum_r_2 += r_2
            sum_r_5 += r_5

        return (sum_m_a_p/total_s,
                sum_m_r_r/total_s,
                  sum_p_1/total_s,
                  sum_r_1/total_s,
                  sum_r_2/total_s,
                  sum_r_5/total_s)