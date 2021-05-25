import logging
import numpy as np
import json
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logger.addHandler(console)

from tqdm import tqdm
# TASK_TOKEN = {"insert": "[INS]", "delete": "[DEL]", "replace": "[REP]"}

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, utt_list=None, utt_b=None, label=None):
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
        self.utt_list = utt_list
        self.utt_b = utt_b
        self.label = label


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self,data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()
    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()


class InsertDataProcessor(DataProcessor):
    def __init__(self):
        super(DataProcessor, self).__init__()
        # self.train_intv=train_intv
        # self.dev_intv=dev_intv
        # self.test_intv=test_intv


    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(data_dir, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(data_dir, "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(data_dir, "test")

    def get_labels(self):
        """See base class."""
        return ["1", "2", "3"]

    def _create_examples(self, data_dir, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        i = 0
        with open(data_dir, 'r', encoding='utf-8') as rf:
            for line in rf:
                line = line.strip()
                if line:
                    guid = "%s-%s" % (set_type, i)
                    labutt_list = line.split("\t")
                    # print(labutt_list)
                    if len(labutt_list) != 6:
                        print("not load example ", i)
                    else:
                        examples.append(InputExample(guid=guid, utt_list=labutt_list[2:], label=labutt_list[1]))
                    i += 1
        print(data_dir + " len of examples is:", len(examples))
        return examples


class DeleteDataProcessor(DataProcessor):
    def __init__(self):
        super(DataProcessor, self).__init__()
        # self.train_intv=train_intv
        # self.dev_intv=dev_intv
        # self.test_intv=test_intv


    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(data_dir, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(data_dir, "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(data_dir, "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def _create_examples(self, data_dir, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        i = 0
        with open(data_dir, 'r', encoding='utf-8') as rf:
            for line in rf:
                line = line.strip()
                if line:
                    guid = "%s-%s" % (set_type, i)
                    labutt_list = line.split("\t")
                    if len(labutt_list) != 6:
                        print("not load example ", i)
                    else:
                        examples.append(InputExample(guid=guid, utt_list=labutt_list[1:-1], utt_b=labutt_list[-1], label=labutt_list[0]))
                    i += 1
        print(data_dir + " len of examples is:", len(examples))
        return examples


class ReplaceDataProcessor(DataProcessor):
    def __init__(self):
        super(DataProcessor, self).__init__()
        # self.train_intv=train_intv
        # self.dev_intv=dev_intv
        # self.test_intv=test_intv

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(data_dir, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(data_dir, "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(data_dir, "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3", "4"]

    def _create_examples(self, data_dir, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        i = 0
        with open(data_dir, 'r', encoding='utf-8') as rf:
            for line in rf:
                line = line.strip()
                if line:
                    guid = "%s-%s" % (set_type, i)
                    labutt_list = line.split("\t")
                    if len(labutt_list) != 6:
                        print("not load example ", i)
                    else:
                        examples.append(InputExample(guid=guid, utt_list=labutt_list[1:], label=labutt_list[0]))
                    i += 1
        print(data_dir + " len of examples is:", len(examples))
        return examples



class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, sep_pos, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.sep_pos = sep_pos
        self.label_id = label_id


def convert_examples_to_features_sptoken_sot(task, examples, label_list,max_seq_length,
                                 tokenizer,
                                 cls_token_at_end=False,
                                 cls_token='[CLS]',
                                 cls_token_segment_id=1,
                                 sep_token='[SEP]',
                                 sep_token_extra=False,
                                 pad_on_left=False,
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 sequence_b_segment_id=1,
                                 mask_padding_with_zero=True,
                                 sot_token = "[SOT]"):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    label_map = {label : i for i, label in enumerate(label_list)}
    # sp_token = TASK_TOKEN[task]
    features = []
    pbar = tqdm(total=len(examples),desc="convert_examples_to_features")
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        # tokens_a = tokenizer.tokenize(example.text_a)
        if task == "delete_v3":
            max_len = max_seq_length - len(example.utt_list) - 1 - 3 # remove cls sep sep SOTS
            utt_list = example.utt_list + [example.utt_b]
            tokens_list = []
            for utt in utt_list:
                tokens_list.append(tokenizer.tokenize(utt))
            _truncate_seq_pair(tokens_list, max_len)

            tokens_b = [sot_token] + tokens_list[-1] + [sep_token]
            tokens_list = tokens_list[:-1]

        else:
            max_len = max_seq_length - len(example.utt_list) - 1 - 1 # remove SOTS sep and cls

            tokens_list = []
            for utt in example.utt_list:
                tokens_list.append(tokenizer.tokenize(utt))
            _truncate_seq_pair(tokens_list, max_len)
            tokens_b = None

        tokens_a = []

        for tokens in tokens_list:
            tokens_a += ([sot_token] + tokens)

        tokens = tokens_a + [sep_token]

        segment_ids = [sequence_a_segment_id] * len(tokens)

        if tokens_b:
            tokens += tokens_b
            segment_ids += [sequence_b_segment_id] * len(tokens_b)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
            # sep_pos = [x-1 for x in sep_pos]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids #[CLS]s1[SEP]s2[SEP]s3[SEP]S4[SEP]res[SEP]

        sep_pos = []
        for index, token in enumerate(tokens):
            if token == sot_token:
                sep_pos.append(index)

        if task == "insert":
            assert len(sep_pos) == 4
        else:
            assert len(sep_pos) == 5
        # if len(sep_pos) != 5:
        #     print(example.utt_list)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # if(not example.text_a and not example.text_b):
        #     input_ids=[]
        #     segment_ids=[]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        # print("len(input_ids)",input_ids,len(input_ids))
        # print("len(input_mask)",input_mask,len(input_mask))
        # print("len(segment_ids)",segment_ids,len(segment_ids))
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        # if output_mode == "classification":
        # print(label_map)
        if(example.label):
            label_id = label_map[example.label]
        else:
            label_id = None
        # elif output_mode == "regression":
        #     label_id = float(example.label)
        # else:
        #     raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            # logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              sep_pos=sep_pos,
                              label_id=label_id))
        pbar.update(1)
    pbar.close()
    return features



def _truncate_seq_pair(tokens_list, max_length):
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