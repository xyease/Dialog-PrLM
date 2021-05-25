import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
BASE_DIR=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from RSmatching_model_2 import RSmatching_model
from transformers import BertTokenizer
from transformers import BertConfig
from transformers import get_linear_schedule_with_warmup, AdamW
import argparse
import torch
import random
import numpy as np
import pickle
import logging
from tqdm import tqdm, trange
from transformers import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from utils_bert_col_wikipretrain_RSmatching_v3 import MyDataProcessorUtt, convert_examples_to_features, Metrics
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
# from torch.utils.data.distributed import DistributedSampler



parser = argparse.ArgumentParser()
## Required parameters
parser.add_argument("--data_dir",
                    default="../../mydata/",
                    type=str,
                    help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
parser.add_argument("--bert_model", default="../../result/zh_wikipretrain_v3/model_save_v1_sptoken3_eot_p2/", type=str,
                    help="Bert pre-trained model selected in the list: bert-base-uncased, "
                    "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                    "bert-base-multilingual-cased, bert-base-chinese.")
parser.add_argument("--orin_bert_model", default="bert-base-chinese", type=str,
                    help="Bert pre-trained model selected in the list: bert-base-uncased, "
                    "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                    "bert-base-multilingual-cased, bert-base-chinese.")
parser.add_argument("--task_name",
                    default="alime",
                    type=str,
                    help="The name of the task to train.")
parser.add_argument("--output_dir",
                    default="../../result/alime/RSmatch2_v3_wikipretrain_3_p2",
                    type=str,
                    help="The output directory where the model predictions and checkpoints will be written.")

parser.add_argument("--temp_score_file_path",
                    default="temp_score_file.txt",
                    type=str,
                    help="temp score_file_path where the model predictions will be written for metrics.")
parser.add_argument("--log_save_path",
                    default="log.txt",
                    type=str,
                    help="log written when training")

# parser.add_argument("--max_utterance_num",
#                     default=10,
#                     type=int,
#                     help="The maximum total utterance number.")
# parser.add_argument("--max_segment_num",
#                     default=5,
#                     type=int,
#                     help="The maximum total segment number.")
# parser.add_argument("--key_utterance_num",
#                     default=3,
#                     type=int,
#                     help="The key utterance number.")
parser.add_argument("--max_seq_length",
                    default=350,
                    type=int,
                    help="The maximum total input sequence length after WordPiece tokenization. \n"
                         "Sequences longer than this will be truncated, and sequences shorter \n"
                         "than this will be padded.")

parser.add_argument("--input_cache_dir",
                    default="input_RSmatch2_v3_zh_wikipretrain_3_p2",
                    type=str,
                    help="Where do you want to store the processed model input")

parser.add_argument("--do_train",
                    default=True,
                    type=bool,
                    help="Whether to run training.")

parser.add_argument("--do_lower_case",
                    default=True,
                    type=bool,
                    help="Set this flag if you are using an uncased model.")
parser.add_argument("--num_train_epochs",
                    default=5,
                    type=float,
                    help="Total number of training epochs to perform.")
parser.add_argument("--train_batch_size",
                    default=32,
                    type=int,
                    help="Total batch size for training.")
# parser.add_argument("--eval_batch_size",
#                     default=4,
#                     type=int,
#                     help="Total batch size for eval.")
parser.add_argument("--learning_rate",
                    default=2e-5,
                    type=float,
                    help="The initial learning rate for Adam.")

parser.add_argument("--warmup_proportion",
                    default=0.1,
                    type=float,
                    help="Proportion of training to perform linear learning rate warmup for. "
                         "E.g., 0.1 = 10%% of training.")
parser.add_argument('--seed',
                    type=int,
                    default=42,
                    help="random seed for initialization")
args = parser.parse_args()
args.temp_score_file_path=os.path.join(args.output_dir,args.temp_score_file_path)
args.log_save_path=os.path.join(args.output_dir,args.log_save_path)

args.input_cache_dir=os.path.join(args.data_dir, args.task_name, args.input_cache_dir)
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
if not os.path.exists(args.input_cache_dir):
    os.makedirs(args.input_cache_dir)

# if not os.path.exists(args.log_save_path):
#     os.makedirs(args.log_save_path)
logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)
handler = logging.FileHandler(args.log_save_path)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
handler.setFormatter(formatter)
console = logging.StreamHandler()
console.setLevel(logging.INFO)

logger.addHandler(handler)
logger.addHandler(console)
# print(args)
logger.info(args)

def set_seed():
    n_gpu = torch.cuda.device_count()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)  # 为所有的GPU设置种子


def get_dataloader(tokenizer, examples_utt,label_list,tag):
    logger.info("start prepare input data")
    cached_train_features_file = os.path.join(args.input_cache_dir, tag+"input.pkl")
    # train_features = None

    try:
        with open(cached_train_features_file, "rb") as reader:
            features_utt = pickle.load(reader)
    except:
        # logger.info("start prepare features_res_lab")
        # features_res_lab = convert_examples_to_features(
        #     examples_res_lab, label_list, args.max_seq_length, tokenizer)
        logger.info("start prepare features_utt")
        features_utt = convert_examples_to_features(
            examples_utt, label_list, args.max_seq_length, tokenizer)
        # logger.info("start prepare features_seg")
        # features_seg = convert_examples_to_features(examples_seg, label_list, args.max_seq_length, tokenizer)
        # # if args.local_rank == -1:
        # logger.info("  Saving train features into cached file %s", cached_train_features_file)
        with open(cached_train_features_file, "wb") as writer:
            pickle.dump(features_utt, writer)

    # logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(examples_utt))
    # print(torch.tensor([f.input_ids for f in features_seg], dtype=torch.long).size())
    # print(torch.tensor([f.input_ids for f in features_utt], dtype=torch.long).size())

    utt_input_ids = torch.tensor([f.input_ids for f in features_utt], dtype=torch.long)
    utt_attention_mask = torch.tensor([f.input_mask for f in features_utt], dtype=torch.long)
    utt_token_type_ids = torch.tensor([f.segment_ids for f in features_utt], dtype=torch.long)
    # seg_input_ids = torch.tensor([f.input_ids for f in features_seg], dtype=torch.long). \
    #     view(-1, args.max_segment_num, args.max_seq_length)
    # seg_token_type_ids = torch.tensor([f.input_mask for f in features_seg], dtype=torch.long). \
    #     view(-1, args.max_segment_num, args.max_seq_length)
    # seg_attention_mask = torch.tensor([f.segment_ids for f in features_seg], dtype=torch.long). \
    #     view(-1, args.max_segment_num, args.max_seq_length)

    # res_input_ids = torch.tensor([f.input_ids for f in features_res_lab], dtype=torch.long)
    # res_token_type_ids = torch.tensor([f.input_mask for f in features_res_lab], dtype=torch.long)
    # res_attention_mask = torch.tensor([f.segment_ids for f in features_res_lab], dtype=torch.long)
    # sep_pos = torch.tensor([f.sep_pos for f in features_utt], dtype=torch.long)
    # true_res_utt_num=torch.tensor([f.true_res_utt_num for f in features_utt], dtype=torch.long)
    #
    # last_utt_ind = torch.tensor([f.last_utt_ind for f in features_utt], dtype=torch.long)
    # last_utt_ind_stop = torch.tensor([f.last_utt_ind_stop for f in features_utt], dtype=torch.long)

    labels = torch.tensor([f.label_id for f in features_utt], dtype=torch.long)
    sot_pos = torch.tensor([f.sot_pos for f in features_utt], dtype=torch.long)
    true_len = torch.tensor([f.true_len for f in features_utt], dtype=torch.long)
    # print(seg_input_ids[0] == seg_input_ids[1])
    # print(utt_input_ids.size(),utt_attention_mask.size(),utt_token_type_ids.size())
    # print(seg_input_ids.size(),seg_token_type_ids.size(),seg_attention_mask.size())
    # print(res_input_ids.size(),res_attention_mask.size(),res_token_type_ids.size())
    # print(labels.size())
    train_data = TensorDataset(utt_input_ids, utt_attention_mask, utt_token_type_ids, labels, sot_pos, true_len)
    if(tag=="train"):
        train_sampler = RandomSampler(train_data)
    else:
        train_sampler = SequentialSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
    return train_dataloader


def eval(model,tokenizer,device,myDataProcessorUtt):
    logger.info("start evaluation")
    uttdatafile = os.path.join(args.data_dir, args.task_name, "test.txt")
    # segdatafile = os.path.join(args.data_dir, args.task_name, "test_seg.txt")
    examples_utt = myDataProcessorUtt.get_test_examples(uttdatafile)
    # examples_seg = myDataProcessorSeg.get_test_examples(segdatafile)
    # print("dev: len(examples_res_lab)", len(examples_res_lab))
    # print("dev: len(examples_utt)", len(examples_utt))
    # print("dev:len(examples_seg)", len(examples_seg))
    label_list = myDataProcessorUtt.get_labels()
    eval_dataloader = get_dataloader(tokenizer, examples_utt,label_list, "valid")
    y_pred = []
    y_label=[]

    metrics = Metrics(args.temp_score_file_path)

    for batch in tqdm(eval_dataloader,desc="Evaluating"):
        batch = tuple(t.to(device) for t in batch)
        utt_input_ids, utt_attention_mask, utt_token_type_ids, labels, sot_pos, true_len = batch
        y_label+=labels.data.cpu().numpy().tolist()
        with torch.no_grad():
            logits = model(input_ids=utt_input_ids, attention_mask=utt_attention_mask,
                          token_type_ids=utt_token_type_ids, labels=None, sot_pos=sot_pos, true_len=true_len)
            # print(logits[:,1].size())
            y_pred += logits[:, 1].data.cpu().numpy().tolist()

    with open(args.temp_score_file_path, 'w', encoding='utf-8') as output:
        for score, label in zip(y_pred, y_label):
            output.write(
                str(score) + '\t' +
                str(int(label)) + '\n'
            )
    result = metrics.evaluate_all_metrics()
    return result


def train(model,tokenizer,device,myDataProcessorUtt,n_gpu):
    uttdatafile=os.path.join(args.data_dir,args.task_name,"train.txt")
    # segdatafile = os.path.join(args.data_dir, args.task_name, "train_seg.txt")
    best_result = [0, 0, 0, 0, 0, 0]
    examples_utt= myDataProcessorUtt.get_train_examples(uttdatafile)
    # examples_seg=myDataProcessorSeg.get_train_examples(segdatafile)
    # print("train: len(examples_res_lab)", len(examples_res_lab))
    # print("train: len(examples_utt)", len(examples_utt))
    # print("train:len(examples_seg)" ,len(examples_seg))
    # print("examples_utt[0]==examples_utt[10]",examples_utt[0].text_a==examples_utt[10].text_a)
    # print("examples_seg[0]==examples_seg[5]",examples_seg[0].text_a==examples_seg[5].text_a)
    num_train_optimization_steps = (len(examples_utt) // args.train_batch_size+1) * args.num_train_epochs
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    label_list = myDataProcessorUtt.get_labels()
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        # betas=(args.adam_beta1, args.adam_beta2),
        # eps=args.adam_epsilon,
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_proportion * num_train_optimization_steps,
        num_training_steps=num_train_optimization_steps
    )
    #
    # optimizer = BertAdam(optimizer_grouped_parameters,
    #                      lr=args.learning_rate,
    #                      warmup=args.warmup_proportion,
    #                      t_total=num_train_optimization_steps)
    train_dataloader=get_dataloader(tokenizer,examples_utt,label_list,"train")
    set_seed()
    model.zero_grad()
    model.train()
    global_step = 0
    for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
        tr_loss = 0
        # nb_tr_examples, nb_tr_steps = 0, 0
        s=0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            model.train()
            batch = tuple(t.to(device) for t in batch)
            utt_input_ids, utt_attention_mask, utt_token_type_ids, labels, sot_pos, true_len = batch
            # print("utt_input_ids.size()",utt_input_ids.size())
            # print("utt_token_type_ids.size()",utt_token_type_ids.size())
            # print("res_input_ids.size()",res_input_ids.size())

            # define a new function to compute loss values for both output_modes
            logits,loss = model(input_ids=utt_input_ids, attention_mask=utt_attention_mask,
                          token_type_ids=utt_token_type_ids, labels=labels, sot_pos=sot_pos, true_len=true_len)
            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            tr_loss += loss.item()
            s+=1
            loss.backward()
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            logger.info('Epoch{} Batch{} - loss: {:.6f}  batch_size:{}'.format(epoch,step, loss.item(), labels.size(0)) )
            global_step += 1
            # if(step%4200==0):
            #     model.eval()
            #     result = eval(model, tokenizer, device, myDataProcessorUtt, myDataProcessorSeg)
            #     logger.info("Evaluation Result: \nMAP: %f\tMRR: %f\tP@1: %f\tR1: %f\tR2: %f\tR5: %f",
            #                 result[0], result[1], result[2], result[3], result[4], result[5])
            #     if (result[3] + result[4] + result[5] > best_result[3] + best_result[4] + best_result[5]):
            #         logger.info("save model")
            #         model_to_save = model
            #
            #         output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
            #         output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
            #
            #         torch.save(model_to_save.state_dict(), output_model_file)
            #         model_to_save.config.to_json_file(output_config_file)
            #         tokenizer.save_vocabulary(args.output_dir)
            #         best_result = result
            #
            #     logger.info("best result")
            #     logger.info("Best Result: \nMAP: %f\tMRR: %f\tP@1: %f\tR1: %f\tR2: %f\tR5: %f",
            #                 best_result[0], best_result[1], best_result[2],
            #                 best_result[3], best_result[4], best_result[5])

        logger.info("average loss(:.6f)".format(tr_loss/s))
        # Save a trained model, configuration and tokenizer
        model.eval()
        result=eval(model, tokenizer, device, myDataProcessorUtt)
        logger.info("Evaluation Result: \nMAP: %f\tMRR: %f\tP@1: %f\tR1: %f\tR2: %f\tR5: %f",
                    result[0], result[1], result[2], result[3], result[4], result[5])
        if(result[3] + result[4] + result[5] > best_result[3] + best_result[4] + best_result[5]):
            logger.info("save model")
            model_to_save = model.module if hasattr(model, 'module') else model

            output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
            output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

            torch.save(model_to_save.state_dict(), output_model_file)
            model_to_save.config.to_json_file(output_config_file)
            tokenizer.save_vocabulary(args.output_dir)
            best_result=result

        logger.info("best result")
        logger.info("Best Result: \nMAP: %f\tMRR: %f\tP@1: %f\tR1: %f\tR2: %f\tR5: %f",
                     best_result[0], best_result[1], best_result[2],
                     best_result[3], best_result[4], best_result[5])


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    set_seed()

    myDataProcessorUtt = MyDataProcessorUtt()
    # myDataProcessorSeg=MyDataProcessorSeg(args.max_segment_num)
    label_list = myDataProcessorUtt.get_labels()
    num_labels = len(label_list)

    tokenizer = BertTokenizer.from_pretrained(args.orin_bert_model, do_lower_case=args.do_lower_case)
    tokenizer.add_tokens(["[SOT]"], special_tokens=True)
    print(len(tokenizer))
    # tokenizer.save_pretrained(training_args.output_dir)
    a = tokenizer.tokenize("tokenizer")#我 去 不 早 说 发韵 达 能 到 我家 那儿 我 就 能 拿到	韵达 不发 的 哦
    print(a)
    ids = tokenizer.convert_tokens_to_ids(["[CLS]", "[SOT]"] + a)
    print(ids)

    # a = tokenizer.tokenize("[PAD][SOT]tokenizer[SOT]sst")
    # ids = tokenizer.convert_tokens_to_ids(a)
    # print(len(tokenizer))
    config = BertConfig.from_pretrained(args.bert_model)
    config.num_labels = num_labels
    print(config.vocab_size)
    # config.device = device
    if args.do_train:
        logger.info("start train...")
        # output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        # if(os.path.exists(output_model_file)):
        #     logger.info("load dict...")
        #     model_state_dict = torch.load(output_model_file)
        #     model = BertForSequenceClassificationTS.from_pretrained(args.bert_model, config=config,
        #                                                             state_dict=model_state_dict, num_labels=num_labels)
        # else:
        model = RSmatching_model.from_pretrained(args.bert_model, config=config, device=device)
        # tokenizer1 = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=False)
        # a = tokenizer1.tokenize("[SOT]tokeNIzer[SOT]sSt")
        # ids = tokenizer.convert_tokens_to_ids(a)
        # model.resize_token_embeddings(len(tokenizer))

        model.to(device)
        if n_gpu > 1:
            model = torch.nn.DataParallel(model)
        train(model, tokenizer, device, myDataProcessorUtt, n_gpu)
    else:
        logger.info("start test...")
        logger.info("load dict...")
        # output_model_file = os.path.join(args.output_dir,  WEIGHTS_NAME)
        # model_state_dict = torch.load(output_model_file)
        model = RSmatching_model.from_pretrained(args.output_dir, config=config, device=device)
        model.to(device)
        if n_gpu > 1:
            model = torch.nn.DataParallel(model)
        result = eval(model, tokenizer, device, myDataProcessorUtt)
        print("Evaluation Result: \nMAP: %f\tMRR: %f\tP@1: %f\tR1: %f\tR2: %f\tR5: %f",
                    result[0], result[1], result[2], result[3], result[4], result[5])
        print(result)

if __name__ == "__main__":
    main()
