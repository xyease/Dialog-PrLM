import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
BASE_DIR=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from bert_multitask_model import Bertmultitaskmodel
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
from utils_bert_RS_multitask_v2 import MyDataProcessorUtt, Metrics, MultiTaskDataset
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
# from bert_multitask_insert_model import testlogger
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
                    default="douban",
                    type=str,
                    help="The name of the task to train.")
parser.add_argument("--multi_tasks",
                    default=["reselect", "insert", "delete","replace"],
                    type=list,
                    help="The name of the task to train.")
parser.add_argument("--output_dir",
                    default="../../result/douban/RSmatch2_v3_wikipretrain_3_p2_multitask_try2",
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
parser.add_argument("--max_seq_length",
                    default=350,
                    type=int,
                    help="The maximum total input sequence length after WordPiece tokenization. \n"
                         "Sequences longer than this will be truncated, and sequences shorter \n"
                         "than this will be padded.")
# parser.add_argument("--input_cache_dir",
#                     default="input_RSmatch2_v3_zh_wikipretrain_3_p2_multitask",
#                     type=str,
#                     help="Where do you want to store the processed model input")
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
                    default=16,
                    type=int,
                    help="Total batch size for training.")
parser.add_argument("--insert_max_num",
                    default=5,
                    type=int,
                    help="insert_max_num.")
parser.add_argument("--delete_max_num",
                    default=5,
                    type=int,
                    help="delete_max_num.")
parser.add_argument("--replace_max_num",
                    default=5,
                    type=int,
                    help="replace_max_num.")
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
args.temp_score_file_path = os.path.join(args.output_dir, args.temp_score_file_path)
args.log_save_path = os.path.join(args.output_dir, args.log_save_path)
# args.input_cache_dir = os.path.join(args.data_dir, args.task_name, args.input_cache_dir)
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
# if not os.path.exists(args.input_cache_dir):
#     os.makedirs(args.input_cache_dir)

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
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
    # pass
    n_gpu = torch.cuda.device_count()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)  # 为所有的GPU设置种子


def get_dataloader(tokenizer, examples, label_list, tag):
    logger.info("start prepare input data")
    # logger.info("***** Running training *****")
    logger.info("Num examples = %d", len(examples))
    train_dataset = MultiTaskDataset(examples, label_list, args, tokenizer)
    if (tag == "train"):
        shuffle = True
    else:
        shuffle = False

    train_dataloader = DataLoader(
      train_dataset,
      batch_size=args.train_batch_size,
      shuffle=shuffle,
    )

    return train_dataloader


def eval(model, tokenizer, device, myDataProcessorUtt):
    logger.info("start evaluation")
    uttdatafile = os.path.join(args.data_dir, args.task_name, "test.txt")
    # segdatafile = os.path.join(args.data_dir, args.task_name, "test_seg.txt")
    examples_utt = myDataProcessorUtt.get_test_examples(uttdatafile)

    label_list = myDataProcessorUtt.get_labels()
    eval_dataloader = get_dataloader(tokenizer, examples_utt, label_list, "valid")
    y_pred, y_label = [], []
    # insert_y_pred, insert_y_label = [], []
    # delete_y_pred, delete_y_label = [], []
    # replace_y_pred, replace_y_label = [], []
    insert_cornum, insert_totnum = 0, 0
    delete_cornum, delete_totnum = 0, 0
    replace_cornum, replace_totnum = 0, 0

    metrics = Metrics(args.temp_score_file_path)
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        for task_key in batch:
            for key in batch[task_key]:
                batch[task_key][key] = batch[task_key][key].to(device)

        with torch.no_grad():
            _, logits, _ = model(task="reselect", **batch["reselect"])
            y_pred += logits[:, 1].data.cpu().numpy().tolist()
            y_label += batch["reselect"]["labels"].data.cpu().numpy().tolist()

            if "insert" in args.multi_tasks:
                _, ins_pred_labels, ins_true_labels = model(task="insert", **batch["insert"])
                if ins_pred_labels is not None:
                    insert_cornum += (ins_pred_labels == ins_true_labels).sum().item()
                    insert_totnum += ins_pred_labels.size()[0]
                # insert_y_pred += ins_pred_labels.data.cpu().numpy().tolist()
                # # print(ins_pred_labels)
                # insert_y_label += ins_true_labels.data.cpu().numpy().tolist()
                # # print(ins_true_labels)
            if "delete" in args.multi_tasks:
                _, del_pred_labels, del_true_labels = model(task="delete", **batch["delete"])
                if del_pred_labels is not None:
                    delete_cornum += (del_pred_labels == del_true_labels).sum().item()
                    delete_totnum += del_pred_labels.size()[0]
                # delete_y_pred += del_pr ed_labels.data.cpu().numpy().tolist()
                # delete_y_label += del_true_labels.data.cpu().numpy().tolist()

            if "replace" in args.multi_tasks:
                _, rep_pred_labels, rep_true_labels = model(task="replace", **batch["replace"])
                if rep_pred_labels is not None:
                    replace_cornum += (rep_pred_labels == rep_true_labels).sum().item()
                    replace_totnum += rep_pred_labels.size()[0]
                # replace_y_pred += rep_pred_labels.data.cpu().numpy().tolist()
                # replace_y_label += rep_true_labels.data.cpu().numpy().tolist()
    # print(insert_y_pred)
    if "insert" in args.multi_tasks:
        logger.info('Insert ACC {:.6f}'.format(insert_cornum/insert_totnum))
    if "delete" in args.multi_tasks:
        # del_acc = (delete_y_pred == delete_y_label).mean()
        logger.info('Delete ACC {:.6f}'.format(delete_cornum/delete_totnum))
    if "replace" in args.multi_tasks:
        # rep_acc = (replace_y_pred == replace_y_label).mean()
        logger.info('Replace ACC {:.6f}'.format(replace_cornum/replace_totnum))

    with open(args.temp_score_file_path, 'w', encoding='utf-8') as output:
        for score, label in zip(y_pred, y_label):
            output.write(
                str(score) + '\t' +
                str(int(label)) + '\n'
            )
    result = metrics.evaluate_all_metrics()
    return result


def train(model, tokenizer, device, myDataProcessorUtt, n_gpu):
    uttdatafile = os.path.join(args.data_dir, args.task_name, "train.txt")
    best_result = [0, 0, 0, 0, 0, 0]
    examples_utt = myDataProcessorUtt.get_train_examples(uttdatafile)
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
    train_dataloader = get_dataloader(tokenizer, examples_utt, label_list, "train")
    set_seed()
    model.zero_grad()
    model.train()
    global_step = 0
    for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
        # tr_loss = 0
        # nb_tr_examples, nb_tr_steps = 0, 0
        # s = 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            model.train()
            # batch = tuple(t.to(device) for t in batch)
            # utt_input_ids, utt_attention_mask, utt_token_type_ids, labels, sot_pos, true_len = batch
            for task_key in batch:
                # print(batch[task_key]["labels"])
                for key in batch[task_key]:
                    batch[task_key][key] = batch[task_key][key].to(device)

            # testlogger.info((batch["insert"]["labels"]))

            # define a new function to compute loss values for both output_modes
            res_loss, ins_loss, del_loss, rep_loss = None, None, None, None

            if "reselect" in args.multi_tasks:
                res_loss, _, _ = model(task="reselect", **batch["reselect"])
                if n_gpu > 1:
                    res_loss = res_loss.mean()  # mean() to average on multi-gpu.
                logger.info('Epoch{} Step{} Response select loss: {:.6f}'.format(epoch, step, res_loss.item()))

            # loss = res_loss
            if "insert" in args.multi_tasks:
                ins_loss, _, _ = model(task="insert", **batch["insert"])
                if n_gpu > 1:
                    ins_loss = ins_loss.mean()
                # loss += ins_loss
                logger.info('Epoch{} Step{} Insert loss: {:.6f}'.format(epoch, step, ins_loss.item()))

            if "delete" in args.multi_tasks:
                del_loss, _, _ = model(task="delete", **batch["delete"])
                if n_gpu > 1:
                    del_loss = del_loss.mean()
                # loss += del_loss
                logger.info('Epoch{} Step{} Delete loss: {:.6f}'.format(epoch, step, del_loss.item()))

            if "replace" in args.multi_tasks:
                rep_loss, _, _ = model(task="replace", **batch["replace"])
                if n_gpu > 1:
                    rep_loss = rep_loss.mean()
                # loss += rep_loss
                logger.info('Epoch{} Step{} Replace loss: {:.6f}'.format(epoch, step, rep_loss.item()))

            loss = None
            for task_tensor_loss in [res_loss, ins_loss, del_loss, rep_loss]:
                if task_tensor_loss is not None:
                    loss = loss + task_tensor_loss if loss is not None else task_tensor_loss

            # tr_loss += loss.item()
            # s += 1
            loss.backward()
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            logger.info('Epoch{} - Step{} Total loss: {:.6f}'.format(epoch, step, loss.item()))
            global_step += 1


        # logger.info("Average loss(:.6f)".format(tr_loss/s))
        # Save a trained model, configuration and tokenizer
        model.eval()
        result = eval(model, tokenizer, device, myDataProcessorUtt)
        logger.info("Evaluation Result: \nMAP: %f\tMRR: %f\tP@1: %f\tR1: %f\tR2: %f\tR5: %f",
                    result[0], result[1], result[2], result[3], result[4], result[5])
        if result[3] + result[4] + result[5] > best_result[3] + best_result[4] + best_result[5]:
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
    config = BertConfig.from_pretrained(args.bert_model)
    config.num_labels = num_labels
    print(config.vocab_size)
    # config.device = device

    if args.do_train:
        logger.info("start train...")
        model = Bertmultitaskmodel.from_pretrained(args.bert_model, config=config)
        model.to(device)
        if n_gpu > 1:
            model = torch.nn.DataParallel(model)
        train(model, tokenizer, device, myDataProcessorUtt, n_gpu)
    else:
        logger.info("start test...")
        logger.info("load dict...")
        # output_model_file = os.path.join(args.output_dir,  WEIGHTS_NAME)
        # model_state_dict = torch.load(output_model_file)
        model = Bertmultitaskmodel.from_pretrained(args.output_dir, config=config)
        model.resize_token_embeddings(len(tokenizer))
        model.to(device)
        if n_gpu > 1:
            model = torch.nn.DataParallel(model)
        result = eval(model, tokenizer, device, myDataProcessorUtt)
        print("Evaluation Result: \nMAP:{:.6f}\tMRR: {:.6f}\tP@1: {:.6f}\tR1: {:.6f}\tR2: {:.6f}\tR5: {:.6f}".format(
                    result[0], result[1], result[2], result[3], result[4], result[5]))
        print(result)

if __name__ == "__main__":
    main()
