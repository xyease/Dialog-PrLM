import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
import sys
BASE_DIR=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from electra_wiki_pretrain_model_3 import ElectraWIKIpretrainmodel
from transformers import ElectraTokenizer
from transformers import ElectraConfig
from transformers import get_linear_schedule_with_warmup, AdamW
import argparse
import torch
import random
import numpy as np
import pickle
import logging
from itertools import cycle
from tqdm import tqdm, trange
from transformers import WEIGHTS_NAME, CONFIG_NAME
from utils_electra_wikipretrain_sot_3 import InsertDataProcessor, DeleteDataProcessor, ReplaceDataProcessor, convert_examples_to_features_sptoken_sot
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
# from torch.utils.data.distributed import DistributedSampler

parser = argparse.ArgumentParser()
## Required parameters
parser.add_argument("--data_dir",
                    default="../../mydata/zh_wikipretrain_v3/",
                    type=str,
                    help="The input data dir. Should contain the .tsv files (or other data files) for the task.")

parser.add_argument("--bert_model", default="hfl/chinese-electra-180g-base-discriminator", type=str,
                    help="Bert pre-trained model selected in the list: bert-base-uncased, "
                    "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                    "bert-base-multilingual-cased, bert-base-chinese.")

parser.add_argument("--pretrain_tasks",
                    default=["insert", "delete_v3", "replace_v2"],
                    type=list,
                    help="The name of the task to train.")

parser.add_argument("--output_dir",
                    default="../../result/electra_base_zh_wikipretrain_v3/model_save_v1_sptoken3_eot_p2",
                    type=str,
                    help="The output directory where the model predictions and checkpoints will be written.")

parser.add_argument("--log_save_path",
                    default="log.txt",
                    type=str,
                    help="log written when training")

parser.add_argument("--max_seq_length",
                    default=512,
                    type=int,
                    help="The maximum total input sequence length after WordPiece tokenization. \n"
                         "Sequences longer than this will be truncated, and sequences shorter \n"
                         "than this will be padded.")

parser.add_argument("--input_cache_dir",
                    default="electra_base_input_cache_v1_sptoken3_eot_p2",
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
                    default=1,
                    type=float,
                    help="Total number of training epochs to perform.")

parser.add_argument("--train_batch_size",
                    default=16,
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

parser.add_argument('--save_steps',
                    type=int,
                    default=10000,
                    help="save and eval steps if do_save_byepoch is False")

parser.add_argument('--do_save_byepoch',
                    type=bool,
                    default=False,
                    help="whether save by epochs or steps")


args = parser.parse_args()

args.log_save_path = os.path.join(args.output_dir, args.log_save_path)
args.input_cache_dir=os.path.join(args.data_dir, args.input_cache_dir)
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
if not os.path.exists(args.input_cache_dir):
    os.makedirs(args.input_cache_dir)

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
logger.info(args)


def set_seed():
    n_gpu = torch.cuda.device_count()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)  # 为所有的GPU设置种子


def save_model(model, tokenizer=None):

    logger.info("Saving model to %s", args.output_dir)
    # Save a trained model and configuration using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`

    # if not isinstance(model, PreTrainedModel):
        # logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    # else:
    #     model.save_pretrained(args.output_dir)

    if tokenizer is not None:
        tokenizer.save_pretrained(args.output_dir)

    # Good practice: save your training arguments together with the trained model
    torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
    ## todo

    # torch.save(optimizer.state_dict(), os.path.join(args.output_dir, "optimizer.pt"))
    # torch.save(lr_scheduler.state_dict(), os.path.join(args.output_dir, "scheduler.pt"))


def get_dataloader(task_index, tokenizer, examples, label_list, tag):
    task = args.pretrain_tasks[task_index]
    logger.info("start prepare " + tag + " input data for task " + task)

    cached_train_features_file = os.path.join(args.input_cache_dir, "{}_{}_{}_{}_input.pkl".format(tokenizer.__class__.__name__,
                str(args.max_seq_length), task, tag))
    # train_features = None

    try:
        with open(cached_train_features_file, "rb") as reader:
            features = pickle.load(reader)
    except:
        logger.info("start prepare features")
        features = convert_examples_to_features_sptoken_sot(task, examples, label_list, max_seq_length=args.max_seq_length, tokenizer=tokenizer)
        logger.info("Saving train features into cached file %s", cached_train_features_file)
        with open(cached_train_features_file, "wb") as writer:
            pickle.dump(features, writer)

    # logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(examples))

    input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    token_type_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    attention_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    sep_pos = torch.tensor([f.sep_pos for f in features], dtype=torch.long)
    labels = torch.tensor([f.label_id for f in features], dtype=torch.long)

    print(input_ids[:3])

    data = TensorDataset(input_ids, token_type_ids, attention_mask, sep_pos, labels)
    if tag == "train":
        sampler = RandomSampler(data)
    else:
        sampler = SequentialSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=args.train_batch_size)
    return dataloader


def eval(model, tokenizer, device, dataprocessor_list, mode):
    logger.info("start evaluation/test")
    # uttdatafile = os.path.join(args.data_dir, args.task_name, "test.txt")
    ACCs = dict()
    for task_index, dataprocessor in enumerate(dataprocessor_list):
        task = args.pretrain_tasks[task_index]
        if mode == "eval":
            readinfile = os.path.join(args.data_dir, task + "_dev.txt")
            examples = dataprocessor.get_dev_examples(readinfile)
        else:
            readinfile = os.path.join(args.data_dir, task + "_test.txt")
            examples = dataprocessor.get_test_examples(readinfile)
        label_list = dataprocessor.get_labels()
        y_pred = []
        y_label = []

        eval_dataloader = get_dataloader(task_index, tokenizer, examples, label_list, mode)

        for batch in tqdm(eval_dataloader, desc=mode + task):
            batch = tuple(t.to(device) for t in batch)
            input_ids, token_type_ids, attention_mask, sep_pos, labels = batch
            y_label += labels.data.cpu().numpy().tolist()
            with torch.no_grad():
                logits, _ = model(task=task, input_ids=input_ids, token_type_ids=token_type_ids,
                                     attention_mask=attention_mask, sep_positions=sep_pos, labels=None)
                y_pred += logits.data.cpu().numpy().tolist()

        preds = np.argmax(y_pred, axis=1)
        acc = (preds == y_label).mean() #todo
        ACCs[task] = acc
    return ACCs


def eval_and_save(model, tokenizer, device, dataprocessor_list, best_mean_acc, best_result, global_step):
    model.eval()
    result = eval(model, tokenizer, device, dataprocessor_list, "test")

    mean_acc = sum(result.values()) / len(result)
    logger.info("global step: " + str(global_step) + " Evaluation Result: " + str(result) + " Mean ACC " + str(mean_acc))

    if mean_acc > best_mean_acc:
        logger.info("save model")
        # model_to_save = model.module if hasattr(model, 'module') else model
        #
        # output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        # output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
        #
        # torch.save(model_to_save.state_dict(), output_model_file)
        # model_to_save.config.to_json_file(output_config_file)
        # tokenizer.save_vocabulary(args.output_dir)
        save_model(model,  tokenizer)
        best_result = result
        best_mean_acc = mean_acc
        save_step_file = os.path.join(args.output_dir, "save_steps_results.txt")
        with open(save_step_file, 'w') as wf:
            wf.write("save model at step: {} result:{} mean acc:{}".format(global_step, str(best_result), best_mean_acc))

    return best_result, best_mean_acc


def train(model, tokenizer, device, dataprocessor_list, n_gpu):
    # uttdatafile=os.path.join(args.data_dir,args.task_name,"train.txt")
    # segdatafile = os.path.join(args.data_dir, args.task_name, "trainseg.txt")
    # best_result = [0, 0, 0, 0, 0, 0]
    # examples_res_lab, examples_utt= myDataProcessorUtt.get_train_examples(uttdatafile)
    examples_list = []
    t_total = 0

    for task_index, dataprocessor in enumerate(dataprocessor_list):
        task = args.pretrain_tasks[task_index]
        readinfile = os.path.join(args.data_dir, task + "_train.txt")
        examples = dataprocessor.get_train_examples(readinfile)
        steps = int(len(examples) / args.train_batch_size) * args.num_train_epochs
        if steps > t_total:
            t_total = steps
        examples_list.append(examples)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    # label_list = myDataProcessorSeg.get_labels()
    # optimizer = BertAdam(optimizer_grouped_parameters,
    #                      lr=args.learning_rate,
    #                      warmup=args.warmup_proportion,
    #                      t_total=num_train_optimization_steps)
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        # betas=(args.adam_beta1, args.adam_beta2),
        # eps=args.adam_epsilon,
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_proportion * t_total,
        num_training_steps=t_total
    )

    train_dataloader_list = []
    for task_index, examples in enumerate(examples_list):
        label_list = dataprocessor_list[task_index].get_labels()
        train_dataloader = get_dataloader(task_index, tokenizer, examples, label_list, "train")
        train_dataloader_list.append(train_dataloader)

    max_batch_num = max([len(dataloader) for dataloader in train_dataloader_list])

    set_seed()
    model.zero_grad()
    model.train()
    global_step = 0
    best_mean_acc = 0
    best_result = None

    tr_loss = 0
    # nb_tr_examples, nb_tr_steps = 0, 0
    s = 0
    for epoch in trange(int(args.num_train_epochs), desc="Epoch"):

        pbar = tqdm(total=max_batch_num, desc="Iteration")

        actual_train_dataloader_dict = []
        for dataloader in train_dataloader_list:
            if len(dataloader) == max_batch_num:
                actual_train_dataloader_dict.append(dataloader)
            else:
                actual_train_dataloader_dict.append(cycle(dataloader))
        # tasks = ["insert", "delete", "replace"]
        for step, all_task_batch in enumerate(zip(*actual_train_dataloader_dict)):
            model.train()

            multitask_loss = None
            for task, batch in zip(args.pretrain_tasks, all_task_batch):
                batch = tuple(t.to(device) for t in batch)

                input_ids, token_type_ids, attention_mask, sep_pos, labels = batch

                # define a new function to compute loss values for both output_modes
                logits, loss = model(task=task, input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, sep_positions=sep_pos, labels=labels)
                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if multitask_loss == None:
                    multitask_loss = loss
                else:
                    multitask_loss += loss
                logger.info('Epoch{} Step{} - loss: {:.6f}  task:{}'.format(epoch+1, step+1, loss.item(), task))

            multitask_loss.backward()
            tr_loss += multitask_loss.item()/len(args.pretrain_tasks)
            s += 1
            logger.info('Epoch{} Step{} - averageloss: {:.6f}'.format(epoch+1, step+1, multitask_loss.item()/len(args.pretrain_tasks)))
            optimizer.step()
            scheduler.step()
            model.zero_grad()

            global_step += 1

            if not args.do_save_byepoch and global_step % args.save_steps == 0:
                logger.info("to global step {} average loss{:.6f}".format(global_step, tr_loss / s))
                best_result, best_mean_acc = eval_and_save(model, tokenizer, device, dataprocessor_list,
                                                           best_mean_acc, best_result, global_step)
                model.eval()
                # test_result = eval(model, tokenizer, device, dataprocessor_list, "test")

                logger.info("To epoch {} global step {} Best Result {}:".format(epoch+1, global_step, str(best_result)))
                logger.info("To epoch {} global step {} Best Mean ACC {}:".format(epoch+1, global_step, best_mean_acc))
                # logger.info("TESTING RESULT {}".format(str(test_result)))
                tr_loss = 0
                s = 0
            pbar.update()
        pbar.close()

        if args.do_save_byepoch:
            logger.info("to global step {} average loss{:.6f}".format(global_step, tr_loss / s))
            # Save a trained model, configuration and tokenizer
            best_result, best_mean_acc = eval_and_save(model, tokenizer, device, dataprocessor_list,
                                                       best_mean_acc, best_result, global_step)
            model.eval()
            # test_result = eval(model, tokenizer, device, dataprocessor_list, "test")

            logger.info("To epoch {} global step {} Best Result {}:".format(epoch+1, global_step, str(best_result)))
            logger.info("To epoch {} global step {} Best Mean ACC {}:".format(epoch+1, global_step, best_mean_acc))
            # logger.info("TESTING RESULT {}".format(str(test_result)))
            tr_loss = 0
            s = 0

    if not args.do_save_byepoch and global_step % args.save_steps != 0:
        logger.info("to global step {} average loss{:.6f}".format(global_step, tr_loss / s))
        best_result, best_mean_acc = eval_and_save(model, tokenizer, device, dataprocessor_list,
                                                   best_mean_acc, best_result, global_step)
        model.eval()
        # test_result = eval(model, tokenizer, device, dataprocessor_list, "test")
        logger.info("To final global step {} Best Result {}:".format(global_step, str(best_result)))
        logger.info("To final global step {} Best Mean ACC {}:".format(global_step, best_mean_acc))
        # logger.info("TESTING RESULT {}".format(str(test_result)))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    set_seed()

    num_new_tok = 0
    tokenizer = ElectraTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    tokenizer.add_tokens(["[SOT]"], special_tokens=True)
    num_new_tok += 1
    dataprocessor_list = []
    for task in args.pretrain_tasks:
        if task == "insert":
            # tokenizer.add_tokens(["[INS]"])
            # num_new_tok += 1
            dataprocessor_list.append(InsertDataProcessor())
        elif task == "delete_v3":
            # tokenizer.add_tokens(["[DEL]"])
            # num_new_tok += 1
            dataprocessor_list.append(DeleteDataProcessor())
        else:
            # tokenizer.add_tokens(["[REP]"])
            # num_new_tok += 1
            dataprocessor_list.append(ReplaceDataProcessor())

    config = ElectraConfig.from_pretrained(args.bert_model)
    if args.do_train:
        logger.info("start train...")
        # output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        # if(os.path.exists(output_model_file)):
        #     logger.info("load dict...")
        #     model_state_dict = torch.load(output_model_file)
        #     model = BertForSequenceClassificationTS.from_pretrained(args.bert_model, config=config,
        #                                                             state_dict=model_state_dict, num_labels=num_labels)
        # else:
        model = ElectraWIKIpretrainmodel.from_pretrained(args.bert_model, config=config)
        model.resize_token_embeddings(model.config.vocab_size + num_new_tok)
        # tokenizer.save_pretrained(args.output_dir)
        # a=tokenizer.tokenize("tokenizer")
        # ids = tokenizer.convert_tokens_to_ids(["[CLS]", "[SOT]"] + a)
        model.to(device)

        if n_gpu > 1:
            model = torch.nn.DataParallel(model)
        train(model, tokenizer, device, dataprocessor_list, n_gpu)
    else:
        logger.info("start test...")
        # logger.info("load dict...")
        # output_model_file = os.path.join(args.output_dir,  WEIGHTS_NAME)
        # model_state_dict = torch.load(output_model_file)
        model = ElectraWIKIpretrainmodel.from_pretrained(args.bert_model, config=config)

        model.to(device)
        if n_gpu > 1:
            model = torch.nn.DataParallel(model)
        # similar_score(model, tokenizer, device,myDataProcessorSeg)
        result = eval(model, tokenizer, device, dataprocessor_list, "test")
        mean_acc = sum(result.values()) / len(result)
        print("Test Result:" + str(result) + " Mean ACC " + str(mean_acc))


if __name__ == "__main__":
    main()




