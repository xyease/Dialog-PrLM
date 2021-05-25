import json
import os
from tqdm import tqdm

# path = "mydata/zhwikitext_simplify/"
path_insert = "mydata/wiki_insert/"
path_delete = "mydata/wiki_delete_3/"
path_replace = "mydata/wiki_replace_2/"
task_path = {"insert": path_insert, "delete_v3": path_delete, "replace_v2": path_replace}  #"delete_v3": path_delete,
pageid_allocate_file = "mydata/pageid_allocate.json"


def get_allocate_pagelist():
    with open(pageid_allocate_file, "r", encoding="utf-8") as reader:
        text = reader.read()
    pageid_allocate = json.loads(text)
    return pageid_allocate


def generate_train_dev_test_dataset(task, task_filelist_path, task_pageid_allocate):
    task_filename_list = os.listdir(task_filelist_path)
    pageid_list = [int(x.split(".")[0]) for x in task_filename_list]

    allocate_pageid_list = []
    for item in task_pageid_allocate.values():
        allocate_pageid_list.extend(item)

    pageid_list.sort()
    allocate_pageid_list.sort()
    # print(pageid_list[:10], allocate_pageid_list[:10])
    # return
    assert pageid_list == allocate_pageid_list

    for label, pageids in task_pageid_allocate.items():
        print("start processing" + task + label)
        write_filename = "../../DialogPretrain_transformers_3_4_0/examples/mydata/wikipretrain_v6/"+task+"_"+label+".txt"
        print(write_filename)
        wf = open(write_filename, 'w')
        pbar = tqdm(total=len(pageids), desc="processing" + task + label)
        sample_num = 0
        for pageid in pageids:
            read_filename = task_filelist_path + str(pageid) + ".txt"
            with open(read_filename, 'r') as rf:
                for line in rf:
                    line = line.strip()
                    if line:
                        if judge(line, task):
                            wf.write(line + '\n')
                            sample_num += 1
            pbar.update(1)
        pbar.close()
        wf.close()
        print(task + label + ": " + str(sample_num))


def judge(line, task):
    ## en
    return True
    ## ch
    # total_len = 0
    # utts = line.split('\t')
    # if task == "insert_v5":
    #     utts = utts[2:]
    #     MAX_LEN = 96
    # else:
    #     utts = utts[1:]
    #     MAX_LEN = 120
    #
    # # print(utts)
    # for utt in utts:
    #     utt = utt.strip()
    #     if len(utt) <= 24:
    #         return False
    #     total_len += len(utt)
    # if total_len > MAX_LEN:
    #     return True
    # else:
    #     return False


def generate_dataset():
    pageid_allocate = get_allocate_pagelist()
    for task, task_pageid_allocate in pageid_allocate.items():
        if task == "insert":
            task = "insert"
        elif task == "delete":
            task = "delete_v3"
        elif task == "replace":
            task = "replace_v2"
        else:
            assert AssertionError("undefined task")
        # if task == "delete":
        #     task = "delete_v4"
        # else:
        #     continue
        print("start processing" + task)

        generate_train_dev_test_dataset(task, task_path[task], task_pageid_allocate)

def count_data_samples(data_file):
    count = 0
    with open(data_file, 'r') as rf:
        for line in rf:
            line = line.strip()
            if line:
                count += 1
    print(data_file, count)

if __name__ == '__main__':
    generate_dataset()
    # data_file = "/home/xuy/DialogPretrain_transformers_3_4_0/examples/mydata/zh_wikipretrain_v3/replace_v2_test.txt"
    # count_data_samples(data_file)

# en*
# inserttrain: 1581056
# insertdev: 48610
# inserttest: 47336
# deletetrain: 1066728
# deletedev: 31644
# deletetest: 31706
# replacetrain: 1071435
# replacedev: 31428
# replacetest: 32316

# ch
# inserttrain: 638406*    638340
# inserttest: 189199*
# delete_v3train: 508752*  508699
# delete_v3test: 146225*
# replace_v2train: 542458*  542345
# replace_v2test: 160410*

# ch
# insert_v4train: 377628   377623
# insert_v4test: 112374
# delete_v4train: 269164   269156
# delete_v4test: 78310
# replace_v4train: 279575  279572
# replace_v4test: 83440

# ch
# insert_v5train: 377628   377623
# insert_v5test: 112374
# delete_v5train: 269805 269798
# delete_v5test: 78583
# replace_v5train: 279575 279572
# replace_v5test: 83440

# en
# insert_v6train: 1581056
# insert_v6dev: 48610
# insert_v6test: 47336
# delete_v6train: 1070785
# delete_v6dev: 31752
# delete_v6test: 31823
# replace_v6train: 1071435
# replace_v6dev: 31428
# replace_v6test: 32316