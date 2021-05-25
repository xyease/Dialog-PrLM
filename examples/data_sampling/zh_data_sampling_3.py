# from nltk.tokenize import sent_tokenize
import random
import os
import json
import re
path = "mydata/zhwikitext_simplify/"
path_insert = "mydata/zh_wiki_insert/"
path_delete = "mydata/zh_wiki_delete_3/"
path_replace = "mydata/zh_wiki_replace_2/"
path_search = "mydata/zh_wiki_search/"
pageid_allocate_file = "mydata/zh_pageid_allocate.json"
MAX_WORD_NUM = 450


def sent_tokenize_deeplayer(x):
    sents_temp = re.split('(：|:|,|，|\;|；)', x)
    if not sents_temp[-1]:
        sents_temp = sents_temp[:-1]
    sents = []
    i = 0
    while i < len(sents_temp):
        if i == len(sents_temp) - 1:
            sents.append(sents_temp[i].strip())
            break
        else:
            sents.append((sents_temp[i] + sents_temp[i + 1]).strip())
            i += 2
    if len(sents) == 1:
        return [x]
    else:
        if random.random() < 0.5:
            return [x]
        else:
            return sents


def sent_tokenize(x):
    sents_temp = re.split('(。|！|\!|？|\?)', x)
    if not sents_temp[-1]:
        sents_temp = sents_temp[:-1]
    result_sents = []
    i = 0
    while i < len(sents_temp):
        if i == len(sents_temp) - 1:
            sent = sents_temp[i].strip()
            result_sents.extend(sent_tokenize_deeplayer(sent))
            break
        else:
            sent = (sents_temp[i] + sents_temp[i+1]).strip()
            result_sents.extend(sent_tokenize_deeplayer(sent))
            i += 2
    return result_sents


def allocate_train_dev_test():
    filename_list = os.listdir(path)
    pageid_list = [int(x.split(".")[0]) for x in filename_list]
    random.shuffle(pageid_list)
    # print(pageid_list[:10])
    # print(len(pageid_list))  # 1060131
    insert_train = pageid_list[:67468]  #67468
    # insert_dev = pageid_list[85000:86000]
    insert_test = pageid_list[67468:87468] #20000

    delete_train = pageid_list[87468:154936]
    # delete_dev = pageid_list[172000:173000]
    delete_test = pageid_list[154936:174936]

    replace_train = pageid_list[174936:242404]
    # replace_dev = pageid_list[259000:260000]
    replace_test = pageid_list[242404:262404]

    # train_pageid_list = pageid_list[:1000000]
    # dev_pageid_list = pageid_list[1000000: 1030000]
    # test_pageid_list = pageid_list[1030000: 1060000]
    writejson =  {
                    "insert": {"train": insert_train, "test": insert_test},
                    "delete": {"train": delete_train, "test": delete_test},
                    "replace": {"train": replace_train, "test": replace_test},
                }
    with open(pageid_allocate_file, 'w', encoding='utf-8') as wf:
        json.dump(writejson, wf, ensure_ascii=False, indent=2)

    with open(pageid_allocate_file, "r", encoding="utf-8") as reader:
        text = reader.read()
    pageid_allocate = json.loads(text)
    for key1, value1 in pageid_allocate.items():
        mylen = 0
        for key2, value2 in value1.items():
            mylen += len(value2)
        print(key1, mylen)


def insert(pageid):
    filename = path + str(pageid) + ".txt"
    writefile = path_insert + str(pageid) + ".txt"
    wf = open(writefile, 'w', encoding="utf-8")
    with open(filename, 'r', encoding='utf-8') as rf:
        i = 0
        pre_sent = []
        for line in rf:
            if i==0:
                title = line.strip()[7:]
                i+=1
                continue
            line = line.strip()
            line = line.replace('*', '')
            # print(line)
            # if len(line) < 50:
            #     continue
            temp_line = line.replace(" ", '')
            if "=参考资料=" in temp_line or "=外部连结=" in temp_line:
                break

            sent_list = sent_tokenize(line)
            if line and len(sent_list) >= 4:
                post_sent = []
                i = 0
                while i < len(sent_list) - 1:
                    post_sent.append([sent_list[i], sent_list[i+1]])
                    i += 2
                random.shuffle(post_sent)
                # start1_range = len(sent_list) - 1 - 2 - 1
                # start1 = random.randint(0, start1_range)
                # post_sent = []
                # post_sent.append([sent_list[start1], sent_list[start1 + 1]])
                # start2 = random.randint(start1 + 2, len(sent_list) - 2)
                # post_sent.append([sent_list[start2], sent_list[start2 + 1]])
                # # if len(sent_list[0]) < 20:
                # #     continue
                if pre_sent:
                    for pre_2sent, post_2sent in zip(pre_sent, post_sent):

                        len_word_list = 0
                        for sent in [pre_2sent[0], pre_2sent[1], post_2sent[0], post_2sent[1]]:
                            len_word_list += len(sent)
                        if len_word_list > MAX_WORD_NUM:
                            continue

                        random_seed = random.randint(0, 2)
                        if random_seed == 0:
                            writestr = "\t".join(["0\t1", pre_2sent[0], pre_2sent[1], post_2sent[0], post_2sent[1]])
                        elif random_seed == 1:
                            writestr = "\t".join(["0\t2", pre_2sent[0], post_2sent[0], pre_2sent[1], post_2sent[1]])
                        else:
                            writestr = "\t".join(["0\t3", pre_2sent[0], post_2sent[0], post_2sent[1], pre_2sent[1]])
                        wf.write(writestr + '\n')
                    pre_sent = []
                else:
                    pre_sent = post_sent
            i += 1
    wf.close()
    print("insert: success processing pageid " + str(pageid))


def get_allpageid(task_name):
    with open(pageid_allocate_file, "r", encoding="utf-8") as reader:
        text = reader.read()
    pageid_allocate = json.loads(text)
    return pageid_allocate[task_name]["train"], pageid_allocate[task_name]["test"]


def generate_insert_dataset():
    # with open(pageid_allocate_file, "r", encoding="utf-8") as reader:
    #     text = reader.read()
    # pageid_allocate = json.loads(text)
    # pageid_list = pageid_allocate["train_pageid_list"] + pageid_allocate["dev_pageid_list"] + pageid_allocate["test_pageid_list"]
    # print(len(pageid_list))
    train, test = get_allpageid("insert")
    pageid_list = train + test
    i = 0
    for pageid in pageid_list:
        insert(pageid)
        i += 1
        # if i == 20:
        #     break


def delete(pageid):
    filename = path + str(pageid) + ".txt"
    writefile = path_delete + str(pageid) + ".txt"

    wf = open(writefile, 'w', encoding="utf-8")
    with open(filename, 'r', encoding='utf-8') as rf:
        i = 0
        for line in rf:
            line = line.replace('*', '')
            temp_line = line.replace(" ", '')
            if "=参考资料=" in temp_line or "=外部连结=" in temp_line:
                break
            if i == 0:
                title = line.strip()[7:]
                i += 1
                continue

            line = line.strip()
            sent_list = sent_tokenize(line)

            # small_sent = []
            # for sent in sent_list:
            #     sent = sent.replace(', ', ',')
            #     pattern = r',|;'
            #     small_sent.extend(re.split(pattern, sent))
            #
            # final_small_sent = []
            # for index, sent in enumerate(small_sent):
            #     if sent.isdigit() and index > 0:  ##去除数字的情况
            #         final_small_sent[-1] += (" " + sent)
            #     else:
            #         final_small_sent.append(sent)
            j = 0
            while len(sent_list) >= 5 and j+5 <= len(sent_list):
                # random_num = random.randint(0, len(sent_list) - 5)
                # begin_5sent = sent_list[random_num: random_num + 5]

                begin_5sent = sent_list[j:j+5]
                j += 5
                assert len(begin_5sent) == 5

                len_word_list = 0
                for sent in sent_list:
                    len_word_list += len(sent)

                if len_word_list > MAX_WORD_NUM:
                    continue

                random_seed = random.randint(0, 3)
                delete_sent = begin_5sent.pop(random_seed)  #todo
                writestr = str(random_seed) + "\t" + "\t".join(begin_5sent) + '\t' + delete_sent
                wf.write(writestr + '\n')
            i += 1

    wf.close()
    print("delete: success processing pageid " + str(pageid))


def generate_delete_dataset():
    # with open(pageid_allocate_file, "r", encoding="utf-8") as reader:
    #     text = reader.read()
    # pageid_allocate = json.loads(text)
    # pageid_list = pageid_allocate["train_pageid_list"] + pageid_allocate["dev_pageid_list"] + pageid_allocate["test_pageid_list"]
    # print(len(pageid_list))
    train, test = get_allpageid("delete")
    pageid_list = train + test
    i = 0
    for pageid in pageid_list:
        delete(pageid)
        i += 1
        # if i == 10:
        #     break


def replace(pageid, randompageid, datatype):
    filename1 = path + str(pageid) + ".txt"
    filename2 = path + str(randompageid) + ".txt"
    writefile = path_replace + str(pageid) + ".txt"
    rf1 = open(filename1, 'r', encoding="utf-8")
    rf2 = open(filename2, 'r', encoding="utf-8")
    wf = open(writefile, 'w', encoding="utf-8")

    j = 0
    all_sent_list = []
    for line in rf2:
        line = line.replace('*', '')
        temp_line = line.replace(" ", '')
        if "=参考资料=" in temp_line or "=外部连结=" in temp_line:
            break
        if j == 0:
            title = line.strip()[7:]
            j += 1
            continue
        line = line.strip()
        sent_list = sent_tokenize(line)
        if line and len(sent_list) > 4: ## todo 不同于英文这里设置为4
            all_sent_list.extend(sent_list)
        j += 1
    if len(all_sent_list) == 0:
        return False

    i = 0
    for line in rf1:
        line = line.replace('*', '')
        temp_line = line.replace(" ", '')
        if "=References=" in temp_line or "=Literature=" in temp_line:
            break
        if i == 0:
            title = line.strip()[7:]
            i += 1
            continue
        line = line.strip()
        sent_list = sent_tokenize(line)
        k = 0
        while line and len(sent_list) >= 5 and k+5 <= len(sent_list):
            random_seed = random.randint(0, len(all_sent_list) - 1)
            targetsent = all_sent_list[random_seed]

            random_seed = random.randint(0, 4)
            begin_5sent = sent_list[k:k+5]
            begin_5sent[random_seed] = targetsent

            word_list_len = 0
            for sent in begin_5sent:   ## todo
                word_list_len += len(sent)
            if word_list_len > MAX_WORD_NUM:
                k += 5
                continue
            writestr = str(random_seed) + "\t" + "\t".join(begin_5sent)
            wf.write(writestr + '\n')
            k += 5
        i += 1

    rf1.close()
    rf2.close()
    wf.close()
    print(datatype + " replace: success processing pageid " + str(pageid))
    return True


def generate_replace_dataset():
    # with open(pageid_allocate_file, "r", encoding="utf-8") as reader:
    #     text = reader.read()
    # pageid_allocate = json.loads(text)
    data = get_allpageid("replace")
    # pageid_list = train + dev + test
    datatype = ["train", "test"]
    # datatype = [train, , "test_pageid_list"]

    for dt, pageid_list in zip(datatype, data):
        # pageid_list = pageid_allocate[dt]
        print(dt, len(pageid_list))
        i = 0
        for pageid in pageid_list:
            flag = False
            # try_num = 0
            while not flag:
                random_pageid = pageid
                while random_pageid == pageid:
                    random_pageid = random.choice(pageid_list)
                flag = replace(pageid, random_pageid, dt)
                # try_num += 1
                # print("try", try_num)
            i += 1
            # if i == 5:
            #     break


def search(pageid):
    filename = path + str(pageid) + ".txt"
    writefile = path_search + str(pageid) + ".txt"
    wf = open(writefile, 'w', encoding="utf-8")

    with open(filename, 'r', encoding='utf-8') as rf:
        i = 0
        for line in rf:
            line = line.replace('*', '')
            temp_line = line.replace(" ", '')
            if "=参考资料=" in temp_line or "=外部连结=" in temp_line:
                break
            if i == 0:
                title = line.strip()[7:]
                i += 1
                continue
            line = line.strip()
            sent_list = sent_tokenize(line)
            k = 0
            while line and len(sent_list) >= 5 and k+5 <= len(sent_list):

                word_list_len = []
                for sent in sent_list[k:k+5]:
                    word_list_len += len(sent)
                if word_list_len > MAX_WORD_NUM:
                    continue

                begin_4sent = sent_list[k:k+4]
                keysent = sent_list[k+4]
                matchsent = sent_list[k+3]
                random.shuffle(begin_4sent)
                count = 0
                for index, item in enumerate(begin_4sent):
                    if item == matchsent:
                        label = index
                        count += 1
                assert count == 1
                writestr = str(label) + "\t" + "\t".join(begin_4sent) + "\t" + keysent
                wf.write(writestr + '\n')
                k += 5
            i += 1

    wf.close()
    print("search: success processing pageid " + str(pageid))


def generate_search_dataset():
    # with open(pageid_allocate_file, "r", encoding="utf-8") as reader:
    #     text = reader.read()
    # pageid_allocate = json.loads(text)
    # pageid_list = pageid_allocate["train_pageid_list"] + pageid_allocate["dev_pageid_list"] + pageid_allocate["test_pageid_list"]
    # print(len(pageid_list))
    train, test = get_allpageid("search")
    pageid_list = train + test
    i = 0
    for pageid in pageid_list:
        search(pageid)
        i += 1
        if i == 10:
            break


if __name__ == '__main__':
    allocate_train_dev_test()
    generate_insert_dataset()
    generate_delete_dataset()
    generate_replace_dataset()
    generate_search_dataset()
    # filename_list = os.listdir(path)
    # pageid_list = [int(x.split(".")[0]) for x in filename_list]
    # print(len(pageid_list))  # 262405

# insert 87468
# delete 87468
# replace 87468


