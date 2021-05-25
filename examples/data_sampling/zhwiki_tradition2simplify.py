from opencc import OpenCC
from tqdm import tqdm
import os
cc = OpenCC('t2s')
path = "./mydata/zhwikitext/"
write_path = "./mydata/zhwikitext_simplify/"

def get_text_list():
    filename_list = os.listdir(path)
    pageid_list = [int(x.split(".")[0]) for x in filename_list]
    return pageid_list


def zh_Traditional_to_Simplified(page_list):
    pbar = tqdm(total=len(page_list), desc="t2s")
    for index, page in enumerate(page_list):
        rf = open(path + str(page) + ".txt", 'r')
        t_text  = rf.read().strip()
        s_text = cc.convert(t_text)
        rf.close()
        with open(write_path + str(page) + ".txt", 'w') as wf:
            wf.write(s_text)
        pbar.update(1)
        if index == 2:
            break
    pbar.close()



if __name__ == '__main__':
    page_list = get_text_list()
    zh_Traditional_to_Simplified(page_list)

