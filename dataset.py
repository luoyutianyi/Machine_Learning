"""
完成数据集的准备
"""

import os
import re
from torch.utils.data import DataLoader,Dataset


def tokenlize(content):
    content = re.sub("<.*?>"," ",content)
    fileters = [r"\.",":",'\t','\n','\x97','\x96','#','$','%','&']
    content = re.sub("|".join(fileters)," ",content)
    tokens = [i.strip() for i in content.split()]
    return tokens

class ImdbDataset(Dataset):
    def __init__(self, train=True):
        self.train_data_path = r"E:\python_file\Practice\文本情感分析\data\train"
        self.test_data_path = r"E:\python_file\Practice\文本情感分析\data\test"
        data_path = self.train_data_path if train else self.test_data_path

        #把所有文件名放入列表
        temp_data_path = [os.path.join(data_path+r"\pos"),os.path.join(data_path+r"\neg")]
        self.total_file_path = [] #所有评论文件的路径
        for path in temp_data_path:
            file_name_list = os.listdir(path)
            file_path_list = [os.path.join(path,i) for i in file_name_list if i.endswith(".txt")]
            self.total_file_path.extend(file_path_list)


    def __getitem__(self, index):
        file_path = self.total_file_path[index]
        #获取label
        label_str = file_path.split("\\")[-2]
        label = 0 if label_str == "neg" else 1
        #获取内容
        tokens = tokenlize(open(file_path).read())
        return tokens, label


    def __len__(self):
        return len(self.total_file_path)


def get_dataloader(train=True):
    imdb_dataset = ImdbDataset(train)
    data_loader = DataLoader(imdb_dataset,batch_size=2,shuffle=True)
    return data_loader


if __name__== '__main__':
    # imdb_dataset = ImdbDataset()
    # print(imdb_dataset[0])
    #print(tokenlize(my_str))
    for idx, (input, target) in enumerate(get_dataloader()):
        print(idx)
        print(input)
        print(target)
        break
