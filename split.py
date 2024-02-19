import os 
import random
from shutil import copy2
class split_data():
    def __init__(self):
        self.file_path= "./dataset/mammals/"
        self.new_path = "./split_dataset/"
        self.split_rate = [0.7, 0.15, 0.15]
        self.split_names = ["train", "val", "test"]
        self.classname = os.listdir(self.file_path)
    def make_dir(self):
        if os.path.isdir(self.new_path):
            pass
        else:
            os.mkdir(self.new_path)
        for split_name in self.split_names:
            self.split_path = self.new_path +"/"+ split_name
            if os.path.isdir(self.split_path):
                pass
            else:
                os.mkdir(self.split_path)
            for class_name in self.classname:
                class_split_path = os.path.join(self.split_path, class_name)
                if os.path.isdir(class_split_path):
                    pass
                else:
                    os.mkdir(class_split_path)
    def split(self):
        for class_name in self.classname:
            current_class_data_path = os.path.join(self.file_path, class_name)
            current_all_data = os.listdir(current_class_data_path)
            current_data_length = len(current_all_data)
            current_data_index_list = list(range(current_data_length))
            random.shuffle(current_data_index_list)
            train_path = os.path.join(os.path.join(self.new_path,'train'),class_name)
            val_path = os.path.join(os.path.join(self.new_path,'val'),class_name)
            test_path = os.path.join(os.path.join(self.new_path,'test'),class_name)
            train_stop_flag = current_data_length * self.split_rate[0]
            val_stop_flag = current_data_length * (self.split_rate[0] + self.split_rate[1])
            current_idx = 0
            train_num = 0
            val_num = 0
            test_num = 0
            for i in current_data_index_list:
                src_imgpath = os.path.join(current_class_data_path, current_all_data[i])
                if current_idx <= train_stop_flag:
                    copy2(src_imgpath, train_path)
                    train_num += 1
                elif (current_idx>train_stop_flag) and (current_idx<=val_stop_flag):
                    copy2(src_imgpath, val_path)
                    val_num += 1
                else:
                    copy2(src_imgpath, test_path)
                    test_num += 1
                current_idx += 1 
if __name__ == "__main__":
    split = split_data()
    split.make_dir()
    split.split()
    print("done")