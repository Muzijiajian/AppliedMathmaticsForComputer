# encoding=utf-8
import argparse
import os
import random   # 调用shuffle函数需要random包的支持
from sklearn.utils import shuffle
import numpy as np

def traverse_folder(src_folder):
    '''
    递归遍历包含所有分类目录的文件夹,将每一个目录的文件名字和目录内的图片文件保存
    :param src_folder: source folder, 分类目录源文件
    :return: category_list, 各个目录的编号的链表
    '''
    category_list = []
    category_images_list = []
    category_num = 0
    # 需要注意的是默认会将mac系统产生的.DS_Store(用来用来存储这个文件夹的显示属性,为隐藏文件)
    # 使用os.listdir(path)会自动按照文件名的顺序自动有小到大进行排列
    # print os.listdir(src_folder)
    image_folders = os.listdir(src_folder)[1:]
    # print len(image_folders), image_folders
    for image_file in image_folders:
        images_path = src_folder + image_file + '/'
        images = os.listdir(images_path)
        category_list.append(image_file)
        # print images
        #category_list.append(image_path)
        for image in images:
            # 这里不以'\'结尾
            image_path = images_path + image
            category_images_list.append((image_path, category_num))
            # print category_list
        category_num = category_num + 1
    #print category_num
    #print category_list
    return category_list, category_images_list

def set_to_csv_file(data_set, filename):
    '''
    将包含每一张图片信息的完备数据按行写入一个txt文档 e.g. /.../.../.../image01 01
    :param data_set: 元组组成的链表数据集
    :param filename: 写入的文档名字
    :return: None
    '''
    try:
        fhandle = open(filename, 'wb')
    except:
        print 'File can not be opened'
    for item in data_set:
        line = item[0] + ' ' + str(item[1]) + '\n'
        fhandle.write(line)
    fhandle.close()

def set_to_folder_file(folder_list, filename):
    '''
    只是将每一个文件目录写入一个txt文档,并未包含整个路径
    :param folder_list: 文件目录链表
    :param filename: 需要写入的文件名
    :return: None
    '''
    try:
        fhandle = open(filename, 'wb')
    except:
        print 'File can not be opened'
    for item in folder_list:
        line = item + '\n'
        fhandle.write(line)
    fhandle.close()

def main():
    category_path =  os.path.abspath('101_ObjectCategories')
    parser = argparse.ArgumentParser()
    #parser.add_argument("square", help="display a square of a  given number", type=int)
    parser.add_argument('--root','-r', dest='root', help="Root directory of video files specified in input file",
                            default=category_path)
    parser.add_argument('--output-dir', '-o', dest='output_dir',
                            default="result")
    args = parser.parse_args()
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    save_image_set = os.path.join(args.output_dir, 'image_set.txt')
    save_folder_set = os.path.join(args.output_dir, 'images_folder.txt')
    save_train_set = os.path.join(args.output_dir, 'train.txt')
    save_val_set = os.path.join(args.output_dir, 'val.txt')
    src_folder = args.root
    print save_train_set
    print save_val_set
    print src_folder
    if not src_folder.endswith('/'):
        src_folder  = src_folder + '/'
        print 'New source folder', src_folder
    # 将源目录文件传入生成可读链表准备分别写进 train.txt val.txt, 先将所有文件写入image_set.txt 和 images_folder.txt
    category_list, category_images_list = traverse_folder(src_folder)
    set_to_csv_file(category_images_list, save_image_set)
    set_to_folder_file(category_list, save_folder_set)
    # 打乱数据集分别制作 train.txt val.txt
    print 'shuffle data...'
    data_set = np.array(category_images_list)
    data_set = shuffle(data_set,random_state=0)
    train = data_set[:-1000]
    test = data_set[-1000:]
    random.shuffle(category_images_list)
    # print "随机排序列表 : ",  category_images_list
    set_to_csv_file(train, save_train_set)
    set_to_csv_file(test, save_val_set)
#print args.square**2

if __name__ == '__main__':
    main()