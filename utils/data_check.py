import os
from shutil import copyfile

ALPHABET = list('abcdefghijklnmopqrstuvwxyzABCDEFGHIJKLNMOPQRSTUVWXYZ')

init_path = '/home/user/VDO/Dataset/v_caption/'
txt_path = '/home/user/detectron/data/v_caption_detection/alphabet_patch_val.txt'
with open(txt_path) as f:
    lines = f.readlines()

for img_info in lines:
    name = img_info.split(' ')[0].split('/')[-1]
    label = img_info.split(' ')[-1]
    name_edited = name.split('.')[0] + '_' + ALPHABET[int(label)] + '.jpg'
    copyfile(os.path.join(init_path, img_info.split(' ')[0]), os.path.join(init_path, 'check', name_edited))


    print('debug')