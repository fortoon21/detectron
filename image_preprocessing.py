import os
from os import listdir
from os.path import isfile, join
import argparse
import json
import cv2
import random
import hgtk

"""
Class_Map
0 : 가
1 : 고 
2 : 과
3 : 각
4 : 곡
5 : 곽
6 : ㄱ
7 : a
8 : 1
9 : !
"""

CHO = (
    u'ㄱ', u'ㄲ', u'ㄴ', u'ㄷ', u'ㄸ', u'ㄹ', u'ㅁ', u'ㅂ', u'ㅃ', u'ㅅ',
    u'ㅆ', u'ㅇ', u'ㅈ', u'ㅉ', u'ㅊ', u'ㅋ', u'ㅌ', u'ㅍ', u'ㅎ'
)

JOONG = (
    u'ㅏ', u'ㅐ', u'ㅑ', u'ㅒ', u'ㅓ', u'ㅔ', u'ㅕ', u'ㅖ', u'ㅣ',  # 길쭉이 0 ~ 8
    u'ㅗ', u'ㅜ', u'ㅛ', u'ㅠ', u'ㅡ',                             # 넓쩍이 9 ~ 13
    u'ㅘ', u'ㅙ', u'ㅚ', u'ㅝ', u'ㅞ', u'ㅟ', u'ㅢ'                # 길넓이 14 ~ 20
)

JONG = (
    u'', u'ㄱ', u'ㄲ', u'ㄳ', u'ㄴ', u'ㄵ', u'ㄶ', u'ㄷ', u'ㄹ', u'ㄺ',
    u'ㄻ', u'ㄼ', u'ㄽ', u'ㄾ', u'ㄿ', u'ㅀ', u'ㅁ', u'ㅂ', u'ㅄ', u'ㅅ',
    u'ㅆ', u'ㅇ', u'ㅈ', u'ㅊ', u'ㅋ', u'ㅌ', u'ㅍ', u'ㅎ'
)

JONG_reduced = (
    u'', u'ㄱ', u'ㄲ', u'ㄳ', u'ㄴ', u'ㄵ', u'ㄶ', u'ㄷ', u'ㄹ',
    u'ㄺ', u'ㄻ', u'ㄼ', u'ㄾ', u'ㄿ', u'ㅀ', u'ㅁ', u'ㅂ', u'ㅄ', u'ㅅ',
    u'ㅆ', u'ㅇ', u'ㅈ', u'ㅊ', u'ㅋ', u'ㅌ', u'ㅍ', u'ㅎ'
)

ALPHABET='abcdefghijklnmopqrstuvwxyzABCDEFGHIJKLNMOPQRSTUVWXYZ'

NUMBER='0123456789'

special_symbols = ("'", '!', '"', '#', '$', '%', '&', '(', ')', '*', '+', ',', '-', '.', ':', ';',
                   '<', '>', '?', '@', '[', ']', '^', '_', '~', '/')

JAMO = CHO + JOONG + JONG[1:]
FIRST_HANGUL_UNICODE = 0xAC00  # '가'
LAST_HANGUL_UNICODE = 0xD7A3  # '힣'

FIRST_NUMBER_UNICODE = 0x0030  # '0'
LAST_NUMBER_UNICODE = 0x0039  # '9'

FIRST_ALPHABET_UPPER_UNICODE = 0x0041  # 'A'
LAST_ALPHABET_UPPER_UNICODE = 0x005A  # 'Z'

FIRST_ALPHABET_LOWER_UNICODE = 0x0061  # 'a'
LAST_ALPHABET_LOWER_UNICODE = 0x007A  # 'z'

Ix_to_title={0:'hangul', 1:'alphabet', 2:'number', 3:'bghangul', 4:'bgalphabet', 5:'bgnumber'}


def is_jamo(letter):
    return letter in JAMO

def is_hangul(phrase): # TODO: need tuning!!
    for letter in phrase:
        code = ord(letter)
        if is_jamo(letter) or (FIRST_HANGUL_UNICODE<=code<=LAST_HANGUL_UNICODE):
            return True
    return False

def is_alphabet(phrase): # TODO: need tuning!!
    unicode_value = ord(phrase)
    if (FIRST_ALPHABET_UPPER_UNICODE<=unicode_value <= LAST_ALPHABET_UPPER_UNICODE) \
       or (FIRST_ALPHABET_LOWER_UNICODE <= unicode_value <= LAST_ALPHABET_LOWER_UNICODE):
        return True
    else:
        return False

def is_number(phrase): # TODO: need tuning!!
    for letter in phrase:
        code = ord(letter)
        if (FIRST_NUMBER_UNICODE <=code <= LAST_NUMBER_UNICODE):
            return True
    return False



def class_assign(caption,Is_bg):

    if len(caption) == 1:
        if is_hangul(caption):

            if is_jamo(caption):

                if caption in result_dict['{}hangul_jamo'.format(Is_bg)]:
                    result_dict['{}hangul_jamo'.format(Is_bg)][caption] += 1
                else:
                    result_dict['{}hangul_jamo'.format(Is_bg)][caption] = 1
                class_num = -1

            else:

                if caption in result_dict['{}hangul'.format(Is_bg)]:
                    result_dict['{}hangul'.format(Is_bg)][caption] += 1
                else:
                    result_dict['{}hangul'.format(Is_bg)][caption] = 1

                eumso_list = hgtk.letter.decompose(caption)
                first_sung = eumso_list[0]
                middle_sung = eumso_list[1]
                last_sung = eumso_list[2]

                if last_sung == JONG[0]:
                    if middle_sung in JOONG[:9]:
                        config_class_num = 0
                    elif middle_sung in JOONG[9:14]:
                        config_class_num = 1
                    elif middle_sung in JOONG[14:]:
                        config_class_num = 2
                elif last_sung in JONG[1:]:
                    if middle_sung in JOONG[0:9]:
                        config_class_num = 3
                    elif middle_sung in JOONG[9:14]:
                        config_class_num = 4
                    elif middle_sung in JOONG[14:]:
                        config_class_num = 5
                else:
                    raise ValueError

                CHO_class_num = CHO.index(first_sung)
                JOONG_class_num = JOONG.index(middle_sung)
                JONG_class_num = JONG.index(last_sung)
                class_num = [CHO_class_num, JOONG_class_num, JONG_class_num, config_class_num]

        elif is_alphabet(caption):

            if caption in result_dict['{}alphabet'.format(Is_bg)]:
                result_dict['{}alphabet'.format(Is_bg)][caption] += 1
            else:
                result_dict['{}alphabet'.format(Is_bg)][caption] = 1

            class_num = ALPHABET.index(caption)

        elif is_number(caption):

            if caption in result_dict['{}number'.format(Is_bg)]:
                result_dict['{}number'.format(Is_bg)][caption] += 1
            else:
                result_dict['{}number'.format(Is_bg)][caption] = 1

            class_num = NUMBER.index(caption)

        else:
            if caption in result_dict['{}etc'.format(Is_bg)]:
                result_dict['{}etc'.format(Is_bg)][caption] += 1
            else:
                result_dict['{}etc'.format(Is_bg)][caption] = 1

            if caption in special_symbols:
                class_num = special_symbols.index(caption)
            else :
                class_num=-1


    else:
        if caption in result_dict['{}non_single'.format(Is_bg)]:
            result_dict['{}non_single'.format(Is_bg)][caption] += 1
        else:
            result_dict['{}non_single'.format(Is_bg)][caption] = 1

        class_num = -1

        if caption == '...':
            class_num = len(special_symbols)

    return class_num

def make_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def write_txtnimg(f, root, info,  img):

    image_path = os.path.join('{}/{}_{}_{}.jpg'.format(info[3], info[0], info[1], info[2]))

    cv2.imwrite(os.path.join(root, '{}/{}_{}_{}.jpg').format(info[3],info[0], info[1],info[2]), img)
    f.write(image_path)


if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--anno_root", type=str, default="/home/user/VDO/Dataset/v_caption_patch")
    parser.add_argument("--valid_set", type=list, default=['000120', '000121', '000122', '000140', '000141', '000142', '000300', '000301', '000302'])
    parser.add_argument("--save_dir", type=str, default='hangul_patch')
    parser.add_argument("--task_name", type=str, default='background_result')

    args = parser.parse_args()

    anno_root = args.anno_root

    result_dict = dict()
    Is_bg=""
    result_dict['{}hangul'.format(Is_bg)] = dict()
    result_dict['{}hangul_jamo'.format(Is_bg)] = dict()
    result_dict['{}number'.format(Is_bg)] = dict()
    result_dict['{}alphabet'.format(Is_bg)] = dict()
    result_dict['{}etc'.format(Is_bg)] = dict()
    result_dict['{}non_single'.format(Is_bg)] = dict()

    Is_bg = "bg"
    result_dict['{}hangul'.format(Is_bg)] = dict()
    result_dict['{}hangul_jamo'.format(Is_bg)] = dict()
    result_dict['{}number'.format(Is_bg)] = dict()
    result_dict['{}alphabet'.format(Is_bg)] = dict()
    result_dict['{}etc'.format(Is_bg)] = dict()
    result_dict['{}non_single'.format(Is_bg)] = dict()


    anno_dic={}
    json_files = [pos_json for pos_json in os.listdir(args.anno_root) if pos_json.endswith('.json') and pos_json.startswith('ocr') ]


    for file in json_files:
        with open(os.path.join(anno_root, file)) as f:
            anno_dic[file[:-5]]= json.load(f)

    for key, anno in anno_dic.items():
        print('Start ',key)
        count = 0
        if key.startswith('background'):
            Is_bg="bg"
        else : Is_bg=""



        stat_dict = dict()
        stat_dict['first'] = dict()
        stat_dict['middle'] = dict()
        stat_dict['last'] = dict()

        save_dirh='{}hangul_patch'.format(Is_bg)
        save_dira='{}alphabet_patch'.format(Is_bg)
        save_dirn='{}number_patch'.format(Is_bg)
        save_dirs = '{}symbol_patch'.format(Is_bg)

        make_dir(os.path.join(args.anno_root, save_dirh))
        make_dir(os.path.join(args.anno_root, save_dira))
        make_dir(os.path.join(args.anno_root, save_dirn))
        make_dir(os.path.join(args.anno_root, save_dirs))

        f_h1 = open(anno_root + '/{}hangul_patch_train.txt'.format(Is_bg), 'a')
        f_h2 = open(anno_root + '/{}hangul_patch_val.txt'.format(Is_bg), 'a')
        f_a1 = open(anno_root + '/{}alphabet_patch_train.txt'.format(Is_bg), 'a')
        f_a2 = open(anno_root + '/{}alphabet_patch_val.txt'.format(Is_bg), 'a')
        f_n1 = open(anno_root + '/{}number_patch_train.txt'.format(Is_bg), 'a')
        f_n2 = open(anno_root + '/{}number_patch_val.txt'.format(Is_bg), 'a')
        f_s1 = open(anno_root + '/{}symbol_patch_train.txt'.format(Is_bg), 'a')
        f_s2 = open(anno_root + '/{}symbol_patch_val.txt'.format(Is_bg), 'a')

        for clip in anno['annotation']['clips']:

            clip_name = clip['clip_name']

            divider = random.randint(0, 9)
            # if divider <= 1:
            #     f = [f_h2, f_a2,f_n2, f_s2] # validation set
            # else:
            #     f = [f_h1, f_a1, f_n1, f_s1]  # training set
            if clip_name in args.valid_set:
                f = [f_h2, f_a2, f_n2, f_s2] # validation set
            else:
                f = [f_h1, f_a1, f_n1, f_s1] # training set

            clip_count = 0
            for image in clip['images']:

                image_name = image['filename']
                im = cv2.imread(os.path.join(anno_root, key, clip_name, image_name))
                image_name_edited = image_name.split('.')[0]

                sub_count = 1
                for bbox in image['bbox']:

                    x1 = int(bbox['start_x'])
                    x2 = int(bbox['end_x'])
                    y1 = int(bbox['start_y'])
                    y2 = int(bbox['end_y'])

                    cropped_img = im[y1:y2, x1:x2]

                    # cv2.imshow('photo', cropped_img)
                    # cv2.waitKey(0)

                    # class number assign with caption
                    caption = bbox['caption']
                    if caption is None:
                        continue
                    class_num = class_assign(caption, Is_bg)

                    # Nullify logo
                    if int(clip_name) < 500 and (x2 < 146 and y2 < 120):
                        if clip_count > 14:
                            continue
                        else:
                            clip_count += 1
                    elif int(clip_name) > 500 and int(clip_name) < 1500 and (x2 < 235 and y2 < 97):
                        if clip_count > 11:
                            continue
                        else:
                            clip_count += 1
                    elif int(clip_name) > 1500 and (x2 < 189 and y2 < 101):
                        if clip_count > 20:
                            continue
                        else:
                            clip_count += 1

                    if is_hangul(caption) :

                        image_path = os.path.join(save_dirh,
                                                  '{}_{}_{}.jpg'.format(clip_name, image_name_edited, sub_count))

                        cv2.imwrite(os.path.join(anno_root,save_dirh,
                                                 '{}_{}_{}.jpg'.format(clip_name, image_name_edited, sub_count)),
                                    cropped_img)

                        f[0].write(image_path)

                        # hangul cho joong jong

                        if isinstance(class_num, list):
                            for i, cls in enumerate(class_num):
                                cls = str(int(cls))
                                f[0].write(' ' + cls)

                                if i == 0:
                                    try:
                                        stat_dict['first'][cls] += 1
                                    except:
                                        stat_dict['first'][cls] = 1
                                elif i == 1:
                                    try:
                                        stat_dict['middle'][cls] += 1
                                    except:
                                        stat_dict['middle'][cls] = 1
                                elif i == 2:
                                    try:
                                        stat_dict['last'][cls] += 1
                                    except:
                                        stat_dict['last'][cls] = 1
                        elif class_num == -1:
                            continue
                        else:
                            raise ValueError

                        f[0].write('\n')
                        sub_count += 1

                    elif class_num != -1:
                        if len(caption) ==1 and is_alphabet(caption):

                            write_txtnimg(f[1],anno_root,[clip_name, image_name_edited, sub_count, save_dira], cropped_img)

                            f[1].write(' '+str(class_num)+'\n')

                        elif len(caption)==1 and is_number(caption):
                            write_txtnimg(f[2], anno_root,[clip_name, image_name_edited, sub_count, save_dirn], cropped_img)
                            f[2].write(' '+str(class_num)+'\n')
                        else :
                            try:
                                write_txtnimg(f[3], anno_root, [clip_name, image_name_edited, sub_count, save_dirs],
                                              cropped_img)
                            except:
                                print('debug')
                            f[3].write(' ' + str(class_num) + '\n')

                        sub_count+=1


                count += 1

                print('{} processed'.format(count))

            print('done')


        for file in f:
            file.close()
    print('debug')
