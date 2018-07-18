import os
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


if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--anno_root", type=str, default="/home/jade/ws/vdotdo")
    parser.add_argument("--valid_set", type=list, default=['000481', '000482', '001293', '001294', '001771', '001772'])
    parser.add_argument("--task_name", type=str, default='ocr_demo2')


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
    json_files = [pos_json for pos_json in os.listdir(args.anno_root) if pos_json.startswith('back') and pos_json.endswith('.json') ]


    for file in json_files:
        with open(os.path.join(anno_root, file)) as f:
            anno_dic[file[:-5]]= json.load(f)

    for key, anno in anno_dic.items():
        print('Start ',key)
        count = 0
        if key.startswith('background'):
            Is_bg="bg"
        else : Is_bg=""

        f_1 = open(anno_root + '/{}Detection_train.txt'.format(Is_bg), 'a')
        f_2 = open(anno_root + '/{}Detection_val.txt'.format(Is_bg), 'a')

        for clip in anno['annotation']['clips']:

            clip_name = clip['clip_name']

            divider = random.randint(0, 9)
            if divider <= 1:
                f=f_1
            else :
                f=f_2

            clip_count=0
            for image in clip['images']:

                image_name = image['filename']
                im = cv2.imread(os.path.join(anno_root, args.task_name, clip_name, image_name))
                image_path = os.path.join(clip_name, image_name)
                f.write(image_path)

                for bbox in image['bbox']:

                # class number assign with caption
                    caption = bbox['caption']
                    try:
                        len(caption)
                    except:
                        print('debug')
                    class_num = class_assign(caption)
                    if class_num == -1:
                        continue

                    x1 = bbox['start_x']
                    y1 = bbox['start_y']
                    x2 = bbox['end_x']
                    y2 = bbox['end_y']

                    bbox_xyxy = [x1, y1, x2, y2]
                    for coord in bbox_xyxy:
                        coord = str(int(coord))

                        f.write(' ' + coord)

                    f.write(' {}'.format(class_num))

                print('Caption : {} | class_num : {}'.format(caption, class_num))

            f.write('\n')
            count += 1

            print('{} processed'.format(count))

        f.close()
print('done')