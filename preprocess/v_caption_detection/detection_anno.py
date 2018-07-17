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

JAMO = CHO + JOONG + JONG[1:]
FIRST_HANGUL_UNICODE = 0xAC00  # '가'
LAST_HANGUL_UNICODE = 0xD7A3  # '힣'

FIRST_NUMBER_UNICODE = 0x0030  # '0'
LAST_NUMBER_UNICODE = 0x0039  # '9'

FIRST_ALPHABET_UPPER_UNICODE = 0x0041  # 'A'
LAST_ALPHABET_UPPER_UNICODE = 0x005A  # 'Z'

FIRST_ALPHABET_LOWER_UNICODE = 0x0061  # 'a'
LAST_ALPHABET_LOWER_UNICODE = 0x007A  # 'z'


def is_jamo(letter):
    return letter in JAMO

def is_hangul(phrase): # TODO: need tuning!!
    for letter in phrase:
        code = ord(letter)
        if (code < FIRST_HANGUL_UNICODE or code > LAST_HANGUL_UNICODE) and not is_jamo(letter):
            return False

    return True

def is_alphabet(phrase): # TODO: need tuning!!
    unicode_value = ord(phrase)
    if (unicode_value >= FIRST_ALPHABET_UPPER_UNICODE and unicode_value <= LAST_ALPHABET_UPPER_UNICODE) \
       or (unicode_value >= FIRST_ALPHABET_LOWER_UNICODE and unicode_value <= LAST_ALPHABET_LOWER_UNICODE):
        return True
    else:
        return False

def is_number(phrase): # TODO: need tuning!!
    for letter in phrase:
        code = ord(letter)
        if (code < FIRST_NUMBER_UNICODE or code > LAST_NUMBER_UNICODE):
            return False
    return True

def class_assign(caption):
    if len(caption) == 1:
        if is_hangul(caption):

            if is_jamo(caption):

                if caption in result_dict['hangul_jamo']:
                    result_dict['hangul_jamo'][caption] += 1
                else:
                    result_dict['hangul_jamo'][caption] = 1

                class_num = 6

            else:

                if caption in result_dict['hangul']:
                    result_dict['hangul'][caption] += 1
                else:
                    result_dict['hangul'][caption] = 1

                eumso_list = hgtk.letter.decompose(caption)
                middle_sung = eumso_list[1]
                last_sung = eumso_list[2]

                if last_sung == JONG[0]:
                    if middle_sung in JOONG[:9]:
                        class_num = 0
                    elif middle_sung in JOONG[9:14]:
                        class_num = 1
                    elif middle_sung in JOONG[14:]:
                        class_num = 2
                elif last_sung in JONG[1:]:
                    if middle_sung in JOONG[0:9]:
                        class_num = 3
                    elif middle_sung in JOONG[9:14]:
                        class_num = 4
                    elif middle_sung in JOONG[14:]:
                        class_num = 5
                else:
                    raise ValueError

        elif is_alphabet(caption):

            if caption in result_dict['alphabet']:
                result_dict['alphabet'][caption] += 1
            else:
                result_dict['alphabet'][caption] = 1

            class_num = 7

        elif is_number(caption):

            if caption in result_dict['number']:
                result_dict['number'][caption] += 1
            else:
                result_dict['number'][caption] = 1

            class_num = 8

        else:
            if caption in result_dict['etc']:
                result_dict['etc'][caption] += 1
            else:
                result_dict['etc'][caption] = 1

            class_num = 9

    else:
        if caption in result_dict['non_single']:
            result_dict['non_single'][caption] += 1
        else:
            result_dict['non_single'][caption] = 1

        if caption == '...':
            class_num = 9
        else:
            class_num = 10000

    return class_num


if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--anno_root", type=str, default="/home/vdo/SSD2TB/V_Caption/dataset")
    parser.add_argument("--valid_set", type=list, default=['000481', '000482', '001293', '001294', '001771', '001772'])
    # parser.add_argument("--json_files",type=list, default=['ocr_demo2.json', 'ocr_demo3.json', 'ocr_demo5.json', 'ocr_demo6.json', 'ocr_demo7.json'])
    parser.add_argument("--json_files",type=list, default=['background_result.json'])

    args = parser.parse_args()

    anno_root = args.anno_root

    anno_dic={}
    for file in args.json_files:
        with open(os.path.join(anno_root, file)) as f:
            anno_dic[file[:-5]]= json.load(f)

    f_1 = open('caption_BG_train.txt', 'w')
    f_2 = open('caption_BG_val.txt', 'w')

    result_dict = dict()
    result_dict['hangul'] = dict()
    result_dict['hangul_jamo'] = dict()
    result_dict['number'] = dict()
    result_dict['alphabet'] = dict()
    result_dict['etc'] = dict()
    result_dict['non_single'] = dict()
    result_dict['error'] = dict()

    count = 0
    for key, anno in anno_dic.items():

        for clip in anno['annotation']['clips']:

            clip_name = clip['clip_name']

            # if clip_name in args.valid_set:
            #     f = f_2  # validation set
            # else:
            #     f = f_1  # training set
            divider = random.randint(0, 9)
            if divider <= 0:
                f = f_1  # training set
            else:
                f = f_2  # validation set


            for image in clip['images']:

                image_name = image['filename']
                im = cv2.imread(os.path.join(anno_root, key, clip_name, image_name))
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
                    if class_num == 10000:
                        continue

                    x1 = bbox['start_x']
                    y1 = bbox['start_y']
                    x2 = bbox['end_x']
                    y2 = bbox['end_y']

                    # if class_num == 10:
                    #     cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    #     cv2.imshow('photo', im)
                    #     cv2.waitKey(0)
                    #     cv2.destroyAllWindows()

                    w = float(x2) - float(x1)
                    h = float(y2) - float(y1)
                    if w <= 0 or h <= 0:
                        if caption in result_dict['error']:
                            result_dict['error'][caption] += 1
                        else:
                            result_dict['error'][caption] = 1

                        # cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        # cv2.imshow('photo', im)
                        # cv2.waitKey(0)
                        # cv2.destroyAllWindows()
                        # raise ValueError

                    bbox_xyxy = [x1, y1, x2, y2]
                    for coord in bbox_xyxy:
                        coord = str(int(coord))

                        f.write(' ' + coord)

                    f.write(' {}'.format(class_num))
                    print('Caption : {} | class_num : {}'.format(caption, class_num))

                f.write('\n')
                count += 1

                print('{} processed'.format(count))

    f_1.close()
    f_2.close()
    print('done')