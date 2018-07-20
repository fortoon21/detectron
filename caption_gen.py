#!/usr/bin/python
#-*-conding:utf-8 -*-

import os
import hgtk
import argparse
import json

from random import randint, choice
from PIL import Image, ImageDraw, ImageFont

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


alphabet = ('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
           'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z')
special_symbols = ("'", '!', '"', '#', '$', '%', '&', '(', ')', '*', '+', ',', '-', '.', ':',
                   '<', '>', '?', '@', '[', ']', '^', '_', '~', '/', '|')
numerals = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

euc_kr=(u'가', u'각', u'간', u'갇', u'갈', u'갉', u'갊', u'감', u'갑', u'값', u'갓', u'갔', u'강', u'갖', u'갗', u'같', u'갚', u'갛', u'개', u'객', u'갠', u'갤', u'갬', u'갭', u'갯', u'갰', u'갱', u'갸', u'갹', u'갼', u'걀', u'걋', u'걍', u'걔', u'걘', u'걜', u'거', u'걱', u'건', u'걷', u'걸', u'걺', u'검', u'겁', u'것', u'겄', u'겅', u'겆', u'겉', u'겊', u'겋', u'게', u'겐', u'겔', u'겜', u'겝', u'겟', u'겠', u'겡', u'겨', u'격', u'겪', u'견', u'겯', u'결', u'겸', u'겹', u'겻', u'겼', u'경', u'곁', u'계', u'곈', u'곌', u'곕', u'곗', u'고', u'곡', u'곤', u'곧', u'골', u'곪', u'곬', u'곯', u'곰', u'곱', u'곳', u'공', u'곶', u'과', u'곽', u'관', u'괄', u'괆', u'괌', u'괍', u'괏', u'광', u'괘', u'괜', u'괠', u'괩', u'괬', u'괭', u'괴', u'괵', u'괸', u'괼', u'굄', u'굅', u'굇', u'굉', u'교', u'굔', u'굘', u'굡', u'굣', u'구', u'국', u'군', u'굳', u'굴', u'굵', u'굶', u'굻', u'굼', u'굽', u'굿', u'궁', u'궂', u'궈', u'궉', u'권', u'궐', u'궜', u'궝', u'궤', u'궷', u'귀', u'귁', u'귄', u'귈', u'귐', u'귑', u'귓', u'규', u'균', u'귤', u'그', u'극', u'근', u'귿', u'글', u'긁', u'금', u'급', u'긋', u'긍', u'긔', u'기', u'긱', u'긴', u'긷', u'길', u'긺', u'김', u'깁', u'깃', u'깅', u'깆', u'깊', u'까', u'깍', u'깎', u'깐', u'깔', u'깖', u'깜', u'깝', u'깟', u'깠', u'깡', u'깥', u'깨', u'깩', u'깬', u'깰', u'깸', u'깹', u'깻', u'깼', u'깽', u'꺄', u'꺅', u'꺌', u'꺼', u'꺽', u'꺾', u'껀', u'껄', u'껌', u'껍', u'껏', u'껐', u'껑', u'께', u'껙', u'껜', u'껨', u'껫', u'껭', u'껴', u'껸', u'껼', u'꼇', u'꼈', u'꼍', u'꼐', u'꼬', u'꼭', u'꼰', u'꼲', u'꼴', u'꼼', u'꼽', u'꼿', u'꽁', u'꽂', u'꽃', u'꽈', u'꽉', u'꽐', u'꽜', u'꽝', u'꽤', u'꽥', u'꽹', u'꾀', u'꾄', u'꾈', u'꾐', u'꾑', u'꾕', u'꾜', u'꾸', u'꾹', u'꾼', u'꿀', u'꿇', u'꿈', u'꿉', u'꿋', u'꿍', u'꿎', u'꿔', u'꿜', u'꿨', u'꿩', u'꿰', u'꿱', u'꿴', u'꿸', u'뀀', u'뀁', u'뀄', u'뀌', u'뀐', u'뀔', u'뀜', u'뀝', u'뀨', u'끄', u'끅', u'끈', u'끊', u'끌', u'끎', u'끓', u'끔', u'끕', u'끗', u'끙', u'끝', u'끼', u'끽', u'낀', u'낄', u'낌', u'낍', u'낏', u'낑', u'나', u'낙', u'낚', u'난', u'낟', u'날', u'낡', u'낢', u'남', u'납', u'낫', u'났', u'낭', u'낮', u'낯', u'낱', u'낳', u'내', u'낵', u'낸', u'낼', u'냄', u'냅', u'냇', u'냈', u'냉', u'냐', u'냑', u'냔', u'냘', u'냠', u'냥', u'너', u'넉', u'넋', u'넌', u'널', u'넒', u'넓', u'넘', u'넙', u'넛', u'넜', u'넝', u'넣', u'네', u'넥', u'넨', u'넬', u'넴', u'넵', u'넷', u'넸', u'넹', u'녀', u'녁', u'년', u'녈', u'념', u'녑', u'녔', u'녕', u'녘', u'녜', u'녠', u'노', u'녹', u'논', u'놀', u'놂', u'놈', u'놉', u'놋', u'농', u'높', u'놓', u'놔', u'놘', u'놜', u'놨', u'뇌', u'뇐', u'뇔', u'뇜', u'뇝', u'뇟', u'뇨', u'뇩', u'뇬', u'뇰', u'뇹', u'뇻', u'뇽', u'누', u'눅', u'눈', u'눋', u'눌', u'눔', u'눕', u'눗', u'눙', u'눠', u'눴', u'눼', u'뉘', u'뉜', u'뉠', u'뉨', u'뉩', u'뉴', u'뉵', u'뉼', u'늄', u'늅', u'늉', u'느', u'늑', u'는', u'늘', u'늙', u'늚', u'늠', u'늡', u'늣', u'능', u'늦', u'늪', u'늬', u'늰', u'늴', u'니', u'닉', u'닌', u'닐', u'닒', u'님', u'닙', u'닛', u'닝', u'닢', u'다', u'닥', u'닦', u'단', u'닫', u'달', u'닭', u'닮', u'닯', u'닳', u'담', u'답', u'닷', u'닸', u'당', u'닺', u'닻', u'닿', u'대', u'댁', u'댄', u'댈', u'댐', u'댑', u'댓', u'댔', u'댕', u'댜', u'더', u'덕', u'덖', u'던', u'덛', u'덜', u'덞', u'덟', u'덤', u'덥', u'덧', u'덩', u'덫', u'덮', u'데', u'덱', u'덴', u'델', u'뎀', u'뎁', u'뎃', u'뎄', u'뎅', u'뎌', u'뎐', u'뎔', u'뎠', u'뎡', u'뎨', u'뎬', u'도', u'독', u'돈', u'돋', u'돌', u'돎', u'돐', u'돔', u'돕', u'돗', u'동', u'돛', u'돝', u'돠', u'돤', u'돨', u'돼', u'됐', u'되', u'된', u'될', u'됨', u'됩', u'됫', u'됴', u'두', u'둑', u'둔', u'둘', u'둠', u'둡', u'둣', u'둥', u'둬', u'뒀', u'뒈', u'뒝', u'뒤', u'뒨', u'뒬', u'뒵', u'뒷', u'뒹', u'듀', u'듄', u'듈', u'듐', u'듕', u'드', u'득', u'든', u'듣', u'들', u'듦', u'듬', u'듭', u'듯', u'등', u'듸', u'디', u'딕', u'딘', u'딛', u'딜', u'딤', u'딥', u'딧', u'딨', u'딩', u'딪', u'따', u'딱', u'딴', u'딸', u'땀', u'땁', u'땃', u'땄', u'땅', u'땋', u'때', u'땍', u'땐', u'땔', u'땜', u'땝', u'땟', u'땠', u'땡', u'떠', u'떡', u'떤', u'떨', u'떪', u'떫', u'떰', u'떱', u'떳', u'떴', u'떵', u'떻', u'떼', u'떽', u'뗀', u'뗄', u'뗌', u'뗍', u'뗏', u'뗐', u'뗑', u'뗘', u'뗬', u'또', u'똑', u'똔', u'똘', u'똥', u'똬', u'똴', u'뙈', u'뙤', u'뙨', u'뚜', u'뚝', u'뚠', u'뚤', u'뚫', u'뚬', u'뚱', u'뛔', u'뛰', u'뛴', u'뛸', u'뜀', u'뜁', u'뜅', u'뜨', u'뜩', u'뜬', u'뜯', u'뜰', u'뜸', u'뜹', u'뜻', u'띄', u'띈', u'띌', u'띔', u'띕', u'띠', u'띤', u'띨', u'띰', u'띱', u'띳', u'띵', u'라', u'락', u'란', u'랄', u'람', u'랍', u'랏', u'랐', u'랑', u'랒', u'랖', u'랗', u'래', u'랙', u'랜', u'랠', u'램', u'랩', u'랫', u'랬', u'랭', u'랴', u'략', u'랸', u'럇', u'량', u'러', u'럭', u'런', u'럴', u'럼', u'럽', u'럿', u'렀', u'렁', u'렇', u'레', u'렉', u'렌', u'렐', u'렘', u'렙', u'렛', u'렝', u'려', u'력', u'련', u'렬', u'렴', u'렵', u'렷', u'렸', u'령', u'례', u'롄', u'롑', u'롓', u'로', u'록', u'론', u'롤', u'롬', u'롭', u'롯', u'롱', u'롸', u'롼', u'뢍', u'뢨', u'뢰', u'뢴', u'뢸', u'룀', u'룁', u'룃', u'룅', u'료', u'룐', u'룔', u'룝', u'룟', u'룡', u'루', u'룩', u'룬', u'룰', u'룸', u'룹', u'룻', u'룽', u'뤄', u'뤘', u'뤠', u'뤼', u'뤽', u'륀', u'륄', u'륌', u'륏', u'륑', u'류', u'륙', u'륜', u'률', u'륨', u'륩', u'륫', u'륭', u'르', u'륵', u'른', u'를', u'름', u'릅', u'릇', u'릉', u'릊', u'릍', u'릎', u'리', u'릭', u'린', u'릴', u'림', u'립', u'릿', u'링', u'마', u'막', u'만', u'많', u'맏', u'말', u'맑', u'맒', u'맘', u'맙', u'맛', u'망', u'맞', u'맡', u'맣', u'매', u'맥', u'맨', u'맬', u'맴', u'맵', u'맷', u'맸', u'맹', u'맺', u'먀', u'먁', u'먈', u'먕', u'머', u'먹', u'먼', u'멀', u'멂', u'멈', u'멉', u'멋', u'멍', u'멎', u'멓', u'메', u'멕', u'멘', u'멜', u'멤', u'멥', u'멧', u'멨', u'멩', u'며', u'멱', u'면', u'멸', u'몃', u'몄', u'명', u'몇', u'몌', u'모', u'목', u'몫', u'몬', u'몰', u'몲', u'몸', u'몹', u'못', u'몽', u'뫄', u'뫈', u'뫘', u'뫙', u'뫼', u'묀', u'묄', u'묍', u'묏', u'묑', u'묘', u'묜', u'묠', u'묩', u'묫', u'무', u'묵', u'묶', u'문', u'묻', u'물', u'묽', u'묾', u'뭄', u'뭅', u'뭇', u'뭉', u'뭍', u'뭏', u'뭐', u'뭔', u'뭘', u'뭡', u'뭣', u'뭬', u'뮈', u'뮌', u'뮐', u'뮤', u'뮨', u'뮬', u'뮴', u'뮷', u'므', u'믄', u'믈', u'믐', u'믓', u'미', u'믹', u'민', u'믿', u'밀', u'밂', u'밈', u'밉', u'밋', u'밌', u'밍', u'및', u'밑', u'바', u'박', u'밖', u'밗', u'반', u'받', u'발', u'밝', u'밞', u'밟', u'밤', u'밥', u'밧', u'방', u'밭', u'배', u'백', u'밴', u'밸', u'뱀', u'뱁', u'뱃', u'뱄', u'뱅', u'뱉', u'뱌', u'뱍', u'뱐', u'뱝', u'버', u'벅', u'번', u'벋', u'벌', u'벎', u'범', u'법', u'벗', u'벙', u'벚', u'베', u'벡', u'벤', u'벧', u'벨', u'벰', u'벱', u'벳', u'벴', u'벵', u'벼', u'벽', u'변', u'별', u'볍', u'볏', u'볐', u'병', u'볕', u'볘', u'볜', u'보', u'복', u'볶', u'본', u'볼', u'봄', u'봅', u'봇', u'봉', u'봐', u'봔', u'봤', u'봬', u'뵀', u'뵈', u'뵉', u'뵌', u'뵐', u'뵘', u'뵙', u'뵤', u'뵨', u'부', u'북', u'분', u'붇', u'불', u'붉', u'붊', u'붐', u'붑', u'붓', u'붕', u'붙', u'붚', u'붜', u'붤', u'붰', u'붸', u'뷔', u'뷕', u'뷘', u'뷜', u'뷩', u'뷰', u'뷴', u'뷸', u'븀', u'븃', u'븅', u'브', u'븍', u'븐', u'블', u'븜', u'븝', u'븟', u'비', u'빅', u'빈', u'빌', u'빎', u'빔', u'빕', u'빗', u'빙', u'빚', u'빛', u'빠', u'빡', u'빤', u'빨', u'빪', u'빰', u'빱', u'빳', u'빴', u'빵', u'빻', u'빼', u'빽', u'뺀', u'뺄', u'뺌', u'뺍', u'뺏', u'뺐', u'뺑', u'뺘', u'뺙', u'뺨', u'뻐', u'뻑', u'뻔', u'뻗', u'뻘', u'뻠', u'뻣', u'뻤', u'뻥', u'뻬', u'뼁', u'뼈', u'뼉', u'뼘', u'뼙', u'뼛', u'뼜', u'뼝', u'뽀', u'뽁', u'뽄', u'뽈', u'뽐', u'뽑', u'뽕', u'뾔', u'뾰', u'뿅', u'뿌', u'뿍', u'뿐', u'뿔', u'뿜', u'뿟', u'뿡', u'쀼', u'쁑', u'쁘', u'쁜', u'쁠', u'쁨', u'쁩', u'삐', u'삑', u'삔', u'삘', u'삠', u'삡', u'삣', u'삥', u'사', u'삭', u'삯', u'산', u'삳', u'살', u'삵', u'삶', u'삼', u'삽', u'삿', u'샀', u'상', u'샅', u'새', u'색', u'샌', u'샐', u'샘', u'샙', u'샛', u'샜', u'생', u'샤', u'샥', u'샨', u'샬', u'샴', u'샵', u'샷', u'샹', u'섀', u'섄', u'섈', u'섐', u'섕', u'서', u'석', u'섞', u'섟', u'선', u'섣', u'설', u'섦', u'섧', u'섬', u'섭', u'섯', u'섰', u'성', u'섶', u'세', u'섹', u'센', u'셀', u'셈', u'셉', u'셋', u'셌', u'셍', u'셔', u'셕', u'션', u'셜', u'셤', u'셥', u'셧', u'셨', u'셩', u'셰', u'셴', u'셸', u'솅', u'소', u'속', u'솎', u'손', u'솔', u'솖', u'솜', u'솝', u'솟', u'송', u'솥', u'솨', u'솩', u'솬', u'솰', u'솽', u'쇄', u'쇈', u'쇌', u'쇔', u'쇗', u'쇘', u'쇠', u'쇤', u'쇨', u'쇰', u'쇱', u'쇳', u'쇼', u'쇽', u'숀', u'숄', u'숌', u'숍', u'숏', u'숑', u'수', u'숙', u'순', u'숟', u'술', u'숨', u'숩', u'숫', u'숭', u'숯', u'숱', u'숲', u'숴', u'쉈', u'쉐', u'쉑', u'쉔', u'쉘', u'쉠', u'쉥', u'쉬', u'쉭', u'쉰', u'쉴', u'쉼', u'쉽', u'쉿', u'슁', u'슈', u'슉', u'슐', u'슘', u'슛', u'슝', u'스', u'슥', u'슨', u'슬', u'슭', u'슴', u'습', u'슷', u'승', u'시', u'식', u'신', u'싣', u'실', u'싫', u'심', u'십', u'싯', u'싱', u'싶', u'싸', u'싹', u'싻', u'싼', u'쌀', u'쌈', u'쌉', u'쌌', u'쌍', u'쌓', u'쌔', u'쌕', u'쌘', u'쌜', u'쌤', u'쌥', u'쌨', u'쌩', u'썅', u'써', u'썩', u'썬', u'썰', u'썲', u'썸', u'썹', u'썼', u'썽', u'쎄', u'쎈', u'쎌', u'쏀', u'쏘', u'쏙', u'쏜', u'쏟', u'쏠', u'쏢', u'쏨', u'쏩', u'쏭', u'쏴', u'쏵', u'쏸', u'쐈', u'쐐', u'쐤', u'쐬', u'쐰', u'쐴', u'쐼', u'쐽', u'쑈', u'쑤', u'쑥', u'쑨', u'쑬', u'쑴', u'쑵', u'쑹', u'쒀', u'쒔', u'쒜', u'쒸', u'쒼', u'쓩', u'쓰', u'쓱', u'쓴', u'쓸', u'쓺', u'쓿', u'씀', u'씁', u'씌', u'씐', u'씔', u'씜', u'씨', u'씩', u'씬', u'씰', u'씸', u'씹', u'씻', u'씽', u'아', u'악', u'안', u'앉', u'않', u'알', u'앍', u'앎', u'앓', u'암', u'압', u'앗', u'았', u'앙', u'앝', u'앞', u'애', u'액', u'앤', u'앨', u'앰', u'앱', u'앳', u'앴', u'앵', u'야', u'약', u'얀', u'얄', u'얇', u'얌', u'얍', u'얏', u'양', u'얕', u'얗', u'얘', u'얜', u'얠', u'얩', u'어', u'억', u'언', u'얹', u'얻', u'얼', u'얽', u'얾', u'엄', u'업', u'없', u'엇', u'었', u'엉', u'엊', u'엌', u'엎', u'에', u'엑', u'엔', u'엘', u'엠', u'엡', u'엣', u'엥', u'여', u'역', u'엮', u'연', u'열', u'엶', u'엷', u'염', u'엽', u'엾', u'엿', u'였', u'영', u'옅', u'옆', u'옇', u'예', u'옌', u'옐', u'옘', u'옙', u'옛', u'옜', u'오', u'옥', u'온', u'올', u'옭', u'옮', u'옰', u'옳', u'옴', u'옵', u'옷', u'옹', u'옻', u'와', u'왁', u'완', u'왈', u'왐', u'왑', u'왓', u'왔', u'왕', u'왜', u'왝', u'왠', u'왬', u'왯', u'왱', u'외', u'왹', u'왼', u'욀', u'욈', u'욉', u'욋', u'욍', u'요', u'욕', u'욘', u'욜', u'욤', u'욥', u'욧', u'용', u'우', u'욱', u'운', u'울', u'욹', u'욺', u'움', u'웁', u'웃', u'웅', u'워', u'웍', u'원', u'월', u'웜', u'웝', u'웠', u'웡', u'웨', u'웩', u'웬', u'웰', u'웸', u'웹', u'웽', u'위', u'윅', u'윈', u'윌', u'윔', u'윕', u'윗', u'윙', u'유', u'육', u'윤', u'율', u'윰', u'윱', u'윳', u'융', u'윷', u'으', u'윽', u'은', u'을', u'읊', u'음', u'읍', u'읏', u'응', u'읒', u'읓', u'읔', u'읕', u'읖', u'읗', u'의', u'읜', u'읠', u'읨', u'읫', u'이', u'익', u'인', u'일', u'읽', u'읾', u'잃', u'임', u'입', u'잇', u'있', u'잉', u'잊', u'잎', u'자', u'작', u'잔', u'잖', u'잗', u'잘', u'잚', u'잠', u'잡', u'잣', u'잤', u'장', u'잦', u'재', u'잭', u'잰', u'잴', u'잼', u'잽', u'잿', u'쟀', u'쟁', u'쟈', u'쟉', u'쟌', u'쟎', u'쟐', u'쟘', u'쟝', u'쟤', u'쟨', u'쟬', u'저', u'적', u'전', u'절', u'젊', u'점', u'접', u'젓', u'정', u'젖', u'제', u'젝', u'젠', u'젤', u'젬', u'젭', u'젯', u'젱', u'져', u'젼', u'졀', u'졈', u'졉', u'졌', u'졍', u'졔', u'조', u'족', u'존', u'졸', u'졺', u'좀', u'좁', u'좃', u'종', u'좆', u'좇', u'좋', u'좌', u'좍', u'좔', u'좝', u'좟', u'좡', u'좨', u'좼', u'좽', u'죄', u'죈', u'죌', u'죔', u'죕', u'죗', u'죙', u'죠', u'죡', u'죤', u'죵', u'주', u'죽', u'준', u'줄', u'줅', u'줆', u'줌', u'줍', u'줏', u'중', u'줘', u'줬', u'줴', u'쥐', u'쥑', u'쥔', u'쥘', u'쥠', u'쥡', u'쥣', u'쥬', u'쥰', u'쥴', u'쥼', u'즈', u'즉', u'즌', u'즐', u'즘', u'즙', u'즛', u'증', u'지', u'직', u'진', u'짇', u'질', u'짊', u'짐', u'집', u'짓', u'징', u'짖', u'짙', u'짚', u'짜', u'짝', u'짠', u'짢', u'짤', u'짧', u'짬', u'짭', u'짯', u'짰', u'짱', u'째', u'짹', u'짼', u'쨀', u'쨈', u'쨉', u'쨋', u'쨌', u'쨍', u'쨔', u'쨘', u'쨩', u'쩌', u'쩍', u'쩐', u'쩔', u'쩜', u'쩝', u'쩟', u'쩠', u'쩡', u'쩨', u'쩽', u'쪄', u'쪘', u'쪼', u'쪽', u'쫀', u'쫄', u'쫌', u'쫍', u'쫏', u'쫑', u'쫓', u'쫘', u'쫙', u'쫠', u'쫬', u'쫴', u'쬈', u'쬐', u'쬔', u'쬘', u'쬠', u'쬡', u'쭁', u'쭈', u'쭉', u'쭌', u'쭐', u'쭘', u'쭙', u'쭝', u'쭤', u'쭸', u'쭹', u'쮜', u'쮸', u'쯔', u'쯤', u'쯧', u'쯩', u'찌', u'찍', u'찐', u'찔', u'찜', u'찝', u'찡', u'찢', u'찧', u'차', u'착', u'찬', u'찮', u'찰', u'참', u'찹', u'찻', u'찼', u'창', u'찾', u'채', u'책', u'챈', u'챌', u'챔', u'챕', u'챗', u'챘', u'챙', u'챠', u'챤', u'챦', u'챨', u'챰', u'챵', u'처', u'척', u'천', u'철', u'첨', u'첩', u'첫', u'첬', u'청', u'체', u'첵', u'첸', u'첼', u'쳄', u'쳅', u'쳇', u'쳉', u'쳐', u'쳔', u'쳤', u'쳬', u'쳰', u'촁', u'초', u'촉', u'촌', u'촐', u'촘', u'촙', u'촛', u'총', u'촤', u'촨', u'촬', u'촹', u'최', u'쵠', u'쵤', u'쵬', u'쵭', u'쵯', u'쵱', u'쵸', u'춈', u'추', u'축', u'춘', u'출', u'춤', u'춥', u'춧', u'충', u'춰', u'췄', u'췌', u'췐', u'취', u'췬', u'췰', u'췸', u'췹', u'췻', u'췽', u'츄', u'츈', u'츌', u'츔', u'츙', u'츠', u'측', u'츤', u'츨', u'츰', u'츱', u'츳', u'층', u'치', u'칙', u'친', u'칟', u'칠', u'칡', u'침', u'칩', u'칫', u'칭', u'카', u'칵', u'칸', u'칼', u'캄', u'캅', u'캇', u'캉', u'캐', u'캑', u'캔', u'캘', u'캠', u'캡', u'캣', u'캤', u'캥', u'캬', u'캭', u'컁', u'커', u'컥', u'컨', u'컫', u'컬', u'컴', u'컵', u'컷', u'컸', u'컹', u'케', u'켁', u'켄', u'켈', u'켐', u'켑', u'켓', u'켕', u'켜', u'켠', u'켤', u'켬', u'켭', u'켯', u'켰', u'켱', u'켸', u'코', u'콕', u'콘', u'콜', u'콤', u'콥', u'콧', u'콩', u'콰', u'콱', u'콴', u'콸', u'쾀', u'쾅', u'쾌', u'쾡', u'쾨', u'쾰', u'쿄', u'쿠', u'쿡', u'쿤', u'쿨', u'쿰', u'쿱', u'쿳', u'쿵', u'쿼', u'퀀', u'퀄', u'퀑', u'퀘', u'퀭', u'퀴', u'퀵', u'퀸', u'퀼', u'큄', u'큅', u'큇', u'큉', u'큐', u'큔', u'큘', u'큠', u'크', u'큭', u'큰', u'클', u'큼', u'큽', u'킁', u'키', u'킥', u'킨', u'킬', u'킴', u'킵', u'킷', u'킹', u'타', u'탁', u'탄', u'탈', u'탉', u'탐', u'탑', u'탓', u'탔', u'탕', u'태', u'택', u'탠', u'탤', u'탬', u'탭', u'탯', u'탰', u'탱', u'탸', u'턍', u'터', u'턱', u'턴', u'털', u'턺', u'텀', u'텁', u'텃', u'텄', u'텅', u'테', u'텍', u'텐', u'텔', u'템', u'텝', u'텟', u'텡', u'텨', u'텬', u'텼', u'톄', u'톈', u'토', u'톡', u'톤', u'톨', u'톰', u'톱', u'톳', u'통', u'톺',  u'힉', u'힌', u'힐', u'힘', u'힙', u'힛', u'힝')


def make_dir(dir_path):
    if not os.path.exists(dir_path)  :
        os.makedirs(dir_path)

if __name__=='__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--background_img_path",type=str, default='/home/jade/ws/vdotdo/background')
    parser.add_argument("--image_save_path",    type=str, default='/home/jade/ws/vdotdo/background_result')
    parser.add_argument("--latin_font_path",    type=str, default='/home/jade/ws/vdotdo/fonts_latin')
    parser.add_argument("--korean_font_path",   type=str, default='/home/jade/ws/vdotdo/fonts_korean')
    parser.add_argument("--generate_image_num", type=int, default=20000)
    parser.add_argument("--image_width",        type=int, default=1280)
    parser.add_argument("--image_height",       type=int, default=720)
    parser.add_argument("--euc_font_list",      type=list,default=['Jua-Regular.ttf', 'Gaegu-Regular.ttf', 'EastSeaDokdo-Regular.ttf', 'YeonSung-Regular.ttf', 'Sunflower-Bold.ttf', 'Sunflower-Light.ttf',
                                                                   'Dokdo-Regular.ttf', 'BlackHanSans-Regular.ttf', 'Gaegu-Bold.ttf', 'CuteFont-Regular.ttf', 'DoHyeon-Regular.ttf', 'SongMyung-Regular.ttf',
                                                                   'Sunflower-Medium.ttf', 'Gaegu-Light.ttf', 'KirangHaerang-Regular.ttf'])

    args = parser.parse_args()
    sentence_type = ((40, 440), (440, 840), (840, 1240))

    top_offset = 15
    bottom_offset = 15
    left_offset = 40
    right_offset = 40
    max_font_size = 10
    img_width = args.image_width
    img_height = args.image_height

    dir_list = os.listdir(args.background_img_path)
    dir_list.sort()

    font_list = os.listdir(args.korean_font_path)
    korean_font_list = os.listdir(args.korean_font_path)

    anno = dict()
    anno = {'annotation': {'clips': []}}

    count = 1
    for clip_num, dir_path in enumerate(dir_list):

        make_dir(os.path.join(args.image_save_path, dir_path))

        image_list = os.listdir(os.path.join(args.background_img_path, dir_path))
        anno['annotation']['clips'].append({'images': [], 'clip_name': dir_path})

        for image_path in image_list:
            # while count <= args.generate_image_num:
            img_complete_Flag = True
            while img_complete_Flag:
                image_info_dict = {'bbox': []}

                # create Image object with the input image
                image = Image.open(os.path.join(args.background_img_path, dir_path, image_path))
                # image = Image.new('RGB', (img_width, img_height), (255, 255, 255))
                draw = ImageDraw.Draw(image)
                draw.rectangle(((1000, 0), (1280, 200)), fill=(128, 128, 128))

                start_y = top_offset + choice([0, 230, 460])
                text_count = 0

                while True:

                    start_x, end_x = sentence_type[randint(0, len(sentence_type) - 1)]

                    font_size_index = randint(0, 10)
                    if font_size_index <= 3:
                        font_size = randint(30, 50)
                    elif font_size_index > 3 and font_size_index <= 6:
                        font_size = randint(50, 80)
                    elif font_size_index > 6 and font_size_index <= 8:
                        font_size = randint(80, 110)
                    elif font_size_index > 8:
                        font_size = randint(110, 150)
                    else:
                        raise NotImplementedError

                    ko_en_index = randint(0, 9)
                    if ko_en_index <= 7:
                        selected_font_list = korean_font_list
                        selected_font = 'Kor'
                    else:
                        selected_font_list = font_list
                        selected_font = 'Latin'

                    font_index = randint(0, len(selected_font_list)-1)
                    font = ImageFont.truetype(selected_font_list[font_index], size=font_size)
                    max_font_size = font.font.ascent + abs(font.font.descent)

                    sparsity_index = randint(0, 9)
                    if sparsity_index <= 3:
                        reverse_index = randint(0, 9)
                        if reverse_index <= 1:
                            reverse_Flag = True
                        else:
                            reverse_Flag = False

                        if reverse_Flag:
                            font_color = 'rgb({}, {}, {})'.format(255, 255, 255)
                            while True:
                                r_value = randint(0, 255)
                                g_value = randint(0, 255)
                                b_value = randint(0, 255)
                                if (r_value + g_value + b_value) < 600:
                                    break
                            background_color = 'rgb({}, {}, {})'.format(r_value, g_value, b_value)
                            draw.rectangle(((max(start_x - font.getsize(' ')[0], 0), start_y - max_font_size*0.05), (1280, start_y + max_font_size*1.05)), fill=background_color)
                        else:
                            while True:
                                r_value = randint(0, 255)
                                g_value = randint(0, 255)
                                b_value = randint(0, 255)
                                if (r_value + g_value + b_value) < 600:
                                    break
                            font_color = 'rgb({}, {}, {})'.format(r_value, g_value, b_value)

                        while True:
                            letter_dict = {}

                            if selected_font == 'Kor':
                                text_index = randint(0, 24)
                                if text_index <= 11:

                                    if selected_font_list[font_index] in args.euc_font_list:
                                        text = euc_kr[randint(0, len(euc_kr) - 1)]
                                    else:
                                        letter_CHO = CHO[randint(0, len(CHO) - 1)]
                                        letter_JOONG = JOONG[randint(0, len(JOONG) - 1)]
                                        letter_JONG = JONG[randint(0, len(JONG) - 1)]
                                        text = hgtk.letter.compose(letter_CHO, letter_JOONG, letter_JONG)
                                elif text_index > 11 and text_index <= 17:
                                    text = alphabet[randint(0, len(alphabet)-1)]
                                    if randint(0, 1):
                                        text = text.upper()
                                elif text_index > 17 and text_index <= 21:
                                    text = numerals[randint(0, len(numerals)-1)]
                                elif text_index > 21 and text_index <= 23:
                                    text = special_symbols[randint(0, len(special_symbols)-1)]
                                elif text_index > 23 and text_index <= 24:
                                    text = ' '
                                else:
                                    raise NotImplementedError

                            elif selected_font == 'Latin':
                                text_index = randint(0, 12)
                                if text_index <= 5:
                                    text = alphabet[randint(0, len(alphabet)-1)]
                                    if randint(0, 1):
                                        text = text.upper()
                                elif text_index > 5 and text_index <= 9:
                                    text = numerals[randint(0, len(numerals) - 1)]
                                elif text_index > 9 and text_index <= 11:
                                    text = special_symbols[randint(0, len(special_symbols) - 1)]
                                elif text_index > 11 and text_index <= 12:
                                    text = ' '
                                else:
                                    raise NotImplementedError

                            mask, offset = font.getmask2(text)
                            text_size = font.getsize(text)

                            if (start_x + text_size[0]) >= end_x or (start_x + text_size[0]) >= (img_width - right_offset):
                                break
                            elif (start_y + max_font_size) >= (img_height - bottom_offset):
                                break

                            y_computed = start_y + max_font_size - text_size[1]
                            draw.text((start_x, y_computed), text, fill=font_color, font=font)
                            # if text == '-' or text == '*' or text == '^' or text == "'" or text == '~' or text == '"':
                            #     draw.rectangle(((start_x + offset[0], y_computed + offset[1]), (start_x + text_size[0], y_computed + text_size[1] - font.font.descent)), outline=(0,255,0))
                            # else:
                            #     draw.rectangle(((start_x + offset[0], y_computed + offset[1]), (start_x + text_size[0], y_computed + text_size[1])), outline=(0,255,0))

                            letter_dict['caption'] = text
                            letter_dict['start_x'] = start_x + offset[0]
                            letter_dict['start_y'] = y_computed + offset[1]
                            letter_dict['end_x'] = start_x + text_size[0]
                            letter_dict['end_y'] = y_computed + text_size[1]

                            if text != ' ':
                                image_info_dict['bbox'].append(letter_dict)
                                text_count += 1

                            if text_count > 30:
                                break

                            start_x += text_size[0] + int(text_size[0] * 0.05)

                    start_y += max_font_size + (max_font_size * 0.2)

                    if start_y >= (img_height - bottom_offset):
                        break
                    if text_count > 30:
                        break

                    # image.show()
                    # print('debug')
                if not image_info_dict['bbox']:
                    continue
                image_name = os.path.join(args.image_save_path, dir_path, image_path)
                image.save(image_name, optimize=True, quality=95)
                image_info_dict['filename'] = image_path
                anno['annotation']['clips'][clip_num]['images'].append(image_info_dict)
                # print('{}/{} Processed'.format(count, args.generate_image_num))
                print('{} Processed'.format(count))
                count += 1
                img_complete_Flag = False

    with open('background_result.json', 'w') as outfile:
        json.dump(anno, outfile)
    print('Finish!')