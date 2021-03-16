# -*- coding: utf-8 -*-
import re
import sys
import codecs
import pypinyin
from pypinyin import lazy_pinyin
from pypinyin import pinyin, TONE3, Style
from .langconv import Converter
SPLIT_REGX = re.compile('([\r\n﹔﹖﹗。！？?!;；]+["’”]{0,2}|\u2026\u2026["’”“‘]{0,1})')
#from pyltp import SentenceSplitter
PUNCTUATION_LIST = ".?!:。，,、？：；{}[]()【】“‘’”《》/！%……（）<>@#$~^￥%&*\"\'=+-;　"
TARGET_REGX = re.compile("[^\u4e00-\u9fa5,]")

def wordnum(text):
    return len(text)

def strip_last_punc(ustring):
    len_list = range(0, len(ustring))
    end = len(ustring)
    for idx in len_list[::-1]:
        if ustring[idx] in PUNCTUATION_LIST:
            end = idx 
        else:
            break
    ustring = ustring[0 : end]
    return ustring

def is_str_same_except_punc(ustr1, ustr2):
    if ustr1 == ustr2:
        return True
    len1, len2, i, j = len(ustr1), len(ustr2), 0, 0
    while True:
        while i < len1 and ustr1[i] in PUNCTUATION_LIST:
            i += 1
        while j < len2 and ustr2[j] in PUNCTUATION_LIST:
            j += 1
        if i >= len1 or j >= len2:
            break
        if ustr1[i] == ustr2[j]:
            i += 1
            j += 1
        else:
            break

    while i < len1 and ustr1[i] in PUNCTUATION_LIST:
        i += 1
    while j < len2 and ustr2[j] in PUNCTUATION_LIST:
        j += 1

    return i == len1 and j == len2

def arrWordnum(wordArr):
    count = 0
    for word in wordArr:
        count += len(word)
    return count

def byteify(input, encoding='utf-8'):
    if isinstance(input, dict):
        return {byteify(key): byteify(value) for key, value in input.iteritems()}
    elif isinstance(input, list):
        return [byteify(element) for element in input]
    elif isinstance(input, str):
        return input.encode(encoding)
    else:
        return input

def sentence_is_valid(sent):
    #digit, ascii, or chinese
    if sent.strip() == "":
        return False
    cnt = 0
    for ch in sent:
        if not is_chinese(ch):
            cnt += 1
    if cnt / len(sent) > 0.7:
        return False
    return True

def replaceChinesePunctuation(text):
    text = text.replace("¨", "\"")
    text = text.replace("─", "-")
    text = text.replace("╔", "\"")
    text = text.replace("╝", "\"")
    text = text.replace("゛", "\"")
    text = text.replace("`", "'")
    text = text.replace("～", "~")
    text = text.replace("！", "!")
    text = text.replace("，", ",")
    text = text.replace("；", ";")
    text = text.replace("？", "?")
    text = text.replace("﹖", "?")
    text = text.replace("：", ":")
    text = text.replace("（", "(")
    text = text.replace("）", ")")
    text = text.replace("【", "(")
    text = text.replace("〔", "(")
    text = text.replace("】", ")")
    text = text.replace("〕", ")")
    text = text.replace("「", "\"")
    text = text.replace("」", "\"")
    text = text.replace("‘", "'")
    text = text.replace("’", "'")
    text = text.replace("『", "《") 
    text = text.replace("』", "》")
    text = text.replace("\t", "。")
    text = text.replace("“", "\"")
    text = text.replace("”", "\"")
    text = text.replace("\n", "。")
    text = text.replace("　", " ")
    text = text.replace("\t", "。")
    text = text.replace("\n", "。")
    #text = text.replace("％", "%")
    text = text.replace("", "")
    text = text.replace("\xe2\x80\x8b", " ")
    text = text.replace("\xe2\x80\x8c", " ")
    text = text.replace("\xe2\x80\x8d", " ")
    
    while text.find("  ") >= 0:
        text = text.replace("  ", " ")
    while text.find("。。") >= 0:
        text = text.replace("。。", "。")
    while text.find("!!") >= 0:
        text = text.replace("!!", "!")
    while text.find("??") >= 0:
        text = text.replace("??", "?")
    text = text.strip()
    return text

def split_para(text, SEP='      '):
    paras = text.split(SEP)
    return paras

def clean(pstring):
    pstring = pstring.replace("\t"," ").replace("\r"," ").replace("\n"," ").replace(' ', ' ').replace("　", " ")
    return pstring

#def split_sentence(para):
#    return SentenceSplitter.split(para)

def split_para_sentence(text, SEP='      '):
    paras = text.split(SEP)
    sentences = []
    for para in paras:
        sentences.extend(split_sentence(para))
    return sentences

def split_sentence(para, has_pun = True):
    para = para.replace("\t"," ").replace("\r"," ").replace("\n"," ")
    sentences = SPLIT_REGX.split(para)
    sent_list = []  
    for sent in sentences:
        if SPLIT_REGX.match(sent) and len(sent_list) > 0:
            if has_pun is True:
                sent_list[-1] += sent
        elif sent:
            sent_list.append(sent)
    return sent_list

def cutContent(text, maxWord, maxPara):
    splitor = "      "
    text = "".join(text.decode("utf-8")[:maxWord])
    text = splitor.join(text.split(splitor)[:maxPara]).strip()
    return text

def is_chinese(uchar):
    """判断一个unicode是否是汉字"""
    if u'\u4e00' <= uchar <= u'\u9fa5':
        return True
    else:
        return False

def is_chinese_string(ustring):
    """判断是否全为汉字"""
    for c in ustring:
        if not is_chinese(c):
            return False
    return True


def is_number(uchar):
    """判断一个unicode是否是数字"""
    if uchar >= u'u0030' and uchar <= u'u0039':
        return True
    else:
        return False


def is_alphabet(uchar):
    """判断一个unicode是否是英文字母"""
    if (uchar >= u'u0041' and uchar <= u'u005a') or (uchar >= u'u0061' and uchar <= u'u007a'):
        return True
    else:
        return False


def is_other(uchar):
    """判断是否非汉字，数字和英文字符"""
    if not (is_chinese(uchar) or is_number(uchar) or is_alphabet(uchar)):
        return True
    else:
        return False

def B2Q(uchar):
    """半角转全角"""
    inside_code = ord(uchar)
    if inside_code < 0x0020 or inside_code > 0x7e:  # 不是半角字符就返回原来的字符
        return uchar
    if inside_code == 0x0020:  # 除了空格其他的全角半角的公式为:半角=全角-0xfee0
        inside_code = 0x3000
    else:
        inside_code += 0xfee0
    return chr(inside_code)

def Q2B(uchar):
    """全角转半角"""
    inside_code = ord(uchar)
    if inside_code == 0x3000:
        inside_code = 0x0020
    else:
        inside_code -= 0xfee0
    if inside_code < 0x0020 or inside_code > 0x7e:  # 转完之后不是半角字符返回原来的字符
        return uchar
    return chr(inside_code)

def stringQ2B(ustring):
    """把字符串全角转半角"""
    return "".join([Q2B(uchar) for uchar in ustring])

def uniform(ustring):
    """格式化字符串，完成全角转半角，大写转小写的工作"""
    return stringQ2B(ustring).lower()

def uniform_punc(ustring):
    END_PUNC = r'!?.,！？，。；;'
    ustr = re.sub('[%s]+' % re.escape(END_PUNC), ',', ustring)
    ustr = TARGET_REGX.sub('', ustr)
    return ustr

def remove_punctuation(strs):
    """
    去除标点符号
    :param strs:
    :return:
    """
    return re.sub(u"[\s+\.\!\/<>“”,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "", strs.strip())


def get_homophones_by_char(input_char):
    """
    根据汉字取同音字
    :param input_char:
    :return:
    """
    result = []
    # CJK统一汉字区的范围是0x4E00-0x9FA5,也就是我们经常提到的20902个汉字
    for i in range(0x4e00, 0x9fa6):
        if pinyin([unichr(i)], style=pypinyin.NORMAL)[0][0] == pinyin(input_char, style=pypinyin.NORMAL)[0][0]:
            result.append(unichr(i))
    return result


def get_homophones_by_pinyin(input_pinyin):
    """
    根据拼音取同音字
    :param input_pinyin:
    :return:
    """
    result = []
    # CJK统一汉字区的范围是0x4E00-0x9FA5,也就是我们经常提到的20902个汉字
    for i in range(0x4e00, 0x9fa6):
        if pinyin([unichr(i)], style=pypinyin.TONE2)[0][0] == input_pinyin:
            # TONE2: 中zho1ng
            result.append(unichr(i))
    return result

def countProcessMemoey(processName):
    cmd = "ps -aux | grep %s | sort -k3,3 -nr | head -1" % processName
    res = os.popen(cmd)
    res = res.readlines()
    if len(res) >= 1:
        arr = res[0].split(" ")
        arr = [i.strip() for i in arr if i.strip() != ""]
        if len(arr) >= 4:
           return float(arr[3])
    return 0.0

     
def writelog(log):
    sys.stderr.write(str(log).strip() + "\n")
    sys.stderr.flush()

def traditional2simplified(sentence):
    """                                                                                                            
    将sentence中的繁体字转为简体字
    :param sentence: 待转换的句子
    :return: 将句子中繁体字转换为简体字之后的句子
    """
    sentence = Converter('zh-hans').convert(sentence)
    return sentence

'''
if __name__ == "__main__":
    a = '壕鑫互联宣布推出全球首个区块链电竞加速基础服务壕鑫竞斗云'
    a = '12345'
    print (sentence_is_valid(a))

    b = traditional2simplified(a.decode('utf8'))
    print type(b)
    #print b
    print b.encode('utf-8')
    sys.exit()
    b = '壕鑫互联宣布推出全球首个区块链电竞加速基础服务壕鑫竞斗云'
    a = '没有化妆美有ps这才是她真实模样'
    b = '没有化妆没有ps这才是她真实模样'
    print is_str_same_except_punc(a, b)
'''
