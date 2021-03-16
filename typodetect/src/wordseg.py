# -*- coding:utf-8 -*-
import logging
import sys
import os
import segmentor_4_python3

# LOCAL = threading.local()


class Segment():

    def __init__(self, dict_path):
        if not segmentor_4_python3.TCInitSeg(dict_path):
            logging.error('[error]:init_word_seg')
            sys.exit(-1)
        else:
            logging.info('[%d] load_worseg_data' % (os.getpid()))

        self.handle = segmentor_4_python3.TCCreateSegHandle(
            segmentor_4_python3.TC_CRF
            | segmentor_4_python3.TC_POS
            | segmentor_4_python3.TC_VIDEO
            | segmentor_4_python3.TC_PER_W
            | segmentor_4_python3.TC_PRODUCTION)
        logging.warning('init_qqseg_handle:%s' % (str(id(self.handle))))

    def close_handle(self, handle):
        segmentor_4_python3.TCCloseSegHandle(handle)

    def get_seg_result(self, string):
        handle = segmentor_4_python3.TCCreateSegHandle(
            segmentor_4_python3.TC_CRF
            | segmentor_4_python3.TC_POS
            | segmentor_4_python3.TC_VIDEO
            | segmentor_4_python3.TC_PER_W
            | segmentor_4_python3.TC_PRODUCTION)
        segmentor_4_python3.TCSegment(
            handle, string,
            len(string.encode('utf8')), segmentor_4_python3.TC_UTF8)
        basic_result = self.get_basic_words(handle)
        mix_result = self.get_mix_words(handle)
        ner_result = self.get_ner_names(handle)
        self.close_handle(handle)
        return basic_result, mix_result, ner_result

    def segment(self, string):
        segmentor_4_python3.TCSegment(
            self.handle, string,
            len(string.encode('utf8')), segmentor_4_python3.TC_UTF8)
        return self.handle

    def get_basic_words(self, handle):
        result = []
        words_count = segmentor_4_python3.TCGetResultCnt(handle)
        for i in range(words_count):
            token = segmentor_4_python3.TCGetBasicTokenAt(handle, i)
            result.append((token.word, token.pos, token.wlen))
        return result

    def get_mix_words(self, handle):
        result = []
        words_count = segmentor_4_python3.TCGetMixWordCnt(handle)
        for i in range(words_count):
            token = segmentor_4_python3.TCGetMixTokenAt(handle, i)
            result.append((token.word, token.pos, token.wlen))
        return result

    def get_phrase_words(self, handle):
        result = []
        words_count = segmentor_4_python3.TCGetPhraseCnt(handle)
        for i in range(words_count):
            token = segmentor_4_python3.TCGetPhraseTokenAt(handle, i)
            result.append((token.word, token.pos, token.wlen,
                           token.cls, token.sidx, token.eidx))
        return result

    def get_ner_names(self, handle):
        ner_infos = []
        words_count = segmentor_4_python3.TCGetPhraseCnt(handle)
        for i in range(words_count):
            word = segmentor_4_python3.TCGetPhraseAt(handle, i)
            token = segmentor_4_python3.TCGetPhraseTokenAt(handle, i)
            cls, sidx, eidx = int(token.cls), token.sidx, token.eidx
            if cls in [segmentor_4_python3.PHRASE_NAME_IDX,
                       segmentor_4_python3.PHRASE_NAME_FR_IDX]:
                ner_infos.append((word, sidx, eidx, 0))
            elif cls == segmentor_4_python3.PHRASE_LOCATION_IDX:
                ner_infos.append((word, sidx, eidx, 1))
            elif cls == segmentor_4_python3.PHRASE_ORGANIZATION_IDX:
                ner_infos.append((word, sidx, eidx, 2))
            elif (cls in [segmentor_4_python3.PHRASE_VIDEO_IDX,
                          segmentor_4_python3.PHRASE_VIDEO_MOVIE_IDX,
                          segmentor_4_python3.PHRASE_VIDEO_TVSERIES_IDX,
                          segmentor_4_python3.PHRASE_VIDEO_TVSHOW_IDX]):
                ner_infos.append((word, sidx, eidx, 3))
            elif cls == segmentor_4_python3.PHRASE_PRODUCTION_IDX:
                ner_infos.append((word, sidx, eidx, 4))
        return ner_infos


"""
if __name__ == '__main__':
    seg = Segment('../py3_grammarerror_baseline/common/data/')
    SENT = '精英律师：何塞一身粉色碎花裙，硬生生往罗槟身上靠，看的罗槟笑弯腰'
    print(type(SENT))
    seg.segment(SENT)
    print('--basic---')
    info = seg.get_basic_words()
    for item in info:
        print(item[0], item[1], item[2], type(item[0]))
    nerinfo = seg.get_ner_names()
    print('--ner---')
    for i in nerinfo:
        print(i[0], i[1], i[2], i[3], type(i[0]))
"""
