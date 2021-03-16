# -*- coding:utf-8 -*-
import os
import sys
import urllib.request as urllib2
import urllib
import time
import json
import random
import traceback
import testserver_pb2
#API = 'http://9.107.36.184:6000/typoserver'
#API = 'http://100.115.139.163:9558/typoserver'
#API = 'http://10.101.126.44:9558/typoserver'

#API = 'http://9.4.148.6:9558/typoserver'

#API = 'http://10.165.26.159:9558'

API = 'http://10.160.161.151:9558'

#API = 'http://9.37.138.182:9558'

#API = 'http://9.37.138.182:9558'

#API = 'http://10.160.154.211:9558'

#API = 'http://127.0.0.1:9558/typoserver'

#API = 'http://127.0.0.1:9558/'

#API = 'http://100.115.139.163:9558/'

def request_client(doc_id, title, body_content, channel_id, data_type):
    start_time = time.time()
    request = testserver_pb2.Request()
    request.business_id = b"grpctest"
    request.title = title.encode('utf8')
    request.doc_id = doc_id.encode('utf8')
    request.body_content = body_content.encode('utf8')
    request.channel_id = channel_id
    request.data_type = data_type
    req_post = request.SerializeToString()

    response = testserver_pb2.Response()
    try:
        #headers = {'Content-Type': 'application/protobuf'}
        headers = {'Content-Type': ''}
        req = urllib2.Request(url=API, headers=headers, data=req_post)
        res = urllib2.urlopen(req, timeout=10).read()
        #print type(res)
        response.ParseFromString(res)
    except Exception as e:
        traceback.print_exc(file=sys.stdout)
        return
        
    
    ret_code = response.ret_code
    #print(ret_code, doc_id)
    result = response.typo_result
    title_num = result.title_num
    fst_num = result.fst_num
    all_num = result.all_num
    typo_level = result.typo_level
    typo_pop_result = result.typo_pop_result

    #print(doc_id, title_num, fst_num, all_num, typo_level, typo_pop_result)
    for info in result.details:
        #print(doc_id, info.sno, info.sent.decode('utf-8', 'ignore'))
        print('final_err\t%s\t%d\t%s\t%s\t%d\t%s\t%s\t%d\t%f\t%d\t%d\t%d\n'
             %(doc_id, info.sno, info.sent.decode('utf8'),  info.ori_sent.decode('utf8'), info.idx, info.wrong.decode('utf8'), info.correct.decode('utf8'), info.typo, info.prob, info.term_imp, info.level, info.pop_result)) 


def request_client_json(doc_id, title, body_content, channel_id, data_type):
    request = {}
    request['business_id'] = "grpctest"
    request['title'] = '北京票亮怎么样'
    request['doc_id'] = 'doc_id'
    request['body_content'] = ''
    #request['channel_id'] = b""
    #request['data_type'] = b""
    #request['channel_name'] = b""
    API = 'http://127.0.0.1:9558/typoserver_debug'
    try:
        #headers = {'Content-Type': 'application/protobuf'}
        headers = {'Content-Type': 'application/json'}
        #req_post = urllib.parse.urlencode(request).encode('utf8')
        #print(req_post)
        #sys.exit()
        #req_post = json.dumps(request).encode('utf8')
        req = urllib2.Request(url=API, headers=headers, data=json.dumps(request).encode('utf8'))
        res = urllib2.urlopen(req, timeout=10).read()
        print(type(res))
        print (res.decode('utf8'))
        data = json.loads(res.decode('utf8'))
        print(data['typo_result'])
    except Exception as e:
        traceback.print_exc(file=sys.stdout)
        return
        
    """ 
    ret_code = response.ret_code
    result = response.typo_result
    title_num = result.title_num
    fst_num = result.fst_num
    all_num = result.all_num
    typo_level = result.typo_level
    typo_pop_result = result.typo_pop_result

    for info in result.details:
        print('final_err\t%s\t%d\t%s\t%s\t%d\t%s\t%s\t%d\t%f\t%d\t%d\t%d\n'
             %(doc_id, info.sno, info.sent.decode('utf8'),  info.ori_sent.decode('utf8'), info.idx, info.wrong.decode('utf8'), info.correct.decode('utf8'), info.typo, info.prob, info.term_imp, info.level, info.pop_result)) 
    """
 
def client_debug():
    total_ts, cnt = 0, 0
    for line in open(sys.argv[1], 'r', encoding="utf-8"):
        lines = line.strip().split("\t")
        sts = time.time()
        request_client(lines[0], lines[1], lines[2], 0, 0)

        #request_client_json(lines[0], lines[1], lines[2], 0, 0)

        elapse = (time.time() - sts) * 1000
        #print("docid: %s cost: %.4f" %(lines[0], elapse))
        total_ts += elapse
        cnt += 1
    print(total_ts, cnt, total_ts/cnt)
         
def client_debug_batch():
    path = "./data/2019_06_04.inc.data"
    count = 0
    with open(path, 'r') as fr:
        for line in fr.readlines():
            count += 1
            line = line.strip().split("\t")
            cmsid = line[0]
            title = line[7]
            body_content = line[8]
            channel_id = 12
            data_type = 0
            doc_id, typo_title_num, typo_fst_num, typo_all_num, typo_level, typo_pop_result, typo_details, grammar_title_num, grammar_fst_num, grammar_all_num, grammar_level, grammar_pop_result, \
                grammar_details, ret_code = request_client(cmsid, title, body_content, channel_id, data_type)
            #print "Typo: ", doc_id, typo_title_num, typo_fst_num, typo_all_num, typo_level, typo_pop_result, ret_code
            for info in typo_details:
                sno, idx, sent, wrong, correct, type, prob, ori_sent, term_imp, level, pop_result = info
                print("Typo: ", sno, idx, sent, wrong, correct, type, prob, ori_sent, term_imp, level, pop_result)
            if grammar_all_num == 0:
                #continue
                pass
            #print "Grammar: ", doc_id, grammar_title_num, grammar_fst_num, grammar_all_num, grammar_level, grammar_pop_result, ret_code
            for info in grammar_details:
                sno, sent, type, prob, ori_sent, level, correct, pop_result = info
                print("Grammar: ", sno, sent, type, prob, ori_sent, level, correct, pop_result)
def client_video_debug_batch():
    path = "./data/kaifangji_0813.dat"
    count = 0
    with open(path, 'r') as fr:
        for line in fr:
            count += 1
            line = line.strip().split("\t")
            cmsid = line[0]
            title = line[-1]
            body_content = ""
            channel_id = 0
            data_type = 1
            doc_id, typo_title_num, typo_fst_num, typo_all_num, typo_level, typo_pop_result, typo_details, grammar_title_num, grammar_fst_num, grammar_all_num, grammar_level, grammar_pop_result, \
                grammar_details, ret_code = request_client(cmsid, title, body_content, channel_id, data_type)
            print("Typo: ", doc_id, typo_title_num, typo_fst_num, typo_all_num, typo_level, typo_pop_result, ret_code)
            for info in typo_details:
                sno, idx, sent, wrong, correct, type, prob, ori_sent, term_imp, level, pop_result = info
                #print sno, idx, sent, wrong, correct, type, prob, ori_sent, term_imp, level, pop_result
            if grammar_all_num == 0:
                continue
                pass
            print("Grammar: ", doc_id, grammar_title_num, grammar_fst_num, grammar_all_num, grammar_level, grammar_pop_result, ret_code)
            for info in grammar_details:
                sno, sent, type, prob, ori_sent, level, correct, pop_result = info
                print(sno, sent, type, prob, ori_sent, level, correct, pop_result)
if __name__ == '__main__':
    client_debug()
    #client_debug_batch()
    #client_video_debug_batch()

