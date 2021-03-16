# TypoDetect
Chinese Spelling Check

一、错别字架构代码介绍
基于策略规则和语言模型将各个位置替换候选字后计算ppl降比，达到阈值则认为有错误并纠正。

二、主服务模块
![Alt text](https://github.com/yangshoujian/TypoDetect/blob/main/packages/%E9%94%99%E5%88%AB%E5%AD%97%E6%9E%B6%E6%9E%84%E5%9B%BE.png)
ResourceManager: 词典资源管理类，管理资源类数据的加载和访问等
ProcessorManger: 策略管理类，管理ResourceManager类加载，调用PrePostProcessor对文章预处理、各个ProcessManager之间的流水线调用
PrePostProcessor: 文章预处理类、对文章分段、分句，分词识别ner，统计专名、人称代词等统计信息
DocInfo：存储文档属性数据
Flyweight：存储单句属性数据（单句以。！；等符号分割）

三、策略逻辑
1.  文章预处理：文章分段、分句，逐句分词识别专名，存储单句属性信息到Flyweight；统计文章整体信息：如人名出现频次等，存储在DocInfo中
2.  整句识别过程：
1)  判断是否为古文，如果为古文直接返回不处理
2） 句子按逗号分割成多个短句, 每个短句如下处理：
A. 使用规则策略识别成语、专名错误
B. 借助ngramlm各种类型识别策略判断短句是否有错、并进行纠错；
判断是否有错的逻辑：使用混淆词典替换当前位置字词，计算替换字词后句子ppl与当前句子ppl比例，达到一定阈值则直接接受(替换字词为正确字词，原字词错误）； 
如果没有达到直接接受的阈值，但在一个范围内，则请求序列标注模型判断错误位置
3） 使用序列标注模型判断整句错误位置，如果序列标注模型及ngramlm均识别有错、但错误位置不同，以序列标注识别位置为准
3. 短句ngramlm各识别策略先后顺序如下：
（1） __judge_guwen：判断整句是否为古文
（2） __split_sent： 将整句以逗号分隔，切分为多个子句
（3） norm_sent_for_ngramlm： 子句归一化处理，去除标点符号等（ngramlm计算的需求）
（4） __detect_chengyu_error_pinyin：识别成语错误
（5） __detect_all_ner_error：识别专名错误
（6） __detect_fixed_error_py：识别同前缀或后缀的词语错误
（7） __detect_single_char_nonword_error：识别单字非词错误，如“北京是手都” 中“首都”错写成“手都”
（8） __detect_single_char_word_error 识别单字错误， 如“北京时首都”
（9） __detect_word_error 识别词语错误，如”他男朋友很帅旗”
（10） __detect_multi_single_char_error 识别多字成词错误，如”他/女朋友/很/票/量/“
（11） __detect_nnlm_errors 调用NNLM模型判断错误字及位置
（12） __detect_one_to_multi_error 识别一个词拆分成多个字的错误
（13） __adjust_error_info  综合多个子句错误、序列标注模型识别错误结合综合判定句子错误位置及修改词
