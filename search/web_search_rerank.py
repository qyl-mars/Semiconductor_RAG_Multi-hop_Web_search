import re
import jieba.analyse
from ingest.text2vec import *

class TextRecallRank():
    """
    实现对检索内容的召回与排序
    """

    def __init__(self ,cfg):
        self.topk = cfg.topk    # query关键词召回的数量
        self.topd = cfg.topd    # 召回文章的数量
        self.topt = cfg.topt    # 召回文本片段的数量
        self.maxlen = cfg.maxlen  # 召回文本片段的长度
        self.recall_way = cfg.recall_way  # 召回方式



    def query_analyze(self ,query):
        """query的解析，目前利用jieba进行关键词提取
        input:query,topk
        output:
            keywords:{'word':[]}
            total_weight: float number
        """
        keywords = jieba.analyse.extract_tags(query, topK=self.topk, withWeight=True)
        total_weight = self.topk / sum([r[1] for r in keywords])
        return keywords ,total_weight

    def text_segmentate(self, text, maxlen, seps='\n', strips=None):
        """将文本按照标点符号划分为若干个短句
        """
        text = text.strip().strip(strips)
        if seps and len(text) > maxlen:
            pieces = text.split(seps[0])
            text, texts = '', []
            for i, p in enumerate(pieces):
                if text and p and len(text) + len(p) > maxlen - 1:
                    texts.extend(self.text_segmentate(text, maxlen, seps[1:], strips))
                    text = ''
                if i + 1 == len(pieces):
                    text = text + p
                else:
                    text = text + p + seps[0]
            if text:
                texts.extend(self.text_segmentate(text, maxlen, seps[1:], strips))
            return texts
        else:
            return [text]

    def recall_title_score(self ,title ,keywords ,total_weight):
        """计算query与标题的匹配度"""
        score = 0
        for item in keywords:
            kw, weight =  item
            if kw in title:
                score += round(weight * total_weight ,4)
        return score

    def recall_text_score(self, text, keywords, total_weight):
        """计算query与text的匹配程度"""
        score = 0
        for item in keywords:
            kw, weight = item
            p11 = re.compile('%s' % kw)
            pr = p11.findall(text)
            # score += round(weight * total_weight, 4) * len(pr)
            score += round(weight * total_weight, 4)
        return score

    def rank_text_by_keywords(self ,query ,data):
        """通过关键词进行召回"""

        # query分析
        keywords ,total_weight = self.query_analyze(query)

        # 先召回title
        title_score = {}
        for line in data:
            title = line['title']
            title_score[title] = self.recall_title_score(title ,keywords ,total_weight)
        title_score = sorted(title_score.items() ,key=lambda x :x[1] ,reverse=True)
        # print(title_score)
        recall_title_list = [t[0] for t in title_score[:self.topd]]

        # 召回sentence
        sentence_score = {}
        for line in data:
            title = line['title']
            text = line['text']
            if title  in recall_title_list:
                for ct in self.text_segmentate(text ,self.maxlen, seps='\n。'):
                    ct = re.sub('\s+', ' ', ct)
                    if len(ct )>= 20:
                        sentence_score[ct] = self.recall_text_score(ct ,keywords ,total_weight)

        sentence_score = sorted(sentence_score.items() ,key=lambda x :x[1] ,reverse=True)
        recall_sentence_list = [s[0] for s in sentence_score[:self.topt]]
        return '\n'.join(recall_sentence_list)

    def rank_text_by_text2vec(self, query, data):
        """通过text2vec召回"""
        if not data:
            print("Warning: No data provided for ranking")
            return ""

        # 先召回title
        title_list = [query]
        for line in data:
            title = line['title']
            title_list.append(title)

        # 确保至少有两个标题，否则无法进行相似度计算
        if len(title_list) <= 1:
            print("Warning: Not enough titles for similarity calculation")
            return ""

        title_vectors = get_vector(title_list, 8)

        # 检查向量化是否成功
        if title_vectors.numel() == 0 or title_vectors.size(0) <= 1:
            print("Warning: Title vectorization failed or returned insufficient vectors")
            return ""

        title_score = get_sim(title_vectors)

        # 检查相似度计算是否成功
        if not title_score:
            print("Warning: Title similarity calculation failed")
            return ""

        title_score = dict(zip(title_score, range(1, len(title_list))))
        title_score = sorted(title_score.items(), key=lambda x :x[0], reverse=True)

        # 确保有足够的标题用于召回
        if not title_score or self.topd <= 0:
            print("Warning: No title scores or invalid topd parameter")
            return ""

        recall_title_list = [title_list[t[1]] for t in title_score[:min(self.topd, len(title_score))]]

        # 召回sentence
        sentence_list = [query]
        for line in data:
            title = line['title']
            text = line['text']
            if title in recall_title_list:
                for ct in self.text_segmentate(text, self.maxlen, seps='\n。'):
                    ct = re.sub('\s+', ' ', ct)
                    if len(ct) >= 20:
                        sentence_list.append(ct)

        # 确保至少有两个句子，否则无法进行相似度计算
        if len(sentence_list) <= 1:
            print("Warning: Not enough sentences for similarity calculation")
            return ""

        sentence_vectors = get_vector(sentence_list, 8)

        # 检查向量化是否成功
        if sentence_vectors.numel() == 0 or sentence_vectors.size(0) <= 1:
            print("Warning: Sentence vectorization failed or returned insufficient vectors")
            return ""

        sentence_score = get_sim(sentence_vectors)

        # 检查相似度计算是否成功
        if not sentence_score:
            print("Warning: Sentence similarity calculation failed")
            return ""

        sentence_score = dict(zip(sentence_score, range(1, len(sentence_list))))
        sentence_score = sorted(sentence_score.items(), key=lambda x :x[0], reverse=True)

        # 确保有足够的句子用于召回
        if not sentence_score or self.topt <= 0:
            print("Warning: No sentence scores or invalid topt parameter")
            return ""

        recall_sentence_list = [sentence_list[s[1]] for s in sentence_score[:min(self.topt, len(sentence_score))]]
        return '\n'.join(recall_sentence_list)


