import requests
import re
from lxml import etree
import chardet
from config.configs import Config
from search.web_search_rerank import TextRecallRank
import requests

# 1. 联网搜索:总指挥函数
def get_web_search_content(query: str, max_length: int = 1500) -> str:
    """
        对外的主入口：搜索 -> 排序 -> 返回文本
    """
    try:
        search_results = web_search_and_rerank(query)
        cleaned_results = re.sub(r'\s+', ' ', search_results).strip()
        return cleaned_results[:max_length]
    except Exception as e:
        print(f"联网搜索失败：{str(e)}")
        return ""

# ---------------------------------------------------
# 2. 爬虫函数 (底层工具)：核心联网搜索功能
# ---------------------------------------------------

# 引入 Tavily
from tavily import TavilyClient
def search_bing(query):
    """
    使用 Tavily API 进行搜索 (专业 RAG 搜索服务，无需代理，无需 Cookie)
    """
    print(f"DEBUG: 正在调用 Tavily 搜索: {query}")
    results = []

    # 🔑 去 https://tavily.com/ 注册领一个免费 Key
    api_key = Config.tavily_api_key

    try:
        tavily = TavilyClient(api_key=api_key)

        # search_depth="advanced" 会更深入，但慢一点；"basic" 较快
        response = tavily.search(query=query, search_depth="basic", max_results=5)

        # Tavily 的返回结构非常干净
        for result in response.get('results', []):
            results.append({
                'title': result.get('title'),
                'url': result.get('url'),
                # content 是它帮你清洗好的正文，直接用！
                'text': result.get('content')
            })

    except Exception as e:
        print(f"Tavily 搜索出错: {e}")

    return results

"""
use_duckgo_search = False
if use_duckgo_search:
    from duckduckgo_search import DDGS
    def search_bing(query):  # 函数名可以保留叫 search_bing 方便兼容，虽然实际用的是 DDG
        # ---
        # 使用 DuckDuckGo 进行搜索 (无需 Cookie，更稳定)
        # ---
        print(f"DEBUG: 正在调用 DuckDuckGo 搜索: {query}")
        results = []
    
        try:
            with DDGS() as ddgs:
                # 搜索前 5 条结果
                ddg_gen = ddgs.text(query, region='cn-zh', max_results=5)
    
                for r in ddg_gen:
                    results.append({
                        'title': r['title'],
                        'url': r['href'],
                        'text': r['body']  # DDG 直接返回摘要，不用自己 xpath 解析了
                    })
    
        except Exception as e:
            print(f"DuckDuckGo 搜索出错: {e}")
    
        return results
    
"""
"""
爬虫cookies过期，需要手动跟新cookies很麻烦，不采用此方案
def search_bing(query):
    # ------------  
    # 利用newbing搜索接口，用于检索与query相关的背景信息，作为检索内容
    # input：query
    # output：{'url':'','text':'','title':''}
    # ---------------
    headers = {
        'Cookie': 'MUID=2CFCFC26663D64393955ED1C623D62A4; MUIDB=2CFCFC26663D64393955ED1C623D62A4; SRCHD=AF=S00028; SRCHUID=V=2&GUID=76DC1CA8309043BBAB81CFC4C47D76DD&dmnchg=1; _UR=QS=0&TQS=0; MicrosoftApplicationsTelemetryDeviceId=64c1979f-ee59-40a7-928e-b755865bc6ae; ABDEF=V=13&ABDV=13&MRNB=1696643925014&MRB=0; ANON=A=15BC3EC2F3AC041DAD2C715CFFFFFFFF&E=1d05&W=2; NAP=V=1.9&E=1cab&C=MnJiRko1YRJfqV6H22giKijH0-4G1Ub50-Cg7gnMPMN4QFF_OeDZsQ&W=2; PPLState=1; _HPVN=CS=eyJQbiI6eyJDbiI6NiwiU3QiOjAsIlFzIjowLCJQcm9kIjoiUCJ9LCJTYyI6eyJDbiI6NiwiU3QiOjAsIlFzIjowLCJQcm9kIjoiSCJ9LCJReiI6eyJDbiI6NiwiU3QiOjAsIlFzIjowLCJQcm9kIjoiVCJ9LCJBcCI6dHJ1ZSwiTXV0ZSI6dHJ1ZSwiTGFkIjoiMjAyMy0xMi0xMVQwMDowMDowMFoiLCJJb3RkIjowLCJHd2IiOjAsIlRucyI6MCwiRGZ0IjpudWxsLCJNdnMiOjAsIkZsdCI6MCwiSW1wIjoxNiwiVG9iYnMiOjB9; _EDGE_S=SID=2E9C7DC6F71A6094195D6E28F6C8614B; USRLOC=HS=1&ELOC=LAT=30.26828956604004|LON=120.13351440429688|N=%E8%A5%BF%E6%B9%96%E5%8C%BA%EF%BC%8C%E6%B5%99%E6%B1%9F%E7%9C%81|ELT=4|; _Rwho=u=d; _SS=SID=2E9C7DC6F71A6094195D6E28F6C8614B&R=64&RB=64&GB=0&RG=0&RP=61; KievRPSSecAuth=FABKBBRaTOJILtFsMkpLVWSG6AN6C/svRwNmAAAEgAAACFAPF0afImbrCASVm1xT1K+FiXTsQgaoq6RydL+Ge3FvFrIbHVbXa7m0OlZNQJT4P62pu6xUtDTqwGPUE13tWBwVPkK1RahHVaGuUSLfwbp5o2HeLnKa+hfc6+sJiYHnxklhiJAzdi/oBbiWdDkf+5A+C0Fbsxeo4pQDt+kmeKhWpMwijA0bVP5ISXdkrLsRv5jiq97srkAMWFHqqGboI70LdX7ahqSSiykzwFdA1np3WhYhURWQ4b3z6uV7nsZpth6lpdafGZ2YLWr0Zwpv1D210P04ovzbbzvuKGoeljS4/SvdX8QUoGONzn0f2OXAOPvsnZJctbwxH/tkembDlpN4liJDCYhlYgoKtg5nuLBNihk75VctLodAQhosDNYM9stJRzQlusK+aEbDQKAgXunPwB0iPq0ECEVmLIApOeXs7DEtj29Q8zuWiOmxXnddGDm4Tf0VWUVjAEfP/PKiiTLAAS/dwPgOslgEdpy3Pw6GQYo3z3dZ16mWuXYX53utgdkK4rtqRj/FmYiTRjL6scm7Ds0UJnVNxdJcFACadTOzNVEGBp2XIb6XEAWZThz21+JJCn325RXG+zwJyjaKI941n6CbQ8Z/dXgUYMBsn/gfdGV3/+vz05pIOtB1zmzkvwds5v4M/zTcf5fgqWwLjSbIBFscYA626llQwDS6LkKwyoB/EB3L0XgLnOFpoSSpk41L/q5e0GkLVxzPA5kZue0iLTNEXUu/KCGnPOBkK0iAZVS/bJPVa3ZBPBOODwXnAUR0s0W1hbHLDW4I1ZrMuujx08DU0/nhhiq0mFgwwxHrd4vE9xdecjlpyL78pzPf5LVAiCKaWZ/BnKqHCYHA4hiEg8ffC5eFwoA6JsL0wtvTSdaAPEcUs103Um9eje8nNKwvDtesqh93lOAbNCfkfC/zAdtsR0dWaZIsYdAeMNQE+6//zLDbGIe24WVsSdiwZqdmYI2ICxE+KqPY++Ei4gfgKt0GNyiAfK0qSfALb01POx95rWOyWSPd0ge6DwM5mHAZfTePR44vBfFdhvUYBg0+47nOzY53hcO/6unDb3u1/PLHM7+rlS+76yjrZ9Dl7cFXRNBINy5afDUY+kU9JQS6QTbc5EIQTynlWkGU38m43GtWXquitzrjHuC0mYsUbLQOuZ1kFWHQXF/4I/aaxU1k0uvEOttkIUkhXi5lKo9uLoPGdha+AIGcDz8icpdDnfAHHpChm0YB8K8lcL0foY6NCib+o+LCLfriZg9Nvtkc8s1+TWPvCvHZX4bZuXyN4tHoQiysRd6j0gyJpLR4yQr5iOyBUgkM9WWKzkFmnzVYlb4ec6wpowsw2643AHs5Ge1FDjzKw3TdSVnwB2dHFh7tdNW1ywYDAGhpv8SSvQ66448UANVqB1uKwxsD0mXJR/tjMy9OuiNR; _U=1S7ve-XVb_pOh5Iig5kQlQDI6wv9BNl9HiCEtz0dS6dNV_UrQUBmAFVEZx7pYNRTwRxGG8eASH_IDUlpJu04SCp8aeYlPHkU_-0xGzlVA3nTqaE9kSUyIm1UVQYovjbOrsh4SeBbU-wrjqz6HV2DeUKJiHyTwYlDeQ8bYboyqhB4-ER5PjMGcp8daGbur9ER2KSm-nJOeUqnWeIawk0BVyw; WLS=C=26d7831f7a57e7fd&N=; SRCHUSR=DOB=20220207&T=1703137884000&TPC=1703137888000&POEX=W; SRCHHPGUSR=SRCHLANG=zh-Hans&BZA=0&BRW=HTP&BRH=M&CW=975&CH=931&SW=1920&SH=1080&DPR=1&UTC=480&DM=0&HV=1703137888&WTS=63838734684&PRVCW=975&PRVCH=931&SCW=1164&SCH=2821&PV=10.0.0; WLID=mA9cZwKIoBbQ9h07pbmfeYJEn7iBxd5sk7A9mKFJf1dP4SWmtri4X9d1xcl06hKEVmEpT+5GB21NeHYv/uk3maNbHalTEB+UwCwfS7RdzoQ=; _RwBf=r=0&ilt=1&ihpd=1&ispd=0&rc=64&rb=64&gb=0&rg=0&pc=61&mtu=0&rbb=0.0&g=0&cid=&clo=0&v=1&l=2023-12-20T08:00:00.0000000Z&lft=0001-01-01T00:00:00.0000000&aof=0&o=0&p=bingcopilotwaitlist&c=MY00IA&t=3001&s=2023-03-20T09:14:27.6545941+00:00&ts=2023-12-21T05:51:26.7082437+00:00&rwred=0&wls=2&wlb=0&lka=0&lkt=0&aad=0&TH=&mta=0&e=CS-LRz6MT6YjyZDqHmn2zXGq0iVnD2Plg7iI7uA3t-iwF4TTPdW2rejPh5N_c6syhuNr1-uNgqm8vKVLqjaaig&A=15BC3EC2F3AC041DAD2C715CFFFFFFFF&wle=1&ccp=0&ard=0001-01-01T00:00:00.0000000; ipv6=hit=1703141490169&t=4',
        'Accept-Encoding': 'gzip, deflate',
        'Accept-Language': 'zh-CN,zh;q=0.9',
        'Accept': '*/*',
        'Referer': 'https://cn.bing.com/search?',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36'
        }

    res = []
    url = 'https://cn.bing.com/search?q=' + query + '&qs=n&form=QBRE'
    r = requests.get(url, headers=headers)
    try:
        encoding = chardet.detect(r.content)['encoding']
        r.encoding = encoding
        dom = etree.HTML(r.content.decode(encoding))
    except:
        dom = etree.HTML(r.content)

    url_list = []
    tmp_url = []
    #只采集列表的第一页
    for sel in dom.xpath('//ol[@id="b_results"]/li/h2'):
        l = ''.join(sel.xpath('a/@href'))
        title = ''.join(sel.xpath('a//text()')).split('-')[0].strip()
        if 'http' in l and l not in tmp_url and 'doc.' not in l:
            url_list.append([l,title])
            tmp_url.append(l)
    for turl,title in url_list:
        try:
            tr = requests.get(turl, headers=headers, timeout=(5, 5))
            tdom = etree.HTML(tr.content.decode('utf-8'))
            text = '\n'.join(tdom.xpath('//p/text()'))
            if len(text) > 15:
                tmp = {}
                tmp['url'] = turl
                tmp['text'] = text
                tmp['title'] = title
                res.append(tmp)
        except Exception as e:
            print(e)
            pass
    return res
"""
# ---------------------------------------------------
# 3. 编排函数
# ---------------------------------------------------
def web_search_and_rerank(query):
    """
         完整的搜索+排序流程
         Input: query (str)
         Output: bg_text (str) - 排序并拼接好的文本
    """
    # 1. 利用搜索引擎获取相关信息
    raw_data = search_bing(query)
    if not raw_data:
        return "未找到相关网络信息"

    # 2. 初始化配置和排序器
    # 注意：这里我们在函数内部按需实例化，不仅节省资源，还解耦
    cfg = Config()
    trr = TextRecallRank(cfg)

    # 3. 根据配置选择排序方式 (逻辑从原 web_search_rerank.py 搬运过来)
    if trr.recall_way == 'keyword':
        bg_text = trr.rank_text_by_keywords(query, raw_data)
    else:
        bg_text = trr.rank_text_by_text2vec(query, raw_data)

    # 兜底：如果排序后没内容，至少返回一点原始的
    if not bg_text:
        return f"搜索到 {len(raw_data)} 条结果，但排序后内容为空。"

    return bg_text


