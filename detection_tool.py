# -*- coding: UTF-8 -*-
import re
import os,sys

def should_ignore_keywords(ignore_list_keywords, line):
    for ignore_key in ignore_list_keywords:
        pattern1 = re.compile(ignore_key, flags=re.I)
        if len(pattern1.findall(line)) >0:
            return True
    return False


def get_file(keywords_list, ignore_list_keywords):
    for root,dirs,files in os.walk(r"/users/test/example"):    # 不区分操作系统，填写项目路径即可
        for file in files:
            if file.endswith(('.gif','.jpg','.png','.jpeg')):
                continue
            with open(os.path.join(root, file), 'r', encoding='ISO-8859-1') as f:
                for line in f.readlines():
                    line = line.strip()
                    if (should_ignore_keywords(ignore_list_keywords, line) == False):
                        for keyword in keywords_list:
                            pattern = re.compile(keyword, flags=re.I)
                            if len(pattern.findall(line)) > 0:
                                print(os.path.join(root, file) + "存在敏感信息:" + str(pattern.findall(line)[0]))
                                with open('./result.txt', 'a+',  encoding='utf-8') as g:
                                    g.write(os.path.join(root, file) + "存在敏感信息:" + str(pattern.findall(line)[0]) + '\n')


def detection_result():
    if os.path.exists('./result.txt'):
        pass
    else:
        print('=======检测通过，未发现敏感信息=======')

def main():
    keywords_list = ["[^*<\s|:>]{0,7}ak[^]()!<>;/@&,]{0,10}[=:]\s{0,6}[\"']{1}(?=[a-zA-Z]*[0-9])(?=[0-9]*[a-zA-Z])[a-zA-Z0-9]{6,32}[\"']{0,1}","[^*<\s|:>]{0,7}key[^]()!<>;/@&,]{0,10}[=:]\s{0,6}[\"']{1}(?=[a-zA-Z]*[0-9])(?=[0-9]*[a-zA-Z])[a-zA-Z0-9]{6,32}[\"']{0,1}","[^*<\s|:>]{0,7}token[^]()!<>;/@&,]{0,10}[=:]\s{0,6}[\"']{1}(?=[a-zA-Z]*[0-9])(?=[0-9]*[a-zA-Z])[a-zA-Z0-9]{6,32}[\"']{0,1}","[^*<\s|:>]{0,7}pass[^]()!<>;/@&,]{0,10}[=:]\s{0,6}[\"']{1}(?=[a-zA-Z]*[0-9])(?=[0-9]*[a-zA-Z])[a-zA-Z0-9]{6,32}[\"']{0,1}","[^*<\s|:>]{0,7}cookie[^]()!<>;/@&,]{0,10}[=:]\s{0,6}[\"']{1}(?=[a-zA-Z]*[0-9])(?=[0-9]*[a-zA-Z])[a-zA-Z0-9]{6,32}[\"']{0,1}","[^*<\s|:>]{0,7}session[^]()!<>;/@&,]{0,10}[=:]\s{0,6}[\"']{1}(?=[a-zA-Z]*[0-9])(?=[0-9]*[a-zA-Z])[a-zA-Z0-9]{6,32}[\"']{0,1}","[^*<\s|:>]{0,7}app_id[^]()!<>;/@&,]{0,10}[=:]\s{0,6}[\"']{1}(?=[a-zA-Z]*[0-9])(?=[0-9]*[a-zA-Z])[a-zA-Z0-9]{6,32}[\"']{0,1}","[^*<\s|:>]{0,7}appid[^]()!<>;/@&,]{0,10}[=:]\s{0,6}[\"']{1}(?=[a-zA-Z]*[0-9])(?=[0-9]*[a-zA-Z])[a-zA-Z0-9]{6,32}[\"']{0,1}",".{0,15}\.?byted.org.{0,20}",".{0,15}\.?bytedance.net.{0,20}",".{0,20}.bytedance\.feishu\.cn.{0,50}","(10\.\d{1,3}\.\d{1,3}\.\d{1,3})","[^*<\s|:>]{0,7}app_id[^]()!<>;/@&,]{0,10}[=:]\s{0,6}[\"']{0,1}[0-9]{6,32}[\"']{0,1}","[^*<\s|:>]{0,7}appid[^]()!<>;/@&,]{0,10}[=:]\s{0,6}[\"']{0,1}[0-9]{6,32}[\"']{0,1}"]
    ignore_list_keywords = ["[^*<>]{0,6}token[^]()!<>;/@&,]{0,10}[=:].{0,1}null,", ".{0,5}user.{0,10}[=:].{ 0,1}null", ".{0,5}pass.{0,10}[=:].{0,1}null", "passport[=:].", "[^*<>]{0,6}key[^]()!<>;/]{0,10}[=:].{0,1}string.{0,10}", ".{0,5}user.{0,10}[=:].{0,1}string", ".{0,5}pass.{0,10}[=:].{0,1}string",".{0,5}app_id[^]()!<>;/@&,]{0,10}[=:].{0,10}\+",".{0,5}appid[^]()!<>;/@&,]{0,10}[=:].{0,10}\+"]
    get_file(keywords_list, ignore_list_keywords)
    detection_result()


if __name__=="__main__":
    main()