from configs.model_config import *
from chains.local_doc_qa import LocalDocQA
import os
import nltk
import sys
import hashlib
import requests
import json
from models.loader.args import parser
import models.shared as shared
from models.loader import LoaderCheckPoint
nltk.data.path = [NLTK_DATA_PATH] + nltk.data.path

# Show reply with source text from input document
REPLY_WITH_SOURCE = True


def main():
    llm_model_ins = shared.loaderLLM()
    llm_model_ins.history_len = LLM_HISTORY_LEN

    local_doc_qa = LocalDocQA()
    local_doc_qa.init_cfg(llm_model=llm_model_ins,
                          embedding_model=EMBEDDING_MODEL,
                          embedding_device=EMBEDDING_DEVICE,
                          top_k=VECTOR_SEARCH_TOP_K)
    # filepath = input("Input your local knowledge file path 请输入本地知识文件路径：")
    # query = sys.argv[1]
    # file_name = sys.argv[2]
    # file_url = sys.argv[3]

    query_text = ''
    query_url = ''
    query_file_name = ''

    query_file = sys.argv[1]
    if os.path.exists(f"/var/pyproj/langchain-ChatGLM/history/tmp/" + query_file):
        with open(f"/var/pyproj/langchain-ChatGLM/history/tmp/" + query_file, "r", encoding="utf-8") as f:
            query = f.read()
            query = json.loads(query)

            query_text = query["text"]
            query_url = query["url"]
            if 'file_name' in query:
                query_file_name = query["file_name"]

    if query_text == '' or query_url == '':
        print('output:query or url load error')
        return

    file_content = requests.get(query_url)
    file_hash = hashlib.md5(file_content.content).hexdigest()
    if query_file_name == '':
        file_name = os.path.basename(query_url)

    filepath = "/var/pyproj/langchain-ChatGLM/history/" + file_hash + "/"
    if not os.path.exists(filepath):
        os.mkdir(filepath)

    filepath += "files/"
    if not os.path.exists(filepath):
        os.mkdir(filepath)

    filepath += file_name
    if not os.path.exists(filepath):
        open(filepath, 'wb').write(file_content.content)

    # 判断 filepath 是否为空，如果为空的话，重新让用户输入,防止用户误触回车
    if not filepath:
        return

    vs_path = "/var/pyproj/langchain-ChatGLM/history/"+file_hash+"/vs_"+EMBEDDING_MODEL
    vs_path, _ = local_doc_qa.init_knowledge_vector_store(filepath,vs_path=vs_path)

    if not vs_path:
        print('output:vs_path load error')
        return

    history = []

    # query = input("Input your question 请输入问题：")

    last_print_len = 0

    print("output:")
    if query_text == 'OCR':
        with open("/var/pyproj/langchain-ChatGLM/history/" + file_hash + "/files/tmp_files/" + file_name + ".txt",
                  "r") as f:
            txt = f.readlines()
            print("".join(txt))
    else:
        for resp, history in local_doc_qa.get_knowledge_based_answer(query=query_text,
                                                                     vs_path=vs_path,
                                                                     chat_history=history,
                                                                     streaming=STREAMING):
            if STREAMING:
                print(resp["result"][last_print_len:], end="", flush=True)
                last_print_len = len(resp["result"])
            else:
                print(resp["result"])


if __name__ == "__main__":
    # args = None
    # args = parser.parse_args()
    # args_dict = vars(args)
    # print(args_dict)
    shared.loaderCheckPoint = LoaderCheckPoint({'no_remote_model': False, 'model_name': 'chatglm2-6b', 'lora': None, 'lora_dir': 'loras/', 'load_in_8bit': False, 'bf16': False})
    main()
