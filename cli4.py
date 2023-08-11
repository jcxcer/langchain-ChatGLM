from configs.model_config import *
from chains.local_doc_qa import LocalDocQA
import os
import nltk
import sys
import json
import hashlib
import requests
from models.loader.args import parser
import models.shared as shared
from models.loader import LoaderCheckPoint

nltk.data.path = [NLTK_DATA_PATH] + nltk.data.path

# Show reply with source text from input document
REPLY_WITH_SOURCE = True

HISTORY_PATH = "/var/pyproj/langchain-ChatGLM/history"


def saveHistory(conversation_key, history):
    with open(f"{HISTORY_PATH}/{conversation_key}.txt", "w", encoding="utf-8") as f:
        f.write(json.dumps(history, ensure_ascii=False))


def loadHistory(conversation_key):
    history = []
    if os.path.exists(f"/{HISTORY_PATH}/{conversation_key}.txt"):
        with open(f"{HISTORY_PATH}/{conversation_key}.txt", "r", encoding="utf-8") as f:
            history = f.read()
            history = json.loads(history)
    return history


def main():
    llm_model_ins = shared.loaderLLM()
    llm_model_ins.history_len = LLM_HISTORY_LEN

    local_doc_qa = LocalDocQA()
    local_doc_qa.init_cfg(llm_model=llm_model_ins,
                          embedding_model=EMBEDDING_MODEL,
                          embedding_device=EMBEDDING_DEVICE,
                          top_k=VECTOR_SEARCH_TOP_K)

    # query = sys.argv[1]
    # base = sys.argv[2]

    query_text = ''
    query_base = ''

    query_file = sys.argv[1]
    if os.path.exists(f"{HISTORY_PATH}/tmp/" + query_file):
        with open(f"{HISTORY_PATH}/tmp/" + query_file, "r", encoding="utf-8") as f:
            query = f.read()
            query = json.loads(query)

            query_text = query["text"]
            query_base = query["base"]

    if query_text == '' or query_base == '':
        print('output:query or base load error')
        return

    vs_path = f"{HISTORY_PATH}/" + query_base + "/vs_" + EMBEDDING_MODEL
    # vs_path, _ = local_doc_qa.init_knowledge_vector_store(filepath,vs_path=vs_path)

    if not vs_path:
        print('output:vs_path load error')
        return

    history = []

    # query = input("Input your question 请输入问题：")

    last_print_len = 0

    print("output:")

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
    shared.loaderCheckPoint = LoaderCheckPoint(
        {'no_remote_model': False, 'model_name': 'chatglm2-6b', 'lora': None, 'lora_dir': 'loras/',
         'load_in_8bit': False, 'bf16': False})
    main()
