from configs.model_config import *
from chains.local_doc_qa import LocalDocQA
import os
import nltk
import sys
import hashlib
import requests
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

    base = sys.argv[1]
    file_name = sys.argv[2]
    file_url = sys.argv[3]

    file_content = requests.get(file_url)
    file_hash = hashlib.md5(file_content.content).hexdigest()
    # if file_name == '':
    #     file_name = os.path.basename(file_url)
    file_name = file_hash + "." + file_name.split('.')[-1]

    filepath = "/var/pyproj/langchain-ChatGLM/history/" + base + "/"
    if not os.path.exists(filepath):
        os.mkdir(filepath)

    filepath += "files/"
    if not os.path.exists(filepath):
        os.mkdir(filepath)

    filepath += file_name
    if not os.path.exists(filepath):
        open(filepath, 'wb').write(file_content.content)
    else:
        return

    # 判断 filepath 是否为空，如果为空的话，重新让用户输入,防止用户误触回车
    if not filepath:
        return

    vs_path = "/var/pyproj/langchain-ChatGLM/history/" + base + "/vs_" + EMBEDDING_MODEL
    vs_path, _ = local_doc_qa.init_knowledge_vector_store(filepath, vs_path=vs_path)

    if not vs_path:
        print('output:vs_path load error')
        return

    print("output:loaded")


if __name__ == "__main__":
    # args = None
    # args = parser.parse_args()
    # args_dict = vars(args)
    # print(args_dict)
    shared.loaderCheckPoint = LoaderCheckPoint(
        {'no_remote_model': False, 'model_name': 'chatglm2-6b', 'lora': None, 'lora_dir': 'loras/',
         'load_in_8bit': False, 'bf16': False})
    main()
