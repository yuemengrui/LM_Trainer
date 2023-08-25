# *_*coding:utf-8 *_*
# @Author : YueMengRui
import os
import json
from loguru import logger
from mteb import MTEB
from mteb.C_MTEB.tasks import *
from mteb.C_MTEB import ChineseTaskList
from models import build_model

if __name__ == '__main__':

    adapter_path = ""
    output_folder = "./results"

    os.makedirs(output_folder, exist_ok=True)

    with open(os.path.join(adapter_path, 'info.json'), 'r') as f:
        data = json.load(f)

    configs = data['configs']

    model = build_model(with_embedding_layer=True,
                        adapter_path=adapter_path,
                        **configs)

    task_names = [t.description["name"] for t in MTEB(task_langs=['zh', 'zh-CN']).tasks]
    logger.info({'len': len(task_names), 'tasks': task_names})

    while True:
        result_files = os.listdir(output_folder)
        logger.info({'len': len(result_files), 'result_files': result_files})
        if len(result_files) >= len(task_names):
            break
        for task in task_names:
            evaluation = MTEB(tasks=[task], task_langs=['zh'])
            try:
                evaluation.run(model, output_folder=output_folder, batch_size=64)
            except Exception as e:
                logger.error(e)
                continue

    logger.info('eval done!!!')
