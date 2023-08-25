# *_*coding:utf-8 *_*
# @Author : YueMengRui
import csv

csv.field_size_limit(100000000)
import os
import json
from loguru import logger
from pathlib import Path
from models import build_model
import typer
from mteb import MTEB, AbsTask
from Embedding.mteb.mteb_zh.tasks import (
    GubaEastmony,
    IFlyTek,
    JDIphone,
    StockComSentiment,
    T2RRetrieval,
    TaskType,
    TNews,
    TYQSentiment,
)

default_tasks: list[AbsTask] = [
    TYQSentiment(),
    TNews(),
    JDIphone(),
    StockComSentiment(),
    GubaEastmony(),
    IFlyTek(),
    # T2RReranking(2),
    T2RRetrieval(10000),
    # MedQQPairs(),
]


def filter_by_name(name: str):
    return [task for task in default_tasks if task.description['name'] == name]  # type: ignore


def filter_by_type(task_type: TaskType):
    if task_type is TaskType.All:
        return default_tasks
    else:
        return [task for task in default_tasks if task.description['type'] == task_type.value]  # type: ignore


def main(
        task_type: TaskType = TaskType.All,
        task_name: str | None = None,
        output_folder: Path = Path('results'),
        adapter_path=""
):
    output_folder = Path(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    with open(os.path.join(adapter_path, 'info.json'), 'r') as f:
        data = json.load(f)

    configs = data['configs']

    model = build_model(with_embedding_layer=True,
                        adapter_path=adapter_path,
                        **configs)

    if task_name:
        tasks = filter_by_name(task_name)
    else:
        tasks = filter_by_type(task_type)

    logger.info({'len': len(tasks), 'tasks': tasks})

    while True:
        result_files = os.listdir(output_folder)
        logger.info({'len': len(result_files), 'result_files': result_files})
        if len(result_files) >= len(tasks):
            break

        for task in tasks:
            evaluation = MTEB(tasks=[task])

            try:
                evaluation.run(model, output_folder=str(output_folder), batch_size=64)
            except Exception as e:
                logger.error(e)
                continue

    logger.info('eval done!!!')


if __name__ == '__main__':
    typer.run(main)
