from pathlib import Path
from models import build_model
import typer
from mteb import MTEB, AbsTask
from mteb_zh.tasks import (
    GubaEastmony,
    IFlyTek,
    JDIphone,
    MedQQPairs,
    StockComSentiment,
    T2RReranking,
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
    T2RReranking(2),
    T2RRetrieval(10000),
    MedQQPairs(),
]


def filter_by_name(name: str):
    return [task for task in default_tasks if task.description['name'] == name]  # type: ignore


def filter_by_type(task_type: TaskType):
    if task_type is TaskType.All:
        return default_tasks
    else:
        return [task for task in default_tasks if task.description['type'] == task_type.value]  # type: ignore


def main(
        task_type: TaskType = TaskType.Classification,
        task_name: str | None = None,
        output_folder: Path = Path('results'),
):
    output_folder = Path(output_folder)
    model = build_model(llm_model_name_or_path="", adapter_path="")

    if task_name:
        tasks = filter_by_name(task_name)
    else:
        tasks = filter_by_type(task_type)

    evaluation = MTEB(tasks=tasks)
    evaluation.run(model, output_folder=str(output_folder))


if __name__ == '__main__':
    typer.run(main)
