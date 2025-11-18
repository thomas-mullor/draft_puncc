from typing import Iterable
from deel.puncc.typing import TensorLike
from deel.puncc._keras import ops

def classification_mean_coverage(
    y_true: TensorLike, set_pred: Iterable[Iterable]
) -> float:
    return ops.sum(ops.array([y in s for y, s in zip(y_true, set_pred)])) / len(y_true)

def classification_mean_size(set_pred: Iterable[TensorLike]) -> float:
    return ops.mean(ops.array([len(s) for s in set_pred]))

def regression_mean_coverage(y_true:TensorLike, y_pred_lower:TensorLike, y_pred_upper:TensorLike) -> float:
    return ops.mean((y_true >= y_pred_lower) & (y_true <= y_pred_upper))

def regression_ace(y_true:TensorLike, y_pred_lower:TensorLike, y_pred_upper:TensorLike, alpha:float) -> float:
    cov = regression_mean_coverage(y_true, y_pred_lower, y_pred_upper)
    return cov - (1 - alpha)

def regression_sharpness(y_pred_lower:TensorLike, y_pred_upper:TensorLike) -> float:
    return ops.mean(ops.abs(y_pred_upper - y_pred_lower))

def object_detection_mean_coverage(
    y_pred_outer: TensorLike, y_true: TensorLike
):
    x_min_true, y_min_true, x_max_true, y_max_true = ops.split(y_true, 4, axis=1)
    x_min, y_min, x_max, y_max = ops.split(y_pred_outer, 4, axis=1)
    cov = (
        (x_min <= x_min_true)
        * (y_min <= y_min_true)
        * (x_max >= x_max_true)
        * (y_max >= y_max_true)
    )
    return ops.mean(cov)


def object_detection_mean_area(y_pred: TensorLike):
    x_min, y_min, x_max, y_max = ops.split(y_pred, 4, axis=1)
    return ops.mean((x_max - x_min) * (y_max - y_min))


def iou(bboxes1: TensorLike, bboxes2: TensorLike) -> TensorLike:
    x1_min, y1_min, x1_max, y1_max = ops.split(bboxes1, 4, axis=1)
    x2_min, y2_min, x2_max, y2_max = ops.split(bboxes2, 4, axis=1)

    inter_x_min = ops.maximum(x1_min, ops.transpose(x2_min))
    inter_y_min = ops.maximum(y1_min, ops.transpose(y2_min))
    inter_x_max = ops.minimum(x1_max, ops.transpose(x2_max))
    inter_y_max = ops.minimum(y1_max, ops.transpose(y2_max))

    inter_width = ops.maximum(inter_x_max - inter_x_min + 1, 0)
    inter_height = ops.maximum(inter_y_max - inter_y_min + 1, 0)
    inter_area = inter_width * inter_height

    box1_area = (x1_max - x1_min + 1) * (y1_max - y1_min + 1)
    box2_area = (x2_max - x2_min + 1) * (y2_max - y2_min + 1)

    result = inter_area / (box1_area + ops.transpose(box2_area) - inter_area)
    return result
