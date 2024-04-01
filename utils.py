import torch
from collections import Counter


def iou(box1: torch.Tensor, box2: torch.Tensor):
    x1, y1 = torch.max(box1[..., 0:1], box2[..., 0:1]), torch.max(box1[..., 1:2], box2[..., 1:2])
    x2, y2 = torch.min(box1[..., 2:3], box2[..., 2:3]), torch.min(box1[..., 3:4], box2[..., 3:4])

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    union = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1]) + \
            (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1]) - intersection
    return intersection/union


def nms(predictions: torch.Tensor, iou_threshold, prob_threshold):
    bboxes = [box for box in predictions if box[1] > prob_threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    while bboxes:
        chosen_box = bboxes.pop(0)
        bboxes = [box for box in bboxes if chosen_box[0] != box[0] or iou(box[2:], chosen_box[2:]) < iou_threshold]
    bboxes.append(chosen_box)
    return bboxes


def mAP(prediction_boxes, true_boxes, iou_threshold=0.5, num_classes=20):
    average_precisions = []
    eps = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths = []

        for detection in prediction_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros(len(detections))
        FP = torch.zeros(len(detections))
        total_true_bboxes = len(ground_truths)

        for detection_idx, detection in enumerate(detections):
            ground_truth_image = [bbox for bbox in ground_truths if bbox[0] == detection[0]]
            num_ground_truths = len(ground_truth_image)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_image):
                my_iou = iou(detection[3:], gt[3:])
                if my_iou > best_iou:
                    best_iou = my_iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum/(total_true_bboxes + eps)
        precisions = TP_cumsum/(TP_cumsum + FP_cumsum + eps)
        precisions = torch.cat(torch.Tensor([1]), precisions)
        recalls = torch.cat(torch.Tensor([0]), recalls)
        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions)/len(average_precisions)


