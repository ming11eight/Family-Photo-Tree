import json
import numpy as np
from PIL import Image
import cv2


def load_file_dict(file_source):
    """
    file_source : image file dict의 txt파일 경로
                ex.)'/home/jupyter/data/dict/sample_file_dict.txt'
    """
    with open(file_source, "r") as f:
        file_dict = json.load(f)
    # assert len(file_dict) == 258
    assert len(file_dict) == 95148
    return file_dict


def load_bbox_dict(bbox_source):
    with open(bbox_source, "r") as f:
        bbox_dict = json.load(f)
    # assert len(bbox_dict) == 258
    assert len(bbox_dict) == 95148
    return bbox_dict


def gt_detect(boxes, img=None):
    """
    ground truth의 바운딩 박스 정보(x_좌상단,y_좌상단,x_우하단,y_우하단)를 반환한다.
    """
    box_label = []
    # boxes = sample_bbox_dict[sample_file_key]
    for box in boxes:
        x_left = box[1]
        x_right = int(box[1]) + int(box[3])
        y_left = box[2]
        y_right = int(box[2]) + int(box[4])
        box_ = list([int(x_left), int(y_left), int(x_right), int(y_right)])
        box_label.append(box_)
        # 사진에 바운딩 박스 그리기
        # img = cv2.rectangle(img, (int(x_left), int(y_left)), (int(x_right), int(y_right)),
        #                 (0, 0, 255),3)
    return box_label


def mtcnn_detect(img: np.ndarray, mtcnn, orig_box=False) -> np.ndarray:
    """
    예측한 바운딩 박스 정보(x_좌상단,y_좌상단,x_우하단,y_우하단)를 반환한다.
    """
    boxes, probs = mtcnn.detect(img)
    box_pred = []
    if boxes is None:
        return box_pred
    for box in boxes:
        x_left = min(box[0], box[2])
        x_right = max(box[0], box[2])
        y_left = min(box[1], box[3])
        y_right = max(box[1], box[3])
        box_area = abs(x_right - x_left) * (y_right - y_left)
        # print(box_area)
        if box_area<5000:
            continue
        # 사진에 바운딩 박스 그리기
        # img = cv2.rectangle(img, (int(x_left), int(y_left)), (int(x_right), int(y_right)),
        #                     (255, 0, 0),3)
        # 좌표저장
        box_ = list([x_left, y_left, x_right, y_right])
        box_pred.append(box_)
    # plt.imshow(img)
    # plt.show()
    if orig_box :
        return box_pred, boxes
    else :
        return box_pred


def face_detect(img: np.ndarray, mtcnn) -> np.ndarray:
    """
    예측한 바운딩 박스 정보(x_좌상단,y_좌상단,x_우하단,y_우하단)와 confidence_score를 반환한다.
    """
    boxes, confidence_score = mtcnn.detect(img)
    box_pred = []
    if boxes is None:
        return box_pred
    for box, conf in zip(boxes,confidence_score):
        x_left = min(box[0], box[2])
        x_right = max(box[0], box[2])
        y_left = min(box[1], box[3])
        y_right = max(box[1], box[3])
        box_area = abs(x_right - x_left) * (y_right - y_left)
        # print(box_area)

        # 사진에 바운딩 박스 그리기
        # img = cv2.rectangle(img, (int(x_left), int(y_left)), (int(x_right), int(y_right)),
        #                     (255, 0, 0),3)
        # 좌표저장
        box_ = list([x_left, y_left, x_right, y_right, conf])
        box_pred.append(box_)
    # plt.imshow(img)
    # plt.show()
    return box_pred


def rescale(orig_image, target_size:int, box_label:list, mtcnn=False,resize_img=False):
    '''
        orig_image : 원본이미지
        target_size : resize할 이미지 크기(예 : 224 -> 224x224)
        box_label : detect한 bbox 리스트
        mtcnn으로 얻은 bbox를 rescale할 때 mtcnn=True
    '''
    x_orig, y_orig = orig_image.size
    x_scale = 224 / x_orig
    y_scale = 224 / y_orig
    
    image = orig_image.resize((target_size,target_size))
    image = np.array(image)

    orig_image = np.array(orig_image)

    new_box_label=[]
    # for box in box_label:
    for box in box_label: 
        orig_x_left = box[0]
        orig_y_left = box[1]
        orig_x_right = box[2]
        orig_y_right = box[3]

        conf = box[4] #conf진행할 때

        re_x_left = int(np.round(orig_x_left*x_scale))
        re_y_left = int(np.round(orig_y_left*y_scale))
        re_x_right = int(np.round(orig_x_right*x_scale))
        re_y_right = int(np.round(orig_y_right*y_scale))
        if mtcnn:
            box_area = (re_x_right-re_x_left)*(re_y_right-re_y_left)
            if box_area<200:
                continue
            # box_ = list([re_x_left,re_y_left,re_x_right,re_y_right])
            box_ = list([re_x_left,re_y_left,re_x_right,re_y_right,conf])
            new_box_label.append(box_)
            # 사진에 pred 바운딩 박스 그리기
            # image = cv2.rectangle(image, (re_x_left,re_y_left), (re_x_right,re_y_right),
            #                     (255, 0, 0),1)
        else :
            box_ = list([re_x_left,re_y_left,re_x_right,re_y_right])
            new_box_label.append(box_)
            # 사진에 gt 바운딩 박스 그리기
            # image = cv2.rectangle(image, (re_x_left,re_y_left), (re_x_right,re_y_right),
            #                     (0, 0, 255),1)
    # plt.imshow(image)
    # plt.show()
    if resize_img :
        return new_box_label, image.resize((target_size,target_size))
    else :
        return new_box_label


def calc_iou(gt_bbox, pred_bbox):
    """
    This function takes the predicted bounding box and ground truth bounding box and
    return the IoU ratio
    """
    x_topleft_gt, y_topleft_gt, x_bottomright_gt, y_bottomright_gt = gt_bbox
    x_topleft_p, y_topleft_p, x_bottomright_p, y_bottomright_p = pred_bbox

    if (x_topleft_gt > x_bottomright_gt) or (y_topleft_gt > y_bottomright_gt):
        raise AssertionError("Ground Truth Bounding Box is not correct")
    if (x_topleft_p > x_bottomright_p) or (y_topleft_p > y_bottomright_p):
        raise AssertionError(
            "Predicted Bounding Box is not correct",
            x_topleft_p,
            x_bottomright_p,
            y_topleft_p,
            y_bottomright_gt,
        )

    # if the GT bbox and predcited BBox do not overlap then iou=0
    if x_bottomright_gt < x_topleft_p:
        # If bottom right of x-coordinate  GT  bbox is less than or above the top left of x coordinate of  the predicted BBox
        return 0.0
    if (
        y_bottomright_gt < y_topleft_p
    ):  # If bottom right of y-coordinate  GT  bbox is less than or above the top left of y coordinate of  the predicted BBox
        return 0.0
    if (
        x_topleft_gt > x_bottomright_p
    ):  # If bottom right of x-coordinate  GT  bbox is greater than or below the bottom right  of x coordinate of  the predcited BBox
        return 0.0
    if (
        y_topleft_gt > y_bottomright_p
    ):  # If bottom right of y-coordinate  GT  bbox is greater than or below the bottom right  of y coordinate of  the predcited BBox
        return 0.0

    GT_bbox_area = (x_bottomright_gt - x_topleft_gt + 1) * (
        y_bottomright_gt - y_topleft_gt + 1
    )
    Pred_bbox_area = (x_bottomright_p - x_topleft_p + 1) * (
        y_bottomright_p - y_topleft_p + 1
    )

    x_top_left = np.max([x_topleft_gt, x_topleft_p])
    y_top_left = np.max([y_topleft_gt, y_topleft_p])
    x_bottom_right = np.min([x_bottomright_gt, x_bottomright_p])
    y_bottom_right = np.min([y_bottomright_gt, y_bottomright_p])

    intersection_area = (x_bottom_right - x_top_left + 1) * (
        y_bottom_right - y_top_left + 1
    )

    union_area = GT_bbox_area + Pred_bbox_area - intersection_area

    return intersection_area / union_area + 1e-6


def get_single_image_results(gt_boxes, pred_boxes, iou_thr):
    """Calculates number of true_pos, false_pos, false_neg from single batch of boxes.
    Args:
        gt_boxes (list of list of floats): list of locations of ground truth
            objects as [xmin, ymin, xmax, ymax]
        pred_boxes (dict): dict of dicts of 'boxes' (formatted like `gt_boxes`)
            and 'scores'
        iou_thr (float): value of IoU to consider as threshold for a
            true prediction.
    Returns:
        dict: true positives (int), false positives (int), false negatives (int)
    """
    all_pred_indices = range(len(pred_boxes))
    all_gt_indices = range(len(gt_boxes))

    if len(all_pred_indices) == 0:
        tp = 0
        fp = 0
        fn = len(gt_boxes)
        iou = 0
        # return {'true_positive':tp, 'false_positive':fp, 'false_negative':fn}
        return tp, fp, fn, iou
    if len(all_gt_indices) == 0:
        tp = 0
        fp = 0
        fn = 0
        iou = 0
        # return {'true_positive':tp, 'false_positive':fp, 'false_negative':fn}
        return tp, fp, fn, iou
    gt_idx_thr = []
    pred_idx_thr = []
    ious = []
    for ipb, pred_box in enumerate(pred_boxes):
        for igb, gt_box in enumerate(gt_boxes):
            iou = calc_iou(gt_box, pred_box)

            if iou > iou_thr:
                gt_idx_thr.append(igb)
                pred_idx_thr.append(ipb)
                ious.append(iou)
    iou_sort = np.argsort(ious)[::1]

    # print(gt_idx_thr)
    # print(pred_idx_thr)
    if len(iou_sort) == 0:
        tp = 0
        fp = 1
        fn = 1
        # iou=[max(ious_)]
        iou = 0
        # return {'true_positive':tp, 'false_positive':fp, 'false_negative':fn}
        return tp, fp, fn, iou
    else:
        gt_match_idx = []
        pred_match_idx = []
        for idx in iou_sort:
            gt_idx = gt_idx_thr[idx]
            pr_idx = pred_idx_thr[idx]
            # If the boxes are unmatched, add them to matches
            if (gt_idx not in gt_match_idx) and (pr_idx not in pred_match_idx):
                gt_match_idx.append(gt_idx)
                pred_match_idx.append(pr_idx)
        tp = len(gt_match_idx)
        fp = len(pred_boxes) - len(pred_match_idx)
        fn = len(gt_boxes) - len(gt_match_idx)

    return tp, fp, fn, ious


def mtcnn_evaluate(mtcnn, file_dict, bbox_dict, target_size, iou_thr):
    total_tp = 0
    total_fp = 0
    total_fn = 0
    iou_list = []

    for i in range(len(bbox_dict.keys())):
        sample_file_key = list(bbox_dict.keys())[i]
        img_dir = file_dict[sample_file_key]["image"]
        # print(img_dir)
        orig_image = Image.open(img_dir)
        image = np.array(orig_image)

        # 바운딩 박스 확인 -> gt_detect(..,img) img추가해야함
        # tp, fp, fn, ious = get_single_image_results(gt_detect(sample_bbox_dict[sample_file_key],img),mtcnn_detect(img),iou_thr=0.5)
        gt_box = gt_detect(bbox_dict[sample_file_key])
        pred_box = mtcnn_detect(image,mtcnn)
        tp, fp, fn, ious = get_single_image_results(
            rescale(orig_image,target_size,gt_box), 
            rescale(orig_image,target_size,pred_box,mtcnn=True),
            iou_thr
        )
        # print(ious)
        if ious != 0:
            iou_list.extend(ious)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        print(i)
        # print({'true_positive': tp, 'false_positive': fp, 'false_negative': fn})

        # 바운딩 박스 확인하기(GT : blue box, pred : red box)
        # plt.imshow(img)
        # plt.show()
    precision = total_tp / (total_tp + total_fp)
    recall = total_tp / (total_tp + total_fn)
    f1_score = 2.0 / (1 / precision + 1 / recall)
    miou = sum(iou_list) / len(iou_list)
    print(
        {
            "true_positive": total_tp,
            "false_positive": total_fp,
            "false_negative": total_fn,
        }
    )
    print(
        {"precision": precision, "recall": recall, "F1_score": f1_score, "mIoU": miou}
    )
