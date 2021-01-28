import cv2
import torch
import numpy as np
from d2dev._C import  batch_nms, retinanet_decode, retinaplate_decode, plate_batch_nms


import pickle


def CenterPadResize(image, bchw_shape, border_value):
    src_h, src_w = image.shape[:2]
    _, _, dst_h, dst_w = bchw_shape
    ratio = min(float(dst_h) / src_h, float(dst_w) / src_w)
    new_size = (round(src_w * ratio), round(src_h * ratio))
    dw = (dst_w - new_size[0]) / 2
    dh = (dst_h - new_size[1]) / 2
    t, b = round(dh - 0.1), round(dh + 0.1)
    l, r = round(dw - 0.1), round(dw + 0.1)
    image = cv2.resize(image, new_size, interpolation=0)
    image = cv2.copyMakeBorder(image, t, b, l, r, cv2.BORDER_CONSTANT, value=border_value)
    return image


def test_nms():
    scores = torch.Tensor([[0.5, 0.9, 0.1]]).cuda()
    bboxes = torch.Tensor([[[0.0, 0.0, 10, 10], [2.0, 2.0, 9.0, 9.0], [2.0, 2.0, 9.0, 9.0]]]).cuda()
    classe = torch.Tensor([[0, 0, 1]]).cuda()
    print('origin')
    print(scores.shape)
    print(bboxes.shape)
    print(classe.shape)

    keep_scores, keep_bboxes, keep_classe = batch_nms(scores, bboxes, classe, 0.5, 10)
    print('keep')
    print(keep_scores)
    print(keep_bboxes)
    print(keep_classe)


def test_retinanet_decode():
    import pickle
    
    with open('data/pred_logits.bin', 'rb') as f:
        cls_head = pickle.load(f)
        
    with open('data/pred_anchor_deltas.bin', 'rb') as f:
        box_head = pickle.load(f)
    
    with open('data/anchor.bin', 'rb') as f:
        anchor = pickle.load(f)

    socres_list, boxes_list, classes_list = [], [], []
    for i in range(len(cls_head)):
        cls = torch.from_numpy(cls_head[i]).cuda()
        box = torch.from_numpy(box_head[i]).cuda()
        anc = anchor[i].flatten().tolist()
        scores, boxes, classes = retinanet_decode(cls, box, anc, [10.0, 10.0, 10.0, 10.0], 0.02, 100)
        socres_list.append(scores)
        boxes_list.append(boxes)
        classes_list.append(classes)
        
    scores = torch.cat(socres_list, axis=1)
    boxes = torch.cat(boxes_list, axis=1)
    classes = torch.cat(classes_list, axis=1)
    print(scores)
    keep_scores, keep_bboxes, keep_classe = batch_nms(scores, boxes, classes, 0.5, 10)

    print('keep')
    print(keep_scores)
    print(keep_bboxes)
    print(keep_classe)


def test_retinaplate_decode():

    with open('data/pred_logits.bin', 'rb') as f:
        cls_head = pickle.load(f)
        
    with open('data/pred_anchor_deltas.bin', 'rb') as f:
        box_head = pickle.load(f)
    
    with open('data/pred_keypoint_deltas.bin', 'rb') as f:
        oks_head = pickle.load(f)
    
    with open('data/anchor.bin', 'rb') as f:
        anchor = pickle.load(f)

    socres_list, boxes_list, classes_list, keypoints_list = [], [], [], []
    for i in range(len(cls_head)):
        cls = torch.from_numpy(cls_head[i]).cuda()
        box = torch.from_numpy(box_head[i]).cuda()
        oks = torch.from_numpy(oks_head[i]).cuda()
        print(cls.shape, box.shape, oks.shape)
        anc = anchor[i].flatten().tolist()
        scores, boxes, classes, keypoints = retinaplate_decode(cls, box, oks, anc, [10.0, 10.0, 10.0, 10.0], [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0], 0.02, 100)
        socres_list.append(scores)
        boxes_list.append(boxes)
        classes_list.append(classes)
        keypoints_list.append(keypoints)
        
    scores = torch.cat(socres_list, axis=1)
    boxes = torch.cat(boxes_list, axis=1)
    classes = torch.cat(classes_list, axis=1)
    keypoints = torch.cat(keypoints_list, axis=1)
    
    keep_scores, keep_bboxes, keep_classe, keep_keypoints = plate_batch_nms(scores, boxes, classes, keypoints, 0.5, 10)

    print('keep')
    print(keep_scores.shape)
    print(keep_bboxes.shape)
    print(keep_classe.shape)
    print(keep_keypoints.shape)

    keep_scores = keep_scores.cpu().numpy()[0]
    keep_bboxes = keep_bboxes.cpu().numpy()[0]
    keep_classe = keep_classe.cpu().numpy()[0]
    keep_keypoints = keep_keypoints.cpu().numpy()[0]


    image = cv2.imread('data/1.jpg')
    image = CenterPadResize(image, (1,3,640,640), (127, 127, 127))
    for i in range(len(keep_scores)):
        sc = keep_scores[i]
        if sc < 0.9:
            continue
        x1, y1, x2, y2 = map(int, keep_bboxes[i])
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 0), 2)
        for oks in keep_keypoints[i].reshape((-4, 2)):
            k1, k2 = int(oks[0]), int(oks[1])
            cv2.circle(image, (k1, k2), 2, (0, 255, 0), -1)

    cv2.imwrite('data/2.jpg', image)


def test_engine():
    with open('data/anchor.bin', 'rb') as f:
        anchor = pickle.load(f)
    anchor = [x.tolist() for x in anchor]
    
    # e = Engine('/d2dev/model_0043999.onnx', 10, 1, 'FP16', 0.5, 100, anchor, 0.5, 10, [], 'model_name', '', True)


if __name__ == '__main__':
    # test_nms()
    # test_retinanet_decode()
    test_retinaplate_decode()
    # test_engine()