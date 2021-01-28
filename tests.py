import torch
import numpy as np
from d2dev._C import  batch_nms, retinanet_decode, retinaplate_decode


import pickle

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
    
    with open('/d2dev/pred_logits.bin', 'rb') as f:
        cls_head = pickle.load(f)
        
    with open('/d2dev/pred_anchor_deltas.bin', 'rb') as f:
        box_head = pickle.load(f)
    
    with open('/d2dev/anchor.bin', 'rb') as f:
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
    
    keep_scores, keep_bboxes, keep_classe = batch_nms(scores, boxes, classes, 0.5, 10)

    print('keep')
    print(keep_scores)
    print(keep_bboxes)
    print(keep_classe)


def test_retinaplate_decode():

    with open('/d2dev/pred_logits.bin', 'rb') as f:
        cls_head = pickle.load(f)
        
    with open('/d2dev/pred_anchor_deltas.bin', 'rb') as f:
        box_head = pickle.load(f)
    
    with open('/d2dev/pred_keypoint_deltas.bin', 'rb') as f:
        oks_head = pickle.load(f)
    
    with open('/d2dev/anchor.bin', 'rb') as f:
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
    
    # keep_scores, keep_bboxes, keep_classe = batch_nms(scores, boxes, classes, 0.5, 10)

    # print('keep')
    # print(keep_scores)
    # print(keep_bboxes)
    # print(keep_classe)


if __name__ == '__main__':
    # test_nms()
    # test_retinanet_decode()
    test_retinaplate_decode()