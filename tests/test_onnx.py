 
import os, sys
import onnxruntime
import onnx
import cv2
import torch
import numpy as np
import pickle

from d2dev._C import  retinaplate_decode, plate_batch_nms

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


class ONNXModel():
    def __init__(self, onnx_path):
        """
        :param onnx_path:
        """
        self.onnx_session = onnxruntime.InferenceSession(onnx_path)
        self.input_name = self.get_input_name(self.onnx_session)
        self.output_name = self.get_output_name(self.onnx_session)
        print("input_name:{}".format(self.input_name))
        print("output_name:{}".format(self.output_name))

    def get_output_name(self, onnx_session):
        """
        output_name = onnx_session.get_outputs()[0].name
        :param onnx_session:
        :return:
        """
        output_name = []
        for node in onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name

    def get_input_name(self, onnx_session):
        """
        input_name = onnx_session.get_inputs()[0].name
        :param onnx_session:
        :return:
        """
        input_name = []
        for node in onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name

    def get_input_feed(self, input_name, image_numpy):
        """
        input_feed={self.input_name: image_numpy}
        :param input_name:
        :param image_numpy:
        :return:
        """
        input_feed = {}
        for name in input_name:
            input_feed[name] = image_numpy
        return input_feed

    def forward(self, image_numpy):
        '''
        # image_numpy = image.transpose(2, 0, 1)
        # image_numpy = image_numpy[np.newaxis, :]
        # onnx_session.run([output_name], {input_name: x})
        # :param image_numpy:
        # :return:
        '''
        # 输入数据的类型必须与模型一致,以下三种写法都是可以的
        # scores, boxes = self.onnx_session.run(None, {self.input_name: image_numpy})
        # scores, boxes = self.onnx_session.run(self.output_name, input_feed={self.input_name: iimage_numpy})
        input_feed = self.get_input_feed(self.input_name, image_numpy)
        result = self.onnx_session.run(self.output_name, input_feed=input_feed)
        # return {name: result[i] for i, name in enumerate(self.output_name)}
        return result

    
if __name__ == '__main__':
    onnx_path = '../../model_0106999.onnx'
    net = ONNXModel(onnx_path)

    img = cv2.imread('../data/2.jpg')
    img = CenterPadResize(img, (1, 3, 640, 640), (127, 127, 127))

    imgt = img.transpose(2, 0, 1)[np.newaxis, ...]
    imgt = np.float32(imgt)
    print(imgt.shape)

    out = net.forward(imgt)

    with open('../data/anchor.bin', 'rb') as f:
        anchor = pickle.load(f)

    cls_head, box_head, oks_head = out[:5], out[5:10], out[10:15]

    socres_list, boxes_list, classes_list, keypoints_list = [], [], [], []
    for i in range(len(cls_head)):
        cls = torch.from_numpy(cls_head[i]).cuda()
        box = torch.from_numpy(box_head[i]).cuda()
        oks = torch.from_numpy(oks_head[i]).cuda()
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

    keep_scores, keep_bboxes, keep_classe, keep_keypoints = plate_batch_nms(scores, boxes, classes, keypoints, 0.3, 100)

    print('keep')
    print(keep_scores.shape)
    print(keep_bboxes.shape)
    print(keep_classe.shape)
    print(keep_keypoints.shape)

    keep_scores = keep_scores.cpu().numpy()[0]
    keep_bboxes = keep_bboxes.cpu().numpy()[0]
    keep_classe = keep_classe.cpu().numpy()[0]
    keep_keypoints = keep_keypoints.cpu().numpy()[0]

    for i in range(len(keep_scores)):
        sc = keep_scores[i]
        if sc < 0.9:
            continue
        x1, y1, x2, y2 = map(int, keep_bboxes[i])
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 2)
        for oks in keep_keypoints[i].reshape((-4, 2)):
            k1, k2 = int(oks[0]), int(oks[1])
            cv2.circle(img, (k1, k2), 2, (0, 255, 0), -1)

    cv2.imwrite('../data/2_2.jpg', img)