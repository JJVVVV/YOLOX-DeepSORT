import time

import cv2
import numpy as np
import torch
from loguru import logger
from numpy import ndarray
from torch import Tensor
from initializer import raw

from .deep.feature_extractor import Extractor, FastReIDExtractor
from .sort.nn_matching import NearestNeighborDistanceMetric
from .sort.preprocessing import non_max_suppression
from .sort.detection import Detection
from .sort.tracker import Tracker
from .sort.id_map import IDmap

__all__ = ['DeepSort']


class DeepSort(object):
    def __init__(self, model_path, model_config=None, max_cosine_distance=0.2, min_confidence=0.3, nms_max_overlap=1.0,
                 max_iou_distance=0.7, max_age=70, n_init=3, nn_budget=100, use_cuda=True, w=None, h=None):
        # self.min_confidence = min_confidence
        self.width = w
        self.height = h
        self.nms_max_overlap = nms_max_overlap

        if model_config is None:
            self.extractor = Extractor(model_path, use_cuda=use_cuda)
        else:
            self.extractor = FastReIDExtractor(model_config, model_path, use_cuda=use_cuda)

        metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        mymap = IDmap(self.width, self.height)
        self.tracker = Tracker(metric, max_iou_distance=max_iou_distance, max_age=max_age, n_init=n_init, IDmap=mymap)

    def update(self, bboxes_xywh: Tensor = None, confidences: Tensor = None, raw_img_rgb: ndarray = None):
        # self.height, self.width = raw_img_rgb.shape[:2]
        # generate detections
        detections = list()
        if bboxes_xywh is not None:
            features, mask = self._get_features(bboxes_xywh, raw_img_rgb)
            bboxes_xywh = bboxes_xywh[mask]
            confidences = confidences[mask]
            bboxes_tlwh = self._xywh_to_tlwh(bboxes_xywh)
            detections = [Detection(bboxes_tlwh[i], conf, features[i]) for i, conf in enumerate(confidences)]

        # # run on non-maximum supression
        # boxes = np.array([d.tlwh for d in detections])
        # scores = np.array([d.confidence for d in detections])
        # indices = non_max_suppression(boxes, self.nms_max_overlap, scores)
        # detections = [detections[i] for i in indices]

        # update tracker
        # start = time.time()
        self.tracker.predict()
        self.tracker.update(detections)
        # logger.info("DeepSORT time: {:.4f}s".format(time.time() - start))
        # output bbox identities
        outputs = []
        for track in self.tracker.tracks:
            # if not track.is_confirmed() or track.time_since_update > 1:
            #     continue
            # track.track_id in {54, 67}
            # or track.times_of_map_predict < 5

            if track.is_confirmed() and (track.time_since_update <= 1 or \
                                         (not raw and track.times_of_map_predict < 10 and track.time_since_map_update==0)):
            # if track.is_confirmed() and track.time_since_update <= 1:
                box = track.to_tlwh()
                x1, y1, x2, y2 = self._tlwh_to_xyxy(box)
                track_id = track.track_id
                outputs.append(np.array([x1, y1, x2, y2, track_id], dtype=np.int))
        if len(outputs) > 0:
            outputs = np.stack(outputs, axis=0)
        return outputs

    """
    TODO:
        Convert bbox from xc_yc_w_h to xtl_ytl_w_h
    Thanks JieChen91@github.com for reporting this bug!
    """

    @staticmethod
    def _xywh_to_tlwh(bboxes_xywh: Tensor) -> Tensor:
        if isinstance(bboxes_xywh, np.ndarray):
            bbox_tlwh = bboxes_xywh.copy()
        elif isinstance(bboxes_xywh, torch.Tensor):
            bbox_tlwh = bboxes_xywh.clone()
            # bbox_tlwh = bboxes_xywh.new(bboxes_xywh.shape)
        bbox_tlwh[:, 0] = bboxes_xywh[:, 0] - bboxes_xywh[:, 2] / 2.
        bbox_tlwh[:, 1] = bboxes_xywh[:, 1] - bboxes_xywh[:, 3] / 2.
        return bbox_tlwh

    def _xywh_to_xyxy(self, bboxes_xywh):
        boxes_corner = bboxes_xywh.new(bboxes_xywh.shape)
        boxes_corner[:, 0] = bboxes_xywh[:, 0] - bboxes_xywh[:, 2] / 2
        boxes_corner[:, 1] = bboxes_xywh[:, 1] - bboxes_xywh[:, 3] / 2
        boxes_corner[:, 2] = bboxes_xywh[:, 0] + bboxes_xywh[:, 2] / 2
        boxes_corner[:, 3] = bboxes_xywh[:, 1] + bboxes_xywh[:, 3] / 2
        return boxes_corner
        # x,y,w,h = bbox_xywh
        # x1 = max(int(x-w/2),0)
        # x2 = min(int(x+w/2),self.width-1)
        # y1 = max(int(y-h/2),0)
        # y2 = min(int(y+h/2),self.height-1)
        # return x1,y1,x2,y2

    def _tlwh_to_xyxy(self, bbox_tlwh):
        """
        TODO:
            Convert bbox from xtl_ytl_w_h to xc_yc_w_h
        Thanks JieChen91@github.com for reporting this bug!
        """
        x, y, w, h = bbox_tlwh
        x1 = max(int(x), 0)
        x2 = min(int(x + w), self.width - 1)
        y1 = max(int(y), 0)
        y2 = min(int(y + h), self.height - 1)
        return x1, y1, x2, y2

    def _xyxy_to_tlwh(self, bbox_xyxy):
        x1, y1, x2, y2 = bbox_xyxy

        t = x1
        l = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        return t, l, w, h

    def _get_features(self, bboxes_xywh: Tensor, raw_img_rgb: ndarray) -> tuple[ndarray, list]:
        bboxes_xyxy = self._xywh_to_xyxy(bboxes_xywh)
        bboxes_xyxy = bboxes_xyxy.int()
        # ori_img = torch.from_numpy(ori_img)
        images = list()
        mask = []
        for box in bboxes_xyxy:
            x1, y1, x2, y2 = box
            image = raw_img_rgb[y1:y2, x1:x2]
            if image.size > 0:
                images.append(image)
                mask.append(True)
                # cv2.imshow('a', image)
                # cv2.waitKey(0)
            else:
                mask.append(False)
        # start = time.time()
        if images:
            features = self.extractor(images)
        else:
            features = np.array([])
        # logger.info("extract features time: {:.4f}s".format(time.time() - start))
        return features, mask

        # im_crops = []
        # for box in bboxes_xywh:
        #     x1,y1,x2,y2 = self._xywh_to_xyxy(box)
        #     im = ori_img[y1:y2,x1:x2]
        #     im_crops.append(im)
        # if im_crops:
        #     features = self.extractor(im_crops)
        # else:
        #     features = np.array([])
        # return features
