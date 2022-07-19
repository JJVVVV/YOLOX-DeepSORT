import os
import time
import warnings

import cv2
import torch
from loguru import logger

from DeepSORT import build_tracker
from detector import predictor
from initializer import args
from utils.draw import draw_boxes
from utils.io import write_results
from utils.log import get_logger
from utils.parser import get_config


class VideoTracker(object):
    def __init__(self, cfg, args, video_path):
        self.cfg = cfg
        self.args = args
        self.video_path = video_path
        self.logger = get_logger("root")
        self.fps = None
        use_cuda = args.use_cuda and torch.cuda.is_available()
        if not use_cuda:
            warnings.warn("Running in cpu mode which maybe very slow!", UserWarning)

        if args.display:
            cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("test", args.display_width, args.display_height)

        if args.cam != -1:
            print("Using webcam " + str(args.cam))
            self.vdo = cv2.VideoCapture(args.cam)
        else:
            self.vdo = cv2.VideoCapture()
        self.detector = predictor
        if self.args.cam != -1:
            ret, frame = self.vdo.read()
            assert ret, "Error: Camera error"
            self.im_width = frame.shape[1]
            self.im_height = frame.shape[0]
            self.fps = self.vdo.get(cv2.CAP_PROP_FPS)

        else:
            assert os.path.isfile(self.video_path), "Path error"
            self.vdo.open(self.video_path)
            self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = self.vdo.get(cv2.CAP_PROP_FPS)
            assert self.vdo.isOpened()
        self.deepsort = build_tracker(cfg, use_cuda=use_cuda, w=self.im_width, h=self.im_height)

    def __enter__(self):
        if self.args.save_result:
            name = self.video_path.split('/')[-1][:-4] if self.args.cam == -1 else 'webcam'
            current_time = time.localtime()
            self.save_folder = os.path.join(
                self.args.save_path, time.strftime("%Y_%m_%d_%H_%M_%S", current_time) + '-' + name
            )
            os.makedirs(self.save_folder, exist_ok=True)
            self.save_images_path = os.path.join(self.save_folder, 'images')
            os.makedirs(self.save_images_path, exist_ok=True)
            # path of saved video and results
            self.save_video_path = os.path.join(self.save_folder, "results.mp4")
            self.save_results_path = os.path.join(self.save_folder, "results.txt")

            # create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(self.save_video_path, fourcc, self.fps if self.fps is not None else 20,
                                          (self.im_width, self.im_height))

            # logging
            # self.logger.info("Save results to {}".format(save_folder))
            logger.info("Save results to {}".format(self.save_folder))
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)

    def run(self):
        results = []
        idx_frame = 0
        sumfps = 0
        while self.vdo.grab():
            idx_frame += 1
            if idx_frame % self.args.frame_interval:
                continue

            start = time.time()
            ok, raw_img_bgr = self.vdo.retrieve()
            if not ok:
                continue
            raw_img_rgb = cv2.cvtColor(raw_img_bgr, cv2.COLOR_BGR2RGB)
            # do detection
            outputs, img_info = self.detector.inference(raw_img_bgr, toxyxy=False)
            bboxes_xywh = cls_conf = None
            if outputs[0] is not None:
                outputs = outputs[0].cpu()
                # outputs = outputs[0]
                bboxes_xywh = outputs[:, :4]  # 缩放图上的bbox
                bboxes_xywh = bboxes_xywh / img_info["ratio"]  # 原图上的bbox
                cls_conf = outputs[:, 4] * outputs[:, 5]  # final score

            # # select person class
            # mask = cls_ids == 0
            #
            # bbox_xywh = bbox_xywh[mask]
            # # bbox dilation just in case bbox too small, delete this line if using a better pedestrian detector
            # bbox_xywh[:, 3:] *= 1.2
            # cls_conf = cls_conf[mask]

            # do tracking
            outputs = self.deepsort.update(bboxes_xywh, cls_conf, raw_img_rgb)

            # draw boxes for visualization
            if len(outputs) > 0:
                bbox_xyxy = outputs[:, :4]
                identities = outputs[:, -1]
                raw_img_bgr = draw_boxes(raw_img_bgr, bbox_xyxy, identities)
                bbox_tlwh = []
                for bb_xyxy in bbox_xyxy:
                    bbox_tlwh.append(self.deepsort._xyxy_to_tlwh(bb_xyxy))
                results.append((idx_frame, bbox_tlwh, identities))

            end = time.time()

            if self.args.display:
                cv2.imshow("test", raw_img_bgr)
                cv2.waitKey(1)

            if self.args.save_result:
                self.writer.write(raw_img_bgr)

                # cv2.imwrite(os.path.join(self.save_images_path, str(idx_frame)+'.jpg'), raw_img_bgr)

                # save results
                write_results(self.save_results_path, results, 'mot')

            # logging
            # self.logger.info("time: {:.03f}s, fps: {:.03f}, detection numbers: {}, tracking numbers: {}" \
            #                  .format(end - start, 1 / (end - start), bbox_xywh.shape[0], len(outputs)))
            logger.info("\nNO.{:4d}\ntime of total per frame: {:.04f}s\nfps: {:.03f}\ndetection numbers: {}, "
                        "tracking numbers: {}" \
                        .format(idx_frame, end - start,
                                1 / (end - start),
                                bboxes_xywh.shape[0] if bboxes_xywh is not None else 0,
                                len(outputs)))

            sumfps += 1 / (end - start)
            # 退出
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        logger.info(f"average fps: {sumfps / (idx_frame - 1): #.4f}")


if __name__ == "__main__":
    args = args
    cfg = get_config()
    cfg.merge_from_file(args.config_deepsort)
    if args.fastreid:
        cfg.merge_from_file(args.config_fastreid)
        cfg.USE_FASTREID = True
    else:
        cfg.USE_FASTREID = False

    with VideoTracker(cfg, args, video_path=args.VIDEO_PATH) as vdo_trk:
        vdo_trk.run()
