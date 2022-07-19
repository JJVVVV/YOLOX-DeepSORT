import argparse
import os
import sys

import torch
from loguru import logger

from YOLOX.yolox.exp import get_exp
from YOLOX.yolox.utils import fuse_model, get_model_info

sys.path.append(sys.path[0] + '\\YOLOX')

detector_name = 'yolox-s'  # eg. yolox-s, yolox-m
path = "YOLOX/videos/MOT20-03.mp4"


raw = False
images_path = "YOLOX/videos/MOT20-03/img1/"
num_images = len(os.listdir(images_path))
det_path = "YOLOX/videos/MOT20-03/det/det.txt"


#      MOT20-1.mp4          MOT20-02-tiny.mp4    MOT20-3-tiny.mp4       MOT20-05-1-1500.mp4
# initialize the detector
parser = argparse.ArgumentParser()


def get_parser_detector():
    parser.add_argument(
        "--source",
        default="video",
        help="source type, eg. video, webcam"
    )
    parser.add_argument(
        "-expn", "--experiment-name",
        default=exp.exp_name)
    parser.add_argument(
        "--path",
        default=path,
        help="path to images or video")
    # parser_detector.add_argument("--camid", type=int, default=0, help="webcam id")
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )
    parser.add_argument(
        "-c", "--ckpt",
        default='YOLOX/yolox_weights/yolox_s.pth',
        type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=0.25, type=float, help="conf threshold")
    parser.add_argument("--nms", default=0.45, type=float, help="nms threshold")
    parser.add_argument("--size", default=640, type=int, help="the size of model's input")

    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="pls input your experiment description file",
    )
    # set True to complicate old version
    parser.add_argument(
        "--legacy",
        dest="legacy",
        default=False,
        action="store_true",
        help="To be compatible with older versions",
    )
    return parser


# initialize the tracker
def get_parse_tracker():
    parser.add_argument("--VIDEO_PATH", type=str, default=path)
    if raw:
        parser.add_argument("--config_deepsort", type=str, default="./configs/DeepSORT_raw.yaml")
    else:
        parser.add_argument("--config_deepsort", type=str, default="./configs/DeepSORT.yaml")
    parser.add_argument("--config_fastreid", type=str, default="./configs/fastreid.yaml")
    parser.add_argument("--fastreid", action="store_true")
    # parser.add_argument("--ignore_display", dest="display", action="store_false", default=True)
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--display_width", type=int, default=640)
    parser.add_argument("--display_height", type=int, default=480)
    parser.add_argument("--save_path", type=str, default='./DeepSORT_outputs')
    parser.add_argument("--cpu", dest="use_cuda", action="store_false")
    parser.add_argument("--camera", action="store", dest="cam", type=int, default="-1")
    return parser


# get the exp(detector) by model name. eg. yolo-s, yolo-m
exp = get_exp(exp_name=detector_name)

# get all parameters of detector
get_parser_detector()
get_parse_tracker()
args = parser.parse_args()

# create the save folder
file_name = os.path.join(exp.output_dir, args.experiment_name)
os.makedirs(file_name, exist_ok=True)
vis_folder = None
if args.save_result:
    vis_folder = os.path.join(file_name, "vis_res")
    os.makedirs(vis_folder, exist_ok=True)

logger.info("Args: {}".format(args))

if args.conf is not None:
    exp.test_conf = args.conf
if args.nms is not None:
    exp.nmsthre = args.nms
if args.size is not None:
    exp.test_size = (args.size, args.size)

# trt must on gpu
if args.trt:
    args.device = "gpu"


def get_model():
    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    if args.device == "gpu":
        model.cuda()
        if args.fp16:
            model.half()  # to FP16
    model.eval()
    return model


def load_weight(model):
    if not args.trt:
        if args.ckpt is None:
            ckpt_file = os.path.join(file_name, "best_ckpt.pth")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")


# trt and fuse
def set_model(model):
    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)
    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None
    return trt_file, decoder


model_detector = get_model()
load_weight(model_detector)
trt_file, decoder = set_model(model_detector)


###############################################################################################


