import cv2
# from cv2 import cv2
import os
import time
from loguru import logger
IMAGE_EXT = {".jpg", ".jpeg", ".webp", ".bmp", ".png"}

root = r'./videos/MOT20-01/'
start = 0
end = 9999

def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names

def imageflow2mp4(vis_folder):
    path = root+'img1'
    files = get_image_list(path)
    files.sort()
    image_ori = cv2.imread(files[0])
    video_size = (image_ori.shape[1], image_ori.shape[0])
    fps = 25
    # save_folder = os.path.join(
    #     vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S",)
    # )
    save_folder = os.path.join(
        vis_folder, '',
    )
    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, root.split("/")[-2]+'.mp4',)
    print(save_path)

    logger.info(f"video save_path is {save_path}")
    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, video_size
    )

    for img in files[start:end]:
        frame = cv2.imread(img)
        vid_writer.write(frame)
    vid_writer.release()



# imageflow2mp4('videos')


def get_GT():
    ret = []
    path = root+'gt'
    with open(os.path.join(path, 'gt.txt'), 'r') as f:
        lines = f.readlines()
        for line in lines:
            if start<=int(line.split(',')[0])<end:
                words = line.split(',')
                words[0] = str(int(words[0])-start+1)
                ret.append(','.join(words))
    with open(os.path.join(path, 'new_gt.txt'), 'w') as f:
        f.writelines(ret)



root = r'./videos/MOT20-01/'

def get_GT():
    ret = []
    path = root+'gt'
    with open(os.path.join(path, 'gt.txt'), 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.split(',')[-3]=='1':
                ret.append(line)
    with open(os.path.join(path, 'new_gt.txt'), 'w') as f:
        f.writelines(ret)


get_GT()