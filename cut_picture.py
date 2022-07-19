import cv2
import os

path = "DeepSORT_outputs/2022_04_26_19_56_30_raw/images"
# path = "DeepSORT_outputs/2022_04_27_11_41_11_my_without_mashi/images"
save_path = 'cut'
# for i in range(125, 155):
#     img = cv2.imread(os.path.join(path, str(i) + '.jpg'))
#     # print(img.shape)
#     cv2.imwrite(os.path.join(save_path, str(i) + '.jpg'), img[500:, 1500:, :])
for i in range(345, 375):
    img = cv2.imread(os.path.join(path, str(i) + '.jpg'))
    # print(img.shape)
    cv2.imwrite(os.path.join(save_path, str(i) + '.jpg'), img[500:, 1250:1700, :])