# import numpy as np
# import cv2
# cap = cv2.VideoCapture("videos/MOT20-07.mp4")
#
#
# while cap.grab():
#     _, frame = cap.retrieve()
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     delay = int(1 / fps * 1000)
#     # frame = cv2.cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     cv2.imshow('frame', frame)
#     ch = cv2.waitKey(delay)
#     if ch & 0xFF == ord('q') or ch==27:
#         break
#
# # while(True):
# # # Capture frame-by-frame
# #     ret, frame = cap.read()
# #     fps = cap.get(cv2.CAP_PROP_FPS)
# #     delay = int(1/fps*1000)
# # # Our operations on the frame come here
# # #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# # # Display the resulting frame
# #     frame = cv2.cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# #     cv2.imshow('frame', frame)
# #     if cv2.waitKey(delay) & 0xFF == ord('q'):
# #         break
# # When everything done, release the capture
# cap.release()
# cv2.destroyAllWindows()

ts_file = "DeepSORT_outputs/2022_05_07_13_53_27-MOT20-01/MOT20-01.txt"
s = set()
with open(ts_file, 'r') as f:
    lines = f.readlines()
    for line in lines:
        s.add(line.split(',')[1])
print(len(s))
print(max(map(int, s)))
print(sorted(map(int, s)))