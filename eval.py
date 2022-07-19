import motmetrics as mm

# 1
# gt_file = "YOLOX/videos/MOT20-03/gt/gt.txt"
# raw_file = "DeepSORT_outputs/deepsort2022_05_05_14_20_17-MOT20-03(70-3)/MOT20-03.txt"
# ts_file = "DeepSORT_outputs/2022_05_06_11_00_37-MOT20-03/MOT20-03.txt"
gt_file = "YOLOX/videos/MOT20-01/gt/gt.txt"
raw_file = "DeepSORT_outputs/Deepsort-MOT20-01/MOT20-01.txt"
ts_file = "DeepSORT_outputs/2022_05_09_15_43_50-MOT20-01/MOT20-01.txt"


# # 2
# gt_file = "YOLOX/videos/MOT20-02/gt/new_gt.txt"
# raw_file = "DeepSORT_outputs/raw_MOT20-02-tiny/results.txt"
# ts_file = "DeepSORT_outputs/my_MOT20-02-tiny/results.txt"
# #
#
# # # 3
# gt_file = "YOLOX/videos/MOT20-03/gt/new_gt.txt"
# raw_file = "DeepSORT_outputs/raw_MOT20-3-tiny/results.txt"
# ts_file = "DeepSORT_outputs/my_MOT20-3-tiny/results.txt"

# 4
# gt_file = "YOLOX/videos/MOT20-05/gt/new_gt.txt"
# raw_file = "DeepSORT_outputs/raw_MOT20-05-tiny/results.txt"
# ts_file = "DeepSORT_outputs/my_MOT20-05-tiny/results.txt"

# print(list(mm.io.Format))
gt = mm.io.loadtxt(gt_file, fmt="mot16", min_confidence=1)  # 读入GT
ts = mm.io.loadtxt(ts_file, fmt="mot16")  # 读入自己生成的跟踪结果
raw = mm.io.loadtxt(raw_file, fmt="mot16")

acc=mm.utils.compare_to_groundtruth(gt, ts, 'iou', distth=0.5)  # 根据GT和自己的结果，生成accumulator，distth是距离阈值
acc_raw=mm.utils.compare_to_groundtruth(gt, raw, 'iou', distth=0.5)


mh = mm.metrics.create()

# # 打印单个accumulator
# summary = mh.compute(acc,
#                      metrics=['num_frames', 'mota', 'motp'], # 一个list，里面装的是想打印的一些度量
#                      name='acc') # 起个名
# print(summary)

# # 打印多个accumulators
# summary = mh.compute_many([acc, acc.events.loc[0:1]], # 多个accumulators组成的list
#                           metrics=['num_frames', 'mota', 'motp', 'num_switches'],
#                           names=['full', 'part']) # 起个名
# print(summary)

#
# # 自定义显示格式
# strsummary = mm.io.render_summary(
#     summary,
#     formatters={'mota' : '{:.2%}'.format},  # 将MOTA的格式改为百分数显示
#     namemap={'mota': 'MOTA', 'motp' : 'MOTP'}  # 将列名改为大写
# )
# print(strsummary)



metrics = list(mm.metrics.motchallenge_metrics)  # 即支持的所有metrics的名字列表
# print(metrics)


# mh模块中有内置的显示格式
summary = mh.compute_many([acc_raw, acc],
                          metrics=metrics,
                          names=['raw', 'my'])

strsummary = mm.io.render_summary(
    summary,
    formatters=mh.formatters,
    namemap=mm.io.motchallenge_metric_names
)
print(strsummary)
