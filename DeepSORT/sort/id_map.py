import numpy as np
from .track import Track
from . import kalman_filter
from collections import Counter

alpha = 0.05

class IDmap:
    def __init__(self, w, h):
        self.map = np.zeros((h, w, 3))
        self.hashmap = dict()
        self.matched_tracks_idx = set()
        self.w = w * 1.0
        self.h = h * 1.0
        self.kf = kalman_filter.KalmanFilter()
        # print(self.map.shape)

    def update(self, tracks: list[Track], id_deleted):
        new_hashmap = dict()
        # for key in list(self.hashmap.keys()):
        #     if self.hashmap[key][2] in idx_deleted:
        #         self.hashmap.pop(key)
        for id in id_deleted:
            if id in self.hashmap:
                self.hashmap.pop(id)
        for idx, track in enumerate(tracks):
            if track.is_confirmed():
                # if track.track_id not in self.hashmap:
                cx, cy = track.mean[:2]
                # cx = cx if cx < self.w else self.w - 1
                # cy = cy if cy < self.h else self.h - 1
                vx, vy = 0, 0
                # 速度向量
                # v = np.array([vx, vy])
                if track.track_id in self.hashmap:
                    if track.time_since_update==0:
                        v = track.mean[4:6]*0.1 + self.hashmap[track.track_id][1]*0.9
                    else:
                        v = self.hashmap[track.track_id][1]
                    # if track.time_since_update==0:
                    #     cnt = self.hashmap[track.track_id][5]+1
                else:
                    v = track.mean[4:6].copy()
                    # cnt = 1
                # 位置向量
                pos = np.array([cx, cy])
                self.hashmap[track.track_id] = [pos, v, idx, track.mean[2:4].copy(), track.time_since_update]
                # else:
                #     old_cx, old_cy = self.hashmap[track.track_id][0]
                #     # old_vx, old_vy = self.hashmap[track.track_id][1]
                #     cx, cy = track.mean[:2]
                #     cx = cx if cx < self.w else self.w - 1
                #     cy = cy if cy < self.h else self.h - 1
                #     vx, vy = cx - old_cx, cy - old_cy
                #     # 速度向量
                #     v = np.array([vx, vy])
                #     v = track.mean[5:6]
                #     # v = v/np.linalg.norm(v)
                #     # 位置向量
                #     pos = np.array([cx, cy])
                #     self.hashmap[track.track_id] = [pos, v, idx, track.mean[2:4]]
            # else:
            #     pos, v, _ = self.hashmap[track.track_id]
            #     new_hashmap[track.track_id] = [pos + v, v, idx]

        # self.hashmap = new_hashmap

    # matches中的tracks已经过卡尔曼的更新。
    def predict(self, matches, unmatched_tracks, unmatched_detections: list, tracks: list[Track], detections):
        if len(unmatched_tracks) == 0 or len(unmatched_detections) == 0:
            # return matches, unmatched_tracks, unmatched_detections
            return [], unmatched_tracks, unmatched_detections
        # hashmap 记录第t-1帧中，匹配成功的tracks：id，pos，v
        # unmatched_tracks：第t帧中，未成功匹配的tracks：id
        cur_matches_set = set([item[0] for item in matches])  # 第t帧中，匹配成功的tracks：idx (As_cur)
        # for value in self.hashmap.values():
        #     print(value[2])
        #     print(len(tracks))
        pre_matches_set = set(item[2] for item in self.hashmap.values() if
                              item[4]==0)  # t-1帧中的tracks：idx
        enable_tracks = cur_matches_set & pre_matches_set  # 可以用来预测未匹配track的tracks
        # print(len(pre_matches_set))
        # print(len(cur_matches_set))
        # print(len(enable_tracks))
        if len(enable_tracks) == 0:
            for idx_unmatched_track in unmatched_tracks:
                tracks[idx_unmatched_track].time_since_map_update += 1
            return [], unmatched_tracks, unmatched_detections
        matches_b = list()
        unmatched_tracks_b = list()
        unmatched_detections_b = list()
        # print(len(unmatched_tracks))
        # print(len(unmatched_detections))
        matched_tracks_without_detection = []
        for idx_unmatched_track in unmatched_tracks:
            tracks[idx_unmatched_track].time_since_map_update += 1
            j = tracks[idx_unmatched_track].track_id
            T, D = [], []
            if j in self.hashmap:
            # if j in self.hashmap:
                B_pre = self.hashmap[j][0]
                vB_pre = self.hashmap[j][1]
                pos_predict = np.zeros(2)
                # pos_predicts = []
                for idx_enable_track in enable_tracks:
                    i = tracks[idx_enable_track].track_id
                    A_cur = tracks[idx_enable_track].mean[:2]
                    A_pre = self.hashmap[i][0]
                    vA_pre = self.hashmap[i][1]
                    # print(vA_pre)
                    # print('A_pre: ', A_pre)
                    # print('A_cur: ', A_cur)
                    d_pre = B_pre - A_pre  # t-1帧中的距离向量
                    d_cur = d_pre - vA_pre + vB_pre  # t帧中的距离向量
                    B_cur = A_cur + d_cur
                    pos_predict += B_cur
                    T.append(d_pre)
                    D.append(d_cur)
                    # print(d_pre)
                    # print(d_cur)
                    # pos_predicts.append(B_cur)
                pos_predict /= len(enable_tracks)  # 预测的位置向量
                # print(pos_predict, B_pre)
                # pos_predict = B_pre+vB_pre
                n = len(T)
                TD = [np.dot(T[i], D[i]) for i in range(n)]
                T_mold = [np.linalg.norm(T[i]) for i in range(n)]
                D_mold = [np.linalg.norm(D[i]) for i in range(n)]
                TD, T_mold, D_mold = np.array(TD), np.array(T_mold), np.array(D_mold)
                TD = TD / T_mold / D_mold
                S = (1 - TD) + (np.abs(T_mold - D_mold) / (T_mold + D_mold))
                # cnt = sum([1 for x in S if x>0.5])
                ave = np.sum(S)/len(S)
                idx_detection = self.inwitch(pos_predict, detections, unmatched_detections)
                if tracks[idx_unmatched_track].time_since_update == 1 and idx_detection is not None and ave<alpha:  # 如果能与某个detection匹配
                    # print('********************************************************************************************************')
                    matches_b.append((idx_unmatched_track, idx_detection))
                    unmatched_detections.remove(idx_detection)
                elif ave < alpha and tracks[idx_unmatched_track].times_of_map_predict < 30:
                    if 0<pos_predict[0] < self.w and 0<pos_predict[1] < self.h:
                    # if True:
                        # print(
                        #     '********************************************************************************************************')
                        # tracks[idx_unmatched_track].time_since_update = 1
                        tracks[idx_unmatched_track].times_of_map_predict += 1
                        tracks[idx_unmatched_track].time_since_map_update = 0
                        # if tracks[idx_unmatched_track].times_of_map_predict<20:
                        #     tracks[idx_unmatched_track].time_since_update = 1
                        tracks[idx_unmatched_track].mean, tracks[idx_unmatched_track].covariance = \
                            self.kf.update(tracks[idx_unmatched_track].mean, tracks[idx_unmatched_track].covariance, \
                                           np.r_[pos_predict, self.hashmap[j][3]])
                    else:
                        unmatched_tracks_b.append(idx_unmatched_track)
                else:
                    unmatched_tracks_b.append(idx_unmatched_track)
            else:
                unmatched_tracks_b.append(idx_unmatched_track)

        return matches_b, unmatched_tracks_b, unmatched_detections

    def inwitch(self, pos_predict, detections, unmatched_detections):
        x, y = pos_predict
        for idx in unmatched_detections:
            tlbr = detections[idx].to_tlbr()
            # w = tlbr[2] - tlbr[0]
            # h = tlbr[3] - tlbr[1]
            # left = tlbr[0] + w / 4
            # right = tlbr[2] - w / 4
            # top = tlbr[1] + h / 4
            # bottom = tlbr[3] - h / 4
            # if left < x < right and top < y < bottom:
            #     # print(w, h)
            #     return idx
            if tlbr[0] < x < tlbr[2] and tlbr[1] < y < tlbr[3]:
                return idx
        return None

    def confirm(self, tracks: list[Track], detections, matches_a, unmatched_tracks_a, unmatched_detections):
        # idx_tracks = [match[0] for match in matches_a \
        #               if tracks[match[0]].time_since_update == 1 or tracks[match[0]].times_of_map_predict < 30]
        # idx_detections = [match[1] for match in matches_a \
        #                   if tracks[match[0]].time_since_update == 1 or tracks[match[0]].times_of_map_predict < 30]
        idx_tracks = [match[0] for match in matches_a if tracks[match[0]].time_since_update == 1]
        idx_detections = [match[1] for match in matches_a if tracks[match[0]].time_since_update == 1]
        n = len(idx_detections)
        matches_b = list(zip(idx_tracks, idx_detections))
        if n<2:
            return matches_a
        # print(n)
        # idx_tracks = [match[0] for match in matches_a if tracks[match[0]].time_since_update<=1]
        # idx_detections = [match[1] for match in matches_a if tracks[match[0]].time_since_update<=1]
        # n = len(idx_tracks)
        # print(n)
        # T = [[tracks[i].mean[:2] - tracks[j].mean[:2] for j in idx_tracks] for i in idx_tracks]
        T = [[self.hashmap[tracks[i].track_id][0]-self.hashmap[tracks[j].track_id][0] for j in idx_tracks] for i in idx_tracks]
        # print(T)
        D = [[detections[i].to_xyah()[:2] - detections[j].to_xyah()[:2] for j in idx_detections] for i in
             idx_detections]
        TD = []
        T_mold = []
        D_mold = []
        for i in range(n):
            row, row_T, row_D = [], [], []
            for j in range(n):
                if i != j:
                    T_norm = np.linalg.norm(T[i][j])
                    D_norm = np.linalg.norm(D[i][j])
                    row.append(np.dot(T[i][j] / T_norm, D[i][j]) / D_norm)
                    row_T.append(T_norm)
                    row_D.append(D_norm)
                else:
                    row.append(1.0)
                    row_T.append(1.0)
                    row_D.append(1.0)
            TD.append(row)
            T_mold.append(row_T)
            D_mold.append(row_D)
        TD, T_mold, D_mold = np.array(TD), np.array(T_mold), np.array(D_mold)
        # print(TD)
        # print((np.abs(T_mold-D_mold)/(T_mold+D_mold)))
        S = (1 - TD) + (np.abs(T_mold - D_mold) / (T_mold + D_mold))
        # print(S.shape)
        # l = [i for i in range(n) for j in range(n) if S[i][j] > 0.1]
        S_sum = np.sum(S, axis=1)
        S_ave = S_sum/(n-1)
        matches = [match for match in matches_a if tracks[match[0]].time_since_update!=1]
        for i in range(n):
            # if i in cnt and cnt[i] > n // 2 and tracks[matches_a[i][0]].time_since_update < 2:
            if S_ave[i]>alpha:
                # print('#\n*\n*\n', i, '\n*\n*\n#\n')
                # unmatched_tracks_a.append(matches_a[i][0])
                # unmatched_detections.append(matches_a[i][1])
                unmatched_tracks_a.append(matches_b[i][0])
                unmatched_detections.append(matches_b[i][1])
            else:
                matches.append(matches_b[i])
        return matches
        # if len(l) > 0:
        #     cnt = Counter(l)
        #     matches = [match for match in matches_a if tracks[match[0]].time_since_update!=1]
        #     # matches = []
        #     # print('*\n*\n*\n', cnt, '\n*\n*\n*\n')
        #     for i in range(n):
        #         # if i in cnt and cnt[i] > n // 2 and tracks[matches_a[i][0]].time_since_update < 2:
        #         if i in cnt and cnt[i] > n // 2:
        #             print('#\n*\n*\n', cnt, '\n*\n*\n#\n')
        #             # unmatched_tracks_a.append(matches_a[i][0])
        #             # unmatched_detections.append(matches_a[i][1])
        #             unmatched_tracks_a.append(matches_b[i][0])
        #             unmatched_detections.append(matches_b[i][1])
        #         else:
        #             matches.append(matches_b[i])
        #     return matches
        # return matches_a
