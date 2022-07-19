# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track
from initializer import raw

class Tracker:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.

    """

    def __init__(self, metric, IDmap, max_iou_distance=0.7, max_age=70, n_init=3):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init

        self.kf = kalman_filter.KalmanFilter()
        self.tracks = []
        self._next_id = 1
        self.IDmap = IDmap

        self.cnt = 0

    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections):
        # 根据当前帧的detections和卡尔曼滤波的预测，更新所有tracks
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[DeepSORT.detection.Detection]
            A list of detections at the current time step.

        """
        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections)

        self.cnt += len(unmatched_detections)
        # print(self.cnt)
        # print(len(unmatched_detections))
        # # Associate tracks using inconnection.
        # for track_idx, detection_idx in matches:
        #     self.tracks[track_idx].update(
        #         self.kf, detections[detection_idx])
        # matches, unmatched_tracks, unmatched_detections = self.IDmap.predict(
        #     matches, unmatched_tracks, unmatched_detections, self.tracks, detections
        # )
        # print('After my: ', len(matches), len(unmatched_tracks), len(unmatched_detections))
        # Update track set.
        # 根据级联匹配，对于3种不同的匹配结果分别更新
        # （track与detection匹配对，没有detection与之匹配的track， 没有track与之匹配的detection）
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(
                self.kf, detections[detection_idx])
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])

        # idx_deleted = set([i for i in range(len(self.tracks)) if self.tracks[i].is_deleted()])
        id_deleted = [t.track_id for t in self.tracks if t.is_deleted()]
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update IDmap.
        self.IDmap.update(self.tracks, id_deleted)
        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)

        # features, targets = [], []
        # for track in self.tracks:
        #     if not track.is_confirmed():
        #         continue
        #     features.append(track.features)
        #     targets.append(track.track_id)
        #     # ******************************************待完善
        #     # track.features = {1: [], 2: [], 3: []}
        # self.metric.my_partial_fit(features, targets)


    def _match(self, detections):
        # 匹配上一帧中的tracks与当前帧中的detections
        # 该调用该函数时，已对tracks进行过predicate，因此此时
        # 在上一帧update的tracks的time_since_update为1
        # 在上一帧initial的tracks的time_since_update为1
        # 在上一帧未能update的tracks的time_since_update>1
        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            # feature_types = np.array([dets[i].feature_type for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            # 余弦距离
            cost_matrix = self.metric.distance(features, targets)
            # 与马氏距离融合，共同形成新的cost_matrix
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices,
                detection_indices)

            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # Associate confirmed tracks using appearance features.
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections, confirmed_tracks)

        # print(len(matches_a), len(unmatched_tracks_a), len(unmatched_detections))
        # *******************************************************************************************************
        if not raw:
            # 二次确认。
            matches_a = self.IDmap.confirm(self.tracks, detections, matches_a, unmatched_tracks_a, unmatched_detections)
            # print(len(matches_a), len(unmatched_tracks_a), len(unmatched_detections))

            # print(len(matches_a), len(unmatched_tracks_a), len(unmatched_detections))
            for track_idx, detection_idx in matches_a:
                self.tracks[track_idx].update(
                    self.kf, detections[detection_idx])
            # Associate tracks using inconnection.
            matches_a, unmatched_tracks_a, unmatched_detections = self.IDmap.predict(
                matches_a, unmatched_tracks_a, unmatched_detections, self.tracks, detections
            )
            # *******************************************************************************************************
        # print(len(matches_a), len(unmatched_tracks_a), len(unmatched_detections))
        # return matches_a, unmatched_tracks_a, unmatched_detections
        # Associate remaining tracks together with unconfirmed tracks using IOU.
        # iou_track_candidates = unconfirmed_tracks
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]  # IOU匹配：只对上一帧中update过的track进行。
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)

        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]
        matches = matches_a + matches_b
        # matches = matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        # print(len(matches), len(unmatched_tracks), len(unmatched_detections))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks.append(Track(
            mean, covariance, self._next_id, self.n_init, self.max_age,
            detection.feature))
        self._next_id += 1
