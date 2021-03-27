
import numpy as np
from scipy.optimize import linear_sum_assignment
from ._base_metric import _BaseMetric
from .. import _timing


# TODO: Is this name OK?
class CLEARDet(_BaseMetric):
    """Class which implements the CLEAR detection metrics.

    Based on: matlab_devkit/evaluateDetection.m
    From: github.com/dendorferpatrick/MOTChallengeEvalKit
    """
    def __init__(self):
        super().__init__()
        # TODO: Are these prefixes OK?
        self.integer_fields = ['CLRDet_TP', 'CLRDet_FP', 'CLRDet_FN', 'CLRDet_num_timesteps']
        # TODO: Name 'MODA' will clash with CLEAR.
        self.float_fields = ['AP', 'MODA', 'MODP', 'MODP_sum', 'FAF', 'CLRDet_Re', 'CLRDet_Pr', 'CLRDet_F1']
        self.fields = self.integer_fields + self.float_fields
        self.summary_fields = ['AP', 'MODA', 'MODP', 'FAF',
                               'CLRDet_TP', 'CLRDet_FP', 'CLRDet_FN',
                               'CLRDet_Re', 'CLRDet_Pr', 'CLRDet_F1']

        self._summed_fields = self.integer_fields + ['MODP_sum']
        # TODO: More idiomatic to use 'AP_sum' and 'CLRDet_Sequences'?
        self._averaged_fields = ['AP']

        self.threshold = 0.5
        # TODO: Should we use a float_array_field?
        self.recall_thresholds = np.arange(0, 10 + 1) / 10.

    @_timing.time
    def eval_sequence(self, data):
        """Calculates CLEAR metrics for one sequence"""
        # Initialise results
        res = {}
        for field in self.fields:
            res[field] = 0
        res['CLRDet_num_timesteps'] = data['num_timesteps']

        # Find per-frame correspondence by priority of score.
        num_timesteps = data['num_timesteps']
        correct = [None for _ in range(num_timesteps)]
        for t in range(num_timesteps):
            correct[t] = _match_by_score(data['tracker_confidences'][t],
                                         data['similarity_scores'][t],
                                         self.threshold)
        # Concatenate results from all frames to compute AUC.
        scores = np.concatenate(data['tracker_confidences'])
        correct = np.concatenate(correct)
        precs = _find_prec_at_recall(data['num_gt_dets'], scores, correct, self.recall_thresholds)
        res['AP'] = np.mean(precs)

        # Find per-frame correspondence (without accounting for switches).
        for t, (gt_ids_t, tracker_ids_t) in enumerate(zip(data['gt_ids'], data['tracker_ids'])):
            # Deal with the case that there are no gt_det/tracker_det in a timestep.
            if len(gt_ids_t) == 0:
                res['CLRDet_FP'] += len(tracker_ids_t)
                continue
            if len(tracker_ids_t) == 0:
                res['CLRDet_FN'] += len(gt_ids_t)
                continue

            # Construct score matrix to optimize number of matches and then localization.
            similarity = data['similarity_scores'][t]
            assert np.all(~(similarity < 0))
            assert np.all(~(similarity > 1))
            eps = 1. / (max(similarity.shape) + 1.)
            overlap_mask = (similarity >= self.threshold)
            score_mat = overlap_mask.astype(np.float64) + eps * (similarity * overlap_mask)
            # Hungarian algorithm to find best matches
            match_rows, match_cols = linear_sum_assignment(-score_mat)
            actually_matched_mask = overlap_mask[match_rows, match_cols]
            match_rows = match_rows[actually_matched_mask]
            match_cols = match_cols[actually_matched_mask]

            # Calculate and accumulate basic statistics
            num_matches = np.sum(overlap_mask[match_rows, match_cols])
            res['CLRDet_TP'] += num_matches
            res['CLRDet_FN'] += len(gt_ids_t) - num_matches
            res['CLRDet_FP'] += len(tracker_ids_t) - num_matches
            res['MODP_sum'] += np.sum(similarity[match_rows, match_cols])

        res = self._compute_final_fields(res)
        return res

    @staticmethod
    def _compute_final_fields(res):
        """Calculate sub-metric ('field') values which only depend on other sub-metric values.
        This function is used both for both per-sequence calculation, and in combining values across sequences.
        """
        res = dict(res)
        res['MODA'] = (res['CLRDet_TP'] - res['CLRDet_FP']) / np.maximum(1.0, res['CLRDet_TP'] + res['CLRDet_FN'])
        res['MODP'] = res['MODP_sum'] / np.maximum(1.0, res['CLRDet_TP'])
        res['CLRDet_Re'] = res['CLRDet_TP'] / np.maximum(1.0, res['CLRDet_TP'] + res['CLRDet_FN'])
        res['CLRDet_Pr'] = res['CLRDet_TP'] / np.maximum(1.0, res['CLRDet_TP'] + res['CLRDet_FP'])
        res['CLRDet_F1'] = res['CLRDet_TP'] / (
                np.maximum(1.0, res['CLRDet_TP'] + 0.5*res['CLRDet_FN'] + 0.5*res['CLRDet_FP']))
        res['FAF'] = res['CLRDet_FP'] / res['CLRDet_num_timesteps']
        return res

    def combine_sequences(self, all_res):
        res = {}
        for field in self._summed_fields:
            res[field] = self._combine_sum(all_res, field)
        for field in self._averaged_fields:
            res[field] = self._combine_sum(all_res, field) / len(all_res)
        res = self._compute_final_fields(res)
        return res

    def combine_classes_det_averaged(self, all_res):
        # TODO: Implement.
        raise NotImplementedError

    def combine_classes_class_averaged(self, all_res):
        # TODO: Implement.
        raise NotImplementedError


def _match_by_score(confidence, similarity, similarity_threshold):
    """Matches by priority of confidence.

    Args:
        confidence: Array of shape [num_pred].
        similarity: Array of shape [num_gt, num_pred].
        similarity_threshold: Scalar constant.

    Returns:
        Array of bools indicating whether prediction was matched.

    Assumes confidence scores are unique.
    """
    # Sort descending by confidence, preserve order of repeated elements.
    order = np.argsort(-confidence, kind='stable')
    is_matched = np.full(confidence.shape, False)
    feasible = (similarity >= similarity_threshold)
    for j in order:
        if not np.any(feasible[:, j]):
            continue
        subset, = np.nonzero(feasible[:, j])
        i = subset[np.argmax(similarity[subset, j])]
        feasible[i, :] = False
        is_matched[j] = True
    return is_matched


def _find_prec_at_recall(num_gt, scores, correct, thresholds):
    """
    Args:
        num_gt: Number of ground-truth elements.
        scores: Score of each prediction.
        correct: Whether or not each prediction is correct.
        thresholds: Recall thresholds at which to evaluate.

    Follows implementation from Piotr Dollar toolbox.
    """
    # Sort descending by score.
    order = np.argsort(-scores, kind='stable')
    correct = correct[order]
    # Note: cumsum() does not include operating point with zero predictions.
    # However, this matches the original implementation.
    tp = np.cumsum(correct)
    num_pred = 1 + np.arange(len(scores))
    recall = np.true_divide(tp, num_gt)
    prec = np.true_divide(tp, num_pred)
    # Extend curve to infinity with zeros.
    recall = np.concatenate([recall, [np.inf]])
    prec = np.concatenate([prec, [0.]])
    assert np.all(np.isfinite(thresholds)), 'assume finite thresholds'
    # Find first element with minimum recall.
    # Use argmax() to take first element that satisfies criterion.
    return np.asarray([prec[np.argmax(recall >= threshold)] for threshold in thresholds])