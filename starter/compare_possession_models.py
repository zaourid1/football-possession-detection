"""Compare aggregate model behaviour without reading any ground truth.

Run: python3 starter/compare_possession_models.py
These statistics measure coverage and fragmentation, not accuracy.
"""

import json

import numpy as np

from possession_data import load_tracking
from possession_features import extract_features
from possession_model import predict_possession
from possession_refinement import bridge_missing_gaps
from possession_stoppages import predict_with_stillness
from possession_pipeline import predict_pipeline
from possession_layered import combine_layers
from load_smart_tagging import load as load_tags
from games import GAME, GAMES


def summary(labels, half):
    graded = half != 0
    adjacent = (half[1:] == half[:-1]) & (half[1:] != 0)
    return {
        "label_counts": {label: int(np.count_nonzero(labels[graded] == label))
                         for label in ("H", "A", "D", "X")},
        "raw_label_changes": int(np.count_nonzero(adjacent & (labels[1:] != labels[:-1]))),
        "coverage": float(np.mean(labels[graded] != "X")),
    }


if __name__ == "__main__":
    data = load_tracking()
    features = extract_features(data)
    events = load_tags()
    pitch = GAMES[GAME]["pitch"]
    original, _ = predict_possession(data, features, events, pitch)
    accumulated, reasons = predict_possession(data, features, events, pitch, accumulated=True)
    refined, refined_reasons = bridge_missing_gaps(accumulated, reasons, data, features, events)
    robust, robust_reasons, stillness = predict_with_stillness(data, features, events, pitch)
    paper = predict_pipeline(data, events, pitch, features=features)
    layered, _, _ = combine_layers(data, robust, robust_reasons, paper)
    for labels in (original, accumulated, refined, robust, paper.labels, layered):
        assert len(labels) == GAMES[GAME]["n_video_frames"]
        assert np.isin(labels, ["H", "A", "D", "X"]).all()
        assert (labels[data.half == 0] == "X").all()
    assert np.all(refined[accumulated != "X"] == accumulated[accumulated != "X"])
    print(json.dumps({
        "original": summary(original, data.half),
        "accumulated_only": summary(accumulated, data.half),
        "accumulated_and_bridged": summary(refined, data.half),
        "original_with_robust_stillness": summary(robust, data.half),
        "paper_pipeline": summary(paper.labels, data.half),
        "layered_pipeline": summary(layered, data.half),
        "robust_stationary_restart_frames": int(np.count_nonzero(stillness.stationary_restart)),
        "bridged_frames": int(np.count_nonzero(refined_reasons == "bridged_same_team")),
        "changed_graded_frames": int(np.count_nonzero((original != refined) & (data.half != 0))),
        "accuracy_improvement": "unknown; no permitted labelled validation set used",
    }, indent=2))
