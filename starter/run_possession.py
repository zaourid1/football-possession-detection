"""Run the integrated offline model; optionally export its frame,label CSV.

python3 starter/run_possession.py
python3 starter/run_possession.py --out evaluation/results/paper_pipeline.csv
No public or private ground truth is loaded.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from possession_data import load_tracking
from possession_pipeline import PipelineConfig, predict_pipeline
from possession_layered import predict_layered
from load_smart_tagging import load as load_tags
from baseline import write_submission
from games import GAME, GAMES


def summarize(result, data):
    graded = data.half != 0
    adjacent = (data.half[1:] == data.half[:-1]) & (data.half[1:] != 0)
    labels = result.labels
    completed = [event for event in result.passes.events if event.completed]
    return {
        "graded_frames": int(graded.sum()),
        "label_counts": {label: int(np.count_nonzero(labels[graded] == label))
                         for label in ("H", "A", "D", "X")},
        "coverage": float(np.mean(labels[graded] != "X")),
        "raw_label_changes": int(np.count_nonzero(adjacent & (labels[1:] != labels[:-1]))),
        "controlled_episodes": int(len(np.unique(result.contacts.episode_id[
            result.contacts.episode_id >= 0]))),
        "completed_pass_candidates": {name: sum(event.team == team for event in completed)
                                      for team, name in ((0, "H"), (1, "A"))},
        "uncertain_pass_candidates": sum(not event.completed for event in result.passes.events),
        "bridged_frames": int(np.count_nonzero(np.isin(
            result.reasons, ["bridged_same_team", "layered_bridged"]))),
        "accuracy": "unverified; diagnostics use no match ground truth",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", help="Override match asset directory")
    parser.add_argument("--out", help="Optional frame,label CSV path")
    parser.add_argument("--report", help="Optional JSON configuration and aggregate diagnostics path")
    parser.add_argument("--model", choices=("paper", "layered"), default="paper",
                        help="paper: strict physical pipeline; layered: retain earlier predictions")
    args = parser.parse_args()
    if args.report and args.out and Path(args.report).resolve() == Path(args.out).resolve():
        parser.error("Report and submission must use different paths")
    config = PipelineConfig()
    data = load_tracking(data_dir=args.data_dir)
    events = load_tags(data_dir=args.data_dir)
    predictor = predict_layered if args.model == "layered" else predict_pipeline
    result = predictor(data, events, GAMES[GAME]["pitch"], config)
    if result.labels.shape != (GAMES[GAME]["n_video_frames"],):
        raise ValueError("Unexpected prediction shape")
    if not np.isin(result.labels, ["H", "A", "D", "X"]).all():
        raise ValueError("Invalid label")
    if not (result.labels[data.half == 0] == "X").all():
        raise ValueError("Ungraded frames must remain X")
    if args.out:
        write_submission(args.out, result.labels)
    diagnostics = summarize(result, data)
    if args.model == "layered":
        graded = data.half != 0
        diagnostics["reference_coverage"] = float(np.mean(result.reference_labels[graded] != "X"))
        diagnostics["reference_answers_erased"] = int(np.count_nonzero(
            graded & (result.reference_labels != "X") & (result.labels == "X")))
        diagnostics["changed_reference_frames"] = int(np.count_nonzero(
            graded & (result.reference_labels != result.labels)))
        sources, counts = np.unique(result.source[graded], return_counts=True)
        diagnostics["prediction_sources"] = {str(key): int(value) for key, value in zip(sources, counts)}
    report = json.dumps({"model": args.model, "config": asdict(config), "summary": diagnostics,
                         "submission": args.out}, indent=2)
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
