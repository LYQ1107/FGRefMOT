import numpy as np

from tools.qim_fast6.rmot_eval import evaluate_sequence_rows, metric_data_from_rows


def test_rmot_metric_adapter_reports_perfect_identity_sequence():
    gt = [[(7, np.asarray([0, 0, 10, 10], dtype=np.float32))] for _ in range(3)]
    pred = [[(4, np.asarray([0, 0, 10, 10], dtype=np.float32))] for _ in range(3)]
    data = metric_data_from_rows(gt, pred)
    summary = evaluate_sequence_rows(gt, pred)
    assert data["num_timesteps"] == 3
    assert summary["HOTA"] > 0.99
    assert summary["DetA"] > 0.99
    assert summary["AssA"] > 0.99
    assert summary["IDF1"] > 0.99
    assert summary["IDSW"] == 0
