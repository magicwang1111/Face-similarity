from __future__ import annotations

import unittest

from face_lora_eval.scoring import RankingWeights, aggregate_lora_scores, percentile


class ScoringTests(unittest.TestCase):
    def test_percentile_interpolates(self) -> None:
        self.assertAlmostEqual(percentile([1.0, 2.0, 3.0, 4.0], 0.25), 1.75)

    def test_aggregate_lora_scores_ranks_by_stable_formula(self) -> None:
        rows = [
            {"lora_id": "stable", "status": "ok", "similarity": 0.60},
            {"lora_id": "stable", "status": "ok", "similarity": 0.62},
            {"lora_id": "stable", "status": "ok", "similarity": 0.61},
            {"lora_id": "spiky", "status": "ok", "similarity": 0.80},
            {"lora_id": "spiky", "status": "no_face", "similarity": ""},
            {"lora_id": "spiky", "status": "ok", "similarity": 0.50},
        ]

        ranking = aggregate_lora_scores(rows, RankingWeights())

        self.assertEqual(ranking[0]["lora_id"], "stable")
        self.assertEqual(ranking[0]["valid_count"], 3)
        self.assertEqual(ranking[1]["failed_count"], 1)


if __name__ == "__main__":
    unittest.main()

