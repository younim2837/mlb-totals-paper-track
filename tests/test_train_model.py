import unittest

import pandas as pd

import train_model


class TrainModelTests(unittest.TestCase):
    def test_train_test_split_by_time_uses_2021_to_2024_for_training_and_2025_for_test(self):
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    ["2020-09-01", "2021-04-01", "2024-09-01", "2025-04-01", "2026-04-01"]
                ),
                "value": [0, 1, 2, 3, 4],
            }
        )

        train, test = train_model.train_test_split_by_time(df)

        self.assertEqual(train["value"].tolist(), [1, 2])
        self.assertEqual(test["value"].tolist(), [3])


if __name__ == "__main__":
    unittest.main()
