import unittest
import math
from solution import attribute_quality as aq




class TestAttributeQuality(unittest.TestCase):

    def setUp(self):
       self.test_cases = {
            "Peaty Depth 1": {
                "table": [[4, 0], [1, 5]],
                "chi_expected": 20/3,
                "ig_expected": 0.6100,
                "ig_ratio_expected": 0.6283,
                "chi_yates_expected": 3.7500,
            },
            "Woody Depth 1": {
                "table": [[2, 3], [3, 2]],
                "chi_expected": 0.4,
                "ig_expected": 0.0290,
                "ig_ratio_expected": 0.0290,
                "chi_yates_expected": 0.0000,
            },
            "Sweet Depth 1": {
                "table": [[2, 5], [3, 0]],
                "chi_expected": 30/7,
                "ig_expected": 0.3958,
                "ig_ratio_expected": 0.4491,
                "chi_yates_expected": 1.9048,
            },
            "Woody Depth 2": {
                "table": [[1, 3], [0, 2]],
                "chi_expected": 0.6,
                "ig_expected": 0.1091,
                "ig_ratio_expected": 0.1188,
                "chi_yates_expected": 0.1500,
            },
            "Sweet Depth 2": {
                "table": [[0, 5], [1, 0]],
                "chi_expected": 6.0,
                "ig_expected": 0.65,
                "ig_ratio_expected": 1.0000,
                "chi_yates_expected": 0.9600,
            },
       }    

    def test_chi_squared(self):
        for name, case in self.test_cases.items():
            result = aq.chi_squared(case["table"])
            self.assertAlmostEqual(
                result, case["chi_expected"], places=3,
                msg=f"Chi-squared mismatch for {name}"
            )
            print(f"Chi-squared for {name} = {result:.4f}")

    def test_information_gain(self):
        for name, case in self.test_cases.items():
            result = aq.information_gain(case["table"])
            self.assertAlmostEqual(
                result, case["ig_expected"], places=3,
                msg=f"Information gain mismatch for {name}"
            )
            print(f"Information gain for {name} = {result:.4f}")

    def test_information_gain_ratio(self):
        for name, case in self.test_cases.items():
            result = aq.information_gain_ratio(case["table"])
            self.assertAlmostEqual(
                result, case["ig_ratio_expected"], places=3,
                msg=f"Information gain ratio mismatch for {name}"
            )
            print(f"Information gain ratio for {name} = {result:.4f}")

    def test_chi_squared_yates(self):
        for name, case in self.test_cases.items():
            result = aq.chi_squared_yates(case["table"])
            self.assertAlmostEqual(
                result, case["chi_yates_expected"], places=3,
                msg=f"Chi-squared mismatch for {name}"
            )
            print(f"Chi-squared for {name} = {result:.4f}")

if __name__ == "__main__":
    unittest.main()