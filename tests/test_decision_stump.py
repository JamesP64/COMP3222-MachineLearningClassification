import pytest
import numpy as np
from solution.decision_stump import DecisionStumpClassifier

class TestDecisionStumpClassifier:
    """Unit tests for the DecisionStumpClassifier."""

    def test_float_in_X_raises_exception(self):
            """Make sure float attributes in X raise an Error."""
            X = np.array([
            ["yes", 1.653, "yes"],
            ["yes", "yes", "yes"],
            ["yes", "no", "no"],
            ["yes", "no", "no"],
            ["no", "yes", "no"],
            ["no", "yes", "yes"],
            ["no", "yes", "yes"],
            ["no", "no", "yes"],
            ["no", "no", "yes"]
            ], dtype=object)

            y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

            clf = DecisionStumpClassifier(quality_measure="ig", random_state=42)
            with pytest.raises(TypeError):
                clf.fit(X,y)

    def test_deterministic_attribute_choice_with_random_state(self):
        """Verify reproducible attribute selection when random_state is fixed."""
        X = np.array([
            ["yes", "no", "yes"],
            ["yes", "yes", "yes"],
            ["yes", "no", "no"],
            ["yes", "no", "no"],
            ["no", "yes", "no"],
            ["no", "yes", "yes"],
            ["no", "yes", "yes"],
            ["no", "no", "yes"],
            ["no", "no", "yes"]
            ], dtype=object)
        
        y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

        clf1 = DecisionStumpClassifier(quality_measure="ig", random_state=42)
        clf1.fit(X, y)

        clf2 = DecisionStumpClassifier(quality_measure="ig", random_state=42)
        clf2.fit(X, y)

        assert clf1.att_index == clf2.att_index, "Attribute selection should be the same with the same random state"


    def test_predict_proba_returns_valid_probabilities(self):
        """
        Check that predict_proba:
          - returns a numpy array of correct shape (n_cases, n_classes)
          - produces rows summing to 1
          - Probability vector is aligned with the learned class order
        """
        X = np.array([
            ["yes", "no", "yes"],
            ["yes", "yes", "yes"],
            ["yes", "no", "no"],
            ["yes", "no", "no"],
            ["no", "yes", "no"],
            ["no", "yes", "yes"],
            ["no", "yes", "yes"],
            ["no", "no", "yes"],
            ["no", "no", "yes"]
            ], dtype=object)
        
        y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

        clf = DecisionStumpClassifier(quality_measure="ig", random_state=42)
        clf.fit(X, y)
        probaArray = clf.predict_proba(X)

        assert probaArray.shape == (X.shape[0], len(clf.classes_)), "Returned array should be of shape (n_cases, n_classes)"

        for row in probaArray:
            assert sum(row) == pytest.approx(1.0), "Each probability row must sum to 1"

        X2 = np.array([
            ["yes"],
            ["yes"],
            ["yes"],
            ["yes"],
            ["no"],
            ["no"],
            ["no"],
            ["no"],
            ["no"]
            ], dtype=object)  
        clf2 = DecisionStumpClassifier(quality_measure="ig", random_state=42)
        clf2.fit(X2, y)
        probaArray2 = clf2.predict_proba(X)
        
        assert probaArray2[0][0] > probaArray2[0][1], "P(0) should be higher than P(1)"


    def test_unseen_category_fallback_to_root_prior(self):
        """Confirm that unseen categories during prediction use the root prior distribution."""
        
        X = np.array([
            ["yes", "no", "yes"],
            ["yes", "yes", "yes"],
            ["yes", "no", "no"],
            ["yes", "no", "no"],
            ["no", "yes", "no"],
            ["no", "yes", "yes"],
            ["no", "yes", "yes"],
            ["no", "no", "yes"],
            ["no", "no", "yes"]
            ], dtype=object)
        
        X_test = np.array([["unseen", "unseen", "unseen"]], dtype=object)
        
        y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

        clf = DecisionStumpClassifier(quality_measure="ig", random_state=42)
        clf.fit(X, y)

        proba = clf.predict_proba(X_test)[0]

        assert np.allclose(proba, clf.root_prior), "Unseen categories should fall back to the root prior"

    def test_tie_breaking_on_equal_quality(self):
        """Check that when two attributes have equal quality, the lower column index is chosen."""
        X = np.array([
            ["yes", "yes"],
            ["yes", "yes"],
            ["yes", "yes"],
            ["yes", "yes"],
            ["no", "no"],
            ["no", "no"],
            ["no", "no"],
            ["no", "no"],
            ["no", "no"]
            ], dtype=object)
        
        y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        
        clf = DecisionStumpClassifier(quality_measure="ig", random_state=42)
        clf.fit(X, y)

        assert clf.att_index == 0, "Should choose first attribute as they are equal"

