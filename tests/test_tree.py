"""Tests for numcompute_stream.tree: the split helpers and DecisionTreeClassifier."""

import numpy as np
import pytest

from numcompute_stream.preprocessing import NotFittedError
from numcompute_stream.tree import DecisionTreeClassifier, _best_split, _impurity

# the twelve students from the course guide: submitted, attended, study_group, passed
STUDENTS = np.array([
    [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 0, 1], [1, 1, 0, 1], [1, 0, 1, 1],
    [1, 0, 0, 0], [1, 0, 0, 0],
    [0, 1, 1, 0], [0, 1, 1, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0],
])
X_STUDENTS, Y_STUDENTS = STUDENTS[:, :3], STUDENTS[:, 3]
X, Y = X_STUDENTS, Y_STUDENTS


# ======================= split helpers: _impurity and _best_split =======================
# --- impurity ---

def test_gini_and_entropy_of_the_course_class():
    assert _impurity([7, 5], "gini") == pytest.approx(0.4861, abs=1e-4)
    assert _impurity([7, 5], "entropy") == pytest.approx(0.9799, abs=1e-4)


def test_a_pure_group_is_exactly_zero():
    for criterion in ("gini", "entropy"):
        value = _impurity([10, 0], criterion)
        assert value == 0.0 and not np.signbit(value)


def test_half_and_half_is_the_most_mixed():
    assert _impurity([5, 5], "gini") == pytest.approx(0.5)
    assert _impurity([5, 5], "entropy") == pytest.approx(1.0)


def test_three_classes():
    assert _impurity([2, 2, 2], "gini") == pytest.approx(2 / 3)
    assert _impurity([2, 2, 2], "entropy") == pytest.approx(np.log2(3))


def test_an_empty_group_has_zero_impurity():
    assert _impurity([0, 0], "gini") == 0.0
    assert _impurity([0, 0], "entropy") == 0.0


def test_many_groups_at_once_match_one_at_a_time():
    counts = np.random.default_rng(0).integers(0, 9, size=(4, 5, 3))
    for criterion in ("gini", "entropy"):
        stacked = _impurity(counts, criterion)
        assert stacked.shape == (4, 5)
        assert stacked[2, 3] == pytest.approx(_impurity(counts[2, 3], criterion))


def test_zero_counts_give_no_numpy_warning():
    with np.errstate(all="raise"):                            # a warning would now be an error
        assert _impurity([[0, 5], [5, 0], [0, 0]], "entropy").tolist() == [0.0, 0.0, 0.0]


def test_unknown_criterion_is_rejected():
    with pytest.raises(ValueError, match="criterion"):
        _impurity([1, 2], "variance")


# --- the course example, step by step ---

def test_course_root_question_is_submitted_for_gini_and_entropy():
    assert _best_split(X_STUDENTS, Y_STUDENTS, 2, "gini") == (0, 0.5, pytest.approx(0.2480, abs=1e-3))
    assert _best_split(X_STUDENTS, Y_STUDENTS, 2, "entropy") == (0, 0.5, pytest.approx(0.4764, abs=1e-3))


def test_course_second_question_among_the_submitters_is_attended():
    submitted = X_STUDENTS[:, 0] == 1
    feature, threshold, gain = _best_split(X_STUDENTS[submitted], Y_STUDENTS[submitted], 2, "gini")
    assert (feature, threshold) == (1, 0.5)
    assert gain == pytest.approx(0.2177, abs=1e-3)


def test_numeric_ages_example_with_a_tie_picks_the_lower_cut():
    age = np.array([[22.0], [25.0], [31.0], [36.0], [40.0], [47.0], [53.0], [59.0]])
    buy = np.array([0, 0, 0, 1, 0, 1, 1, 1])
    feature, threshold, gain = _best_split(age, buy, 2, "gini")
    assert (feature, threshold) == (0, 33.5)                  # 43.5 is just as good, the lower one wins
    assert gain == pytest.approx(0.3)


# --- when there is no question worth asking ---

def test_a_pure_group_has_no_split():
    assert _best_split(np.array([[1.0], [2.0], [3.0]]), np.array([1, 1, 1]), 2, "gini") == (None, None, 0.0)


def test_one_row_has_no_split():
    assert _best_split(np.array([[1.0]]), np.array([0]), 2, "gini") == (None, None, 0.0)


def test_columns_with_all_the_same_value_have_no_split():
    assert _best_split(np.full((4, 2), 7.0), np.array([0, 1, 0, 1]), 2, "gini") == (None, None, 0.0)


def test_a_constant_column_is_skipped_for_a_useful_one():
    X = np.column_stack([np.full(4, 5.0), [1.0, 2.0, 3.0, 4.0]])
    feature, threshold, _ = _best_split(X, np.array([0, 0, 1, 1]), 2, "gini")
    assert (feature, threshold) == (1, 2.5)


def test_a_cut_that_does_not_help_is_not_made():
    X = np.array([[0.0], [0.0], [1.0], [1.0]])
    assert _best_split(X, np.array([0, 1, 0, 1]), 2, "gini") == (None, None, 0.0)


def test_no_columns_to_use_means_no_split():
    assert _best_split(np.array([[1.0], [2.0]]), np.array([0, 1]), 2, "gini", feature_ids=np.array([], dtype=int)) == (None, None, 0.0)


# --- ties, subsets and awkward numbers ---

def test_two_equally_good_columns_the_lowest_column_wins():
    column = np.array([1.0, 2.0, 3.0, 4.0])
    X = np.column_stack([column, column])
    assert _best_split(X, np.array([0, 0, 1, 1]), 2, "gini")[0] == 0


def test_feature_ids_limit_the_search_and_report_the_real_column_number():
    X = np.column_stack([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0], [1.0, 2.0, 3.0, 4.0]])
    feature, _, _ = _best_split(X, np.array([0, 0, 1, 1]), 2, "gini", feature_ids=np.array([2]))
    assert feature == 2


def test_repeated_values_are_only_cut_between_different_values():
    X = np.array([[1.0], [1.0], [1.0], [2.0], [2.0], [2.0]])
    assert _best_split(X, np.array([0, 0, 0, 1, 1, 1]), 2, "gini")[:2] == (0, 1.5)


def test_huge_values_do_not_overflow():
    X = np.array([[-1e308], [1e308]])
    feature, threshold, _ = _best_split(X, np.array([0, 1]), 2, "gini")
    assert np.isfinite(threshold) and -1e308 <= threshold < 1e308


def test_neighbouring_floats_are_still_separated():
    low, high = 1.0, np.nextafter(1.0, 2.0)
    X = np.array([[low], [high]])
    _, threshold, _ = _best_split(X, np.array([0, 1]), 2, "gini")
    assert low <= threshold < high                            # "x <= threshold" splits them


def test_three_classes_and_negative_numbers():
    X = np.array([[-5.0], [-4.0], [0.0], [1.0], [9.0], [10.0]])
    y = np.array([0, 0, 1, 1, 2, 2])
    feature, threshold, gain = _best_split(X, y, 3, "entropy")
    assert feature == 0 and threshold in (-2.0, 5.0)
    assert gain > 0


def test_integer_features_are_accepted_and_the_inputs_are_not_changed():
    X = np.array([[1, 5], [2, 4], [3, 3], [4, 2]])
    y = np.array([0, 0, 1, 1])
    X_before, y_before = X.copy(), y.copy()
    feature, threshold, gain = _best_split(X, y, 2, "gini")
    assert np.array_equal(X, X_before) and np.array_equal(y, y_before)
    assert isinstance(feature, int) and isinstance(threshold, float) and isinstance(gain, float)


# --- against a slow loop version, on data full of ties and repeats ---

def slow_best_split(X, y, n_classes, criterion):
    def impurity(group):
        if len(group) == 0:
            return 0.0
        p = np.bincount(group, minlength=n_classes) / len(group)
        if criterion == "gini":
            return 1.0 - sum(v * v for v in p)
        return -sum(v * np.log2(v) for v in p if v > 0)

    best, found = 1e-12, (None, None, 0.0)
    gains = []
    for f in range(X.shape[1]):
        values = np.unique(X[:, f])
        for lo, hi in zip(values[:-1], values[1:]):
            t = lo / 2 + hi / 2
            if t >= hi:
                t = lo
            m = X[:, f] <= t
            g = impurity(y) - (m.sum() * impurity(y[m]) + (~m).sum() * impurity(y[~m])) / len(y)
            gains.append((f, t, g))
    if not gains:
        return found
    top = max(g for _, _, g in gains)
    if not top > best:
        return found
    f, t, g = next(item for item in gains if item[2] >= top - 1e-12)
    return f, t, g


@pytest.mark.parametrize("criterion", ["gini", "entropy"])
def test_matches_the_slow_version_on_random_data(criterion):
    rng = np.random.default_rng(1)
    for _ in range(150):
        n, d, c = rng.integers(2, 25), rng.integers(1, 4), rng.integers(2, 4)
        X = rng.integers(0, 6, size=(n, d)).astype(float)       # small whole numbers: many ties and repeats
        y = rng.integers(0, c, size=n)
        fast = _best_split(X, y, c, criterion)
        slow = slow_best_split(X, y, c, criterion)
        assert fast[:2] == slow[:2]
        assert fast[2] == pytest.approx(slow[2], abs=1e-9)


# ======================= DecisionTreeClassifier =======================


# --- fit: the course example ---

def test_course_tree_arrays():
    tree = DecisionTreeClassifier(max_depth=2).fit(X, Y)
    assert tree.classes_.tolist() == [0, 1]
    assert tree.feature_.tolist() == [0, -1, 1, -1, -1]
    assert np.allclose(tree.threshold_, [0.5, np.nan, 0.5, np.nan, np.nan], equal_nan=True)
    assert tree.left_.tolist() == [1, -1, 3, -1, -1]
    assert tree.right_.tolist() == [2, -1, 4, -1, -1]
    assert tree.value_.tolist() == [[7, 5], [5, 0], [2, 5], [2, 1], [0, 4]]
    assert tree.n_nodes_ == 5


def test_course_predictions_get_11_of_12_right():
    tree = DecisionTreeClassifier(max_depth=2).fit(X, Y)
    assert tree.predict(X).tolist() == [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]


def test_max_depth_1_is_a_single_question():
    tree = DecisionTreeClassifier(max_depth=1).fit(X, Y)
    assert tree.feature_.tolist() == [0, -1, -1]
    assert tree.value_.tolist() == [[7, 5], [5, 0], [2, 5]]


def test_entropy_asks_the_same_questions_here():
    tree = DecisionTreeClassifier(criterion="entropy", max_depth=2).fit(X, Y)
    assert tree.feature_.tolist() == [0, -1, 1, -1, -1]


def test_string_labels():
    tree = DecisionTreeClassifier().fit([[1.], [2.], [10.], [11.]], ["low", "low", "high", "high"])
    assert tree.classes_.tolist() == ["high", "low"]
    assert tree.threshold_[0] == 6.0
    assert tree.predict([[1.5], [10.5]]).tolist() == ["low", "high"]


def test_one_class_is_a_single_leaf():
    tree = DecisionTreeClassifier().fit([[1.], [2.], [3.]], [7, 7, 7])
    assert tree.n_nodes_ == 1
    assert tree.value_.tolist() == [[3]]
    assert tree.predict([[9.]]).tolist() == [7]


def test_fit_returns_self():
    tree = DecisionTreeClassifier()
    assert tree.fit(X, Y) is tree


def test_deep_tree_memorises_distinct_points():
    rng = np.random.default_rng(0)
    data = rng.normal(size=(200, 4))
    labels = rng.integers(0, 3, size=200)
    tree = DecisionTreeClassifier(max_depth=60).fit(data, labels)
    assert (tree.predict(data) == labels).all()


def test_min_samples_split_stops_early():
    tree = DecisionTreeClassifier(min_samples_split=13).fit(X, Y)
    assert tree.n_nodes_ == 1
    assert tree.predict(X).tolist() == [0] * 12          # 7 fail, 5 pass


def test_max_features_is_repeatable_with_a_seed():
    rng = np.random.default_rng(1)
    data = rng.normal(size=(100, 6))
    labels = (data[:, 2] > 0).astype(int)
    a = DecisionTreeClassifier(max_features=2, random_state=7).fit(data, labels)
    b = DecisionTreeClassifier(max_features=2, random_state=7).fit(data, labels)
    assert a.feature_.tolist() == b.feature_.tolist()
    assert np.array_equal(a.threshold_, b.threshold_, equal_nan=True)


def test_max_features_larger_than_columns_is_fine():
    tree = DecisionTreeClassifier(max_features=50, max_depth=2).fit(X, Y)
    assert tree.feature_.tolist() == [0, -1, 1, -1, -1]


def test_predict_matches_walking_one_row_at_a_time():
    rng = np.random.default_rng(2)
    data = rng.normal(size=(150, 3))
    labels = rng.integers(0, 2, size=150)
    tree = DecisionTreeClassifier(max_depth=4).fit(data, labels)
    new = rng.normal(size=(40, 3))
    slow = []
    for row in new:
        node = 0
        while tree.feature_[node] >= 0:
            go_left = row[tree.feature_[node]] <= tree.threshold_[node]
            node = tree.left_[node] if go_left else tree.right_[node]
        slow.append(tree.classes_[tree.value_[node].argmax()])
    assert tree.predict(new).tolist() == slow


# --- streaming ---

def test_stream_timeline():
    tree = DecisionTreeClassifier(max_depth=2, buffer_size=8)
    tree.partial_fit(X[:4], Y[:4])
    assert tree.predict([[0, 0, 0]]).tolist() == [1]      # no tree yet: the most common label
    assert not hasattr(tree, "feature_") and tree._n_buffered == 4
    tree.partial_fit(X[4:8], Y[4:8])
    assert tree._n_buffered == 0 and tree.n_nodes_ == 5   # rebuilt, buffer emptied
    assert tree.predict([[0, 0, 0]]).tolist() == [0]
    tree.partial_fit(X[8:], Y[8:])
    assert tree._n_buffered == 4 and tree.n_nodes_ == 5   # still the old tree
    assert tree.predict([[0, 0, 0]]).tolist() == [0]


def test_a_chunk_can_push_the_buffer_past_buffer_size():
    tree = DecisionTreeClassifier(buffer_size=5)
    tree.partial_fit(X[:4], Y[:4])
    assert not hasattr(tree, "feature_")
    tree.partial_fit(X[4:8], Y[4:8])
    assert tree.value_[0].sum() == 8


def test_the_tree_forgets_old_data_after_a_rebuild():
    tree = DecisionTreeClassifier(buffer_size=3)
    tree.partial_fit([[0.], [1.], [2.]], [0, 0, 0])
    tree.partial_fit([[0.], [1.], [2.]], [1, 1, 1])
    assert tree.classes_.tolist() == [1]
    assert tree.predict([[0.]]).tolist() == [1]


def test_tie_before_the_first_tree_goes_to_the_smaller_label():
    tree = DecisionTreeClassifier().partial_fit([[0.]] * 4, ["b", "a", "b", "a"])
    assert tree.predict([[0.]]).tolist() == ["a"]


def test_empty_chunk_is_ignored():
    tree = DecisionTreeClassifier().partial_fit(np.empty((0, 3)), [])
    assert tree._n_buffered == 0
    with pytest.raises(NotFittedError):
        tree.predict([[1, 2, 3]])


def test_fit_empties_the_buffer():
    tree = DecisionTreeClassifier(buffer_size=100)
    tree.partial_fit(X[:4], Y[:4])
    tree.fit(X, Y)
    assert tree._n_buffered == 0


def test_partial_fit_returns_self():
    tree = DecisionTreeClassifier()
    assert tree.partial_fit(X[:2], Y[:2]) is tree


def test_rejected_chunk_changes_nothing():
    tree = DecisionTreeClassifier(buffer_size=100).partial_fit(X[:4], Y[:4])
    with pytest.raises(ValueError):
        tree.partial_fit(np.ones((2, 5)), [0, 1])
    with pytest.raises(ValueError):
        tree.partial_fit(np.array([[1.0, np.nan, 0.0]]), [1])
    assert tree._n_buffered == 4 and len(tree._buffer_y) == 1


# --- errors ---

def test_predict_before_anything_is_seen():
    with pytest.raises(NotFittedError):
        DecisionTreeClassifier().predict([[1, 2, 3]])


def test_predict_with_the_wrong_number_of_columns():
    tree = DecisionTreeClassifier().fit(X, Y)
    with pytest.raises(ValueError, match="3 features"):
        tree.predict([[1, 2]])


def test_nan_in_x_says_to_impute():
    with pytest.raises(ValueError, match="impute"):
        DecisionTreeClassifier().fit([[1.0], [np.nan]], [0, 1])


def test_bad_x():
    with pytest.raises(TypeError):
        DecisionTreeClassifier().fit([["a"], ["b"]], [0, 1])
    with pytest.raises(ValueError):
        DecisionTreeClassifier().fit([1, 2, 3], [0, 1, 0])
    with pytest.raises(ValueError):
        DecisionTreeClassifier().fit(np.empty((0, 2)), [])
    with pytest.raises(ValueError):
        DecisionTreeClassifier().fit([[np.inf], [1.0]], [0, 1])


def test_bad_y():
    with pytest.raises(ValueError, match="same number of rows"):
        DecisionTreeClassifier().fit(X, Y[:-1])
    with pytest.raises(ValueError):
        DecisionTreeClassifier().fit(X, np.column_stack([Y, Y]))
    with pytest.raises(ValueError):
        DecisionTreeClassifier().fit([[1.], [2.]], [0.0, np.nan])
    with pytest.raises(TypeError):
        DecisionTreeClassifier().fit([[1.], [2.]], [None, None])


@pytest.mark.parametrize("settings, error", [
    ({"criterion": "mse"}, ValueError),
    ({"max_depth": 0}, ValueError),
    ({"max_depth": "5"}, TypeError),
    ({"max_depth": True}, TypeError),
    ({"min_samples_split": 1}, ValueError),
    ({"max_features": 0}, ValueError),
    ({"max_features": 1.5}, TypeError),
    ({"buffer_size": 0}, ValueError),
    ({"buffer_size": 2.0}, TypeError),
    ({"random_state": "a"}, TypeError),
])
def test_bad_settings(settings, error):
    with pytest.raises(error):
        DecisionTreeClassifier(**settings)


def test_numpy_integers_are_accepted_as_settings():
    DecisionTreeClassifier(max_depth=np.int64(3), buffer_size=np.int32(10), random_state=np.int64(1))
