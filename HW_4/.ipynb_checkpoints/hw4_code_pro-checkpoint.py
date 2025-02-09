import numpy as np
from collections import Counter
from sklearn.base import BaseEstimator, ClassifierMixin

def find_best_split(feature_vector, target_vector):
    n = len(feature_vector)
    if n == 0:
        return np.array([]), np.array([]), None, None

    sorted_idx = np.argsort(feature_vector)
    sorted_feat = feature_vector[sorted_idx]
    sorted_targ = target_vector[sorted_idx]

    diffs = np.diff(sorted_feat)
    valid = (diffs != 0)
    thresholds = (sorted_feat[1:] + sorted_feat[:-1]) / 2.0
    thresholds = thresholds[valid]

    if len(thresholds) == 0:
        return np.array([]), np.array([]), None, None

    cumsum_ones = np.cumsum(sorted_targ)
    cumsum_zeros = np.cumsum(1 - sorted_targ)

    total_1 = cumsum_ones[-1]
    total_0 = cumsum_zeros[-1]

    idxs_valid = np.nonzero(valid)[0]

    left_1 = cumsum_ones[idxs_valid]
    left_0 = cumsum_zeros[idxs_valid]
    left_size = left_1 + left_0

    right_1 = total_1 - left_1
    right_0 = total_0 - left_0
    right_size = right_1 + right_0

    # -- Важный блок: убираем пороги, которые дают пустое (или нулевое) разбиение
    mask_nonempty = (left_size > 0) & (right_size > 0)
    if not np.any(mask_nonempty):
        return np.array([]), np.array([]), None, None

    thresholds = thresholds[mask_nonempty]
    left_size = left_size[mask_nonempty]
    right_size = right_size[mask_nonempty]
    left_1 = left_1[mask_nonempty]
    right_1 = right_1[mask_nonempty]

    p_left_1 = left_1 / left_size
    H_left = 1 - p_left_1**2 - (1 - p_left_1)**2

    p_right_1 = right_1 / right_size
    H_right = 1 - p_right_1**2 - (1 - p_right_1)**2

    ginis = - (left_size / n) * H_left - (right_size / n) * H_right

    gini_best_val = ginis.max()
    best_idxs = np.where(ginis == gini_best_val)[0]
    best_idx_local = best_idxs[np.argmin(thresholds[best_idxs])]
    threshold_best = thresholds[best_idx_local]
    gini_best = ginis[best_idx_local]

    return thresholds, ginis, threshold_best, gini_best



class DecisionTree(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        feature_types,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1
    ):
        if any(ft not in ("real", "categorical") for ft in feature_types):
            raise ValueError("Unknown feature type.")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node, depth=0):
        # Если все объекты в sub_y одного класса, делаем лист
        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        # Если достигли max_depth или мало объектов -> делаем лист
        if (self._max_depth is not None and depth >= self._max_depth) \
           or (len(sub_y) < self._min_samples_split):
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        feature_best = None
        threshold_best = None
        gini_best = None
        split_mask_best = None
        categories_split_best = None

        n_features = sub_X.shape[1]

        for feature_idx in range(n_features):
            feature_type = self._feature_types[feature_idx]
            feature_vector = sub_X[:, feature_idx]

            # Если признак категориальный, переупорядочим его значения в 0..K-1
            if feature_type == "categorical":
                counts = Counter(feature_vector)
                clicks = Counter(feature_vector[sub_y == 1])
                ratio = {
                    cat_value: clicks[cat_value] / (counts[cat_value] + 1e-9)
                    for cat_value in counts
                }
                sorted_pairs = sorted(ratio.items(), key=lambda x: x[1])
                cat_to_int = {
                    cat_value: new_code
                    for new_code, (cat_value, _) in enumerate(sorted_pairs)
                }
                feature_vector = np.array([cat_to_int[val] for val in feature_vector])
            elif feature_type != "real":
                raise ValueError("Неизвестный тип признака!")

            thresholds, ginis, best_thr, best_gini = find_best_split(feature_vector, sub_y)

            # Смотрим, лучше ли нашёлся сплит здесь, чем предыдущий
            if best_gini is not None and (gini_best is None or best_gini > gini_best):
                gini_best = best_gini
                feature_best = feature_idx
                split_mask_best = (feature_vector < best_thr)

                if feature_type == "real":
                    threshold_best = best_thr
                    categories_split_best = None
                else:
                    # Собираем список категорий, чьи коды < best_thr
                    categories_split_best = [
                        cat for cat, code in cat_to_int.items() if code < best_thr
                    ]
                    threshold_best = None

        # Если не нашёлся ни один «хороший» порог, делаем лист
        if feature_best is None or split_mask_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        # Проверяем, не слишком ли мала выборка в «левом» или «правом» узле
        left_count = np.sum(split_mask_best)
        right_count = len(sub_y) - left_count
        if left_count < self._min_samples_leaf or right_count < self._min_samples_leaf:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"
        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        else:
            node["categories_split"] = categories_split_best

        node["left_child"] = {}
        node["right_child"] = {}

        left_mask = split_mask_best
        right_mask = ~left_mask

        self._fit_node(sub_X[left_mask], sub_y[left_mask], node["left_child"], depth+1)
        self._fit_node(sub_X[right_mask], sub_y[right_mask], node["right_child"], depth+1)

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["class"]

        feature_idx = node["feature_split"]
        feature_type = self._feature_types[feature_idx]

        if feature_type == "real":
            thr = node["threshold"]
            return self._predict_node(x, node["left_child"] if x[feature_idx] < thr else node["right_child"])
        else:
            cats = node["categories_split"]
            return self._predict_node(x, node["left_child"] if x[feature_idx] in cats else node["right_child"])

    def fit(self, X, y):
        self._fit_node(X, y, self._tree, depth=0)
        return self

    def predict(self, X):
        return np.array([self._predict_node(x, self._tree) for x in X])

    # Для совместимости со sklearn:
    def get_params(self, deep=True):
        return {
            "feature_types": self._feature_types,
            "max_depth": self._max_depth,
            "min_samples_split": self._min_samples_split,
            "min_samples_leaf": self._min_samples_leaf,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self