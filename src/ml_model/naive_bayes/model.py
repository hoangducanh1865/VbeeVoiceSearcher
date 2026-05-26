import joblib
from pathlib import Path

from sklearn.naive_bayes import ComplementNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from src.ml_model.base import BaseMLClassifier
from src.ml_model.features import build_vectorizer


class NaiveBayesClassifier(BaseMLClassifier):
    name = "naive_bayes"

    def train(self, texts: list[str], axes: list[dict], labels: dict[str, list]) -> None:
        self.vec = build_vectorizer()
        X = self.vec.fit_transform(texts)
        # TF-IDF with sublinear_tf may produce small negatives after scaling;
        # ComplementNB requires non-negative input — shift to ensure X >= 0.
        self._x_min = X.min()
        if self._x_min < 0:
            X = X - self._x_min

        self.classifiers: dict = {}
        self.mlbs: dict = {}

        for axis in axes:
            axis_en  = axis["metadata_en"]
            is_multi = axis["loai_nhan"] == "Đa nhãn"
            y_raw    = labels[axis_en]

            if is_multi:
                mlb = MultiLabelBinarizer(classes=axis["cac_gia_tri"])
                Y   = mlb.fit_transform(y_raw)
                clf = OneVsRestClassifier(ComplementNB(alpha=0.1))
                clf.fit(X, Y)
                self.classifiers[axis_en] = clf
                self.mlbs[axis_en]        = mlb
            else:
                clf = ComplementNB(alpha=0.1)
                clf.fit(X, y_raw)
                self.classifiers[axis_en] = clf

    def _transform(self, text: str):
        X = self.vec.transform([text])
        if self._x_min < 0:
            X = X - self._x_min
        return X

    def predict_one(self, text: str, axes: list[dict]) -> dict:
        X = self._transform(text)
        result = {}
        for axis in axes:
            axis_en  = axis["metadata_en"]
            is_multi = axis["loai_nhan"] == "Đa nhãn"
            pred     = self.classifiers[axis_en].predict(X)
            if is_multi:
                labels = list(self.mlbs[axis_en].inverse_transform(pred)[0])
                result[axis_en] = labels if labels else [axis["cac_gia_tri"][0]]
            else:
                result[axis_en] = pred[0]
        return result

    def save(self, model_dir: Path) -> None:
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "vec": self.vec,
                "classifiers": self.classifiers,
                "mlbs": self.mlbs,
                "x_min": self._x_min,
            },
            model_dir / "model.pkl",
        )

    @classmethod
    def load(cls, model_dir: Path) -> "NaiveBayesClassifier":
        obj = cls.__new__(cls)
        d = joblib.load(model_dir / "model.pkl")
        obj.vec         = d["vec"]
        obj.classifiers = d["classifiers"]
        obj.mlbs        = d["mlbs"]
        obj._x_min      = d["x_min"]
        return obj
