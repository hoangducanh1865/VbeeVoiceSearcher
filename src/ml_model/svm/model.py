import joblib
from pathlib import Path

from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC

from src.ml_model.base import BaseMLClassifier
from src.ml_model.features import build_vectorizer


class SVMClassifier(BaseMLClassifier):
    name = "svm"

    def train(self, texts: list[str], axes: list[dict], labels: dict[str, list]) -> None:
        self.vec = build_vectorizer()
        X = self.vec.fit_transform(texts)
        self.classifiers: dict = {}
        self.mlbs: dict = {}

        for axis in axes:
            axis_en  = axis["metadata_en"]
            is_multi = axis["loai_nhan"] == "Đa nhãn"
            y_raw    = labels[axis_en]

            if is_multi:
                mlb = MultiLabelBinarizer(classes=axis["cac_gia_tri"])
                Y   = mlb.fit_transform(y_raw)
                clf = OneVsRestClassifier(LinearSVC(max_iter=2000))
                clf.fit(X, Y)
                self.classifiers[axis_en] = clf
                self.mlbs[axis_en]        = mlb
            else:
                clf = LinearSVC(max_iter=2000)
                clf.fit(X, y_raw)
                self.classifiers[axis_en] = clf

    def predict_one(self, text: str, axes: list[dict]) -> dict:
        X = self.vec.transform([text])
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
            {"vec": self.vec, "classifiers": self.classifiers, "mlbs": self.mlbs},
            model_dir / "model.pkl",
        )

    @classmethod
    def load(cls, model_dir: Path) -> "SVMClassifier":
        obj = cls.__new__(cls)
        d = joblib.load(model_dir / "model.pkl")
        obj.vec         = d["vec"]
        obj.classifiers = d["classifiers"]
        obj.mlbs        = d["mlbs"]
        return obj
