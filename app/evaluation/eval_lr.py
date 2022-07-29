from pathlib import Path
from typing import List

import click
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold


class LogisticRegressionEvaluation:
    def __init__(self, train_data_path: str, output_dir: str) -> None:
        self._train_data_path = Path(train_data_path)
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        for file_path in [self._train_data_path]:
            if not file_path.exists():
                raise ValueError(f"File '{file_path}' does not exist.")

    def run(self) -> None:
        print(f"Loading training data from '{self._train_data_path}'...")
        train_data = np.load(self._train_data_path)

        X: np.ndarray = train_data["features"]
        y: np.ndarray = train_data["labels"]
        class_names = train_data["class_names"]

        random_seed = 42  # TODO: make this a parameter
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)

        y_predictions = np.zeros_like(y)
        print("Running k-fold cross-validation...")
        for train_idx, val_idx in kfold.split(X=X, y=y):
            X_train_i, X_val_i = X[train_idx], X[val_idx]
            y_train_i = y[train_idx]

            print(f"Fitting model on {len(X_train_i)} training examples...")
            lr = LogisticRegression(max_iter=4000)
            lr.fit(X_train_i, y_train_i)

            print(f"Predicting on {len(X_val_i)} validation examples...")
            y_pred_i = lr.predict(X_val_i)

            # record results
            y_predictions[val_idx] = y_pred_i
        self._generate_classification_report(
            y_true=y, y_pred=y_predictions, class_names=class_names
        )
        self._generate_confusion_matrix(
            y_true=y, y_pred=y_predictions, class_names=class_names
        )

    def _generate_confusion_matrix(
        self, y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str]
    ) -> None:
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="g",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
        )
        ax.set_title("Confusion Matrix")
        fig.savefig(self._output_dir / "confusion_matrix.png", bbox_inches="tight")

    def _generate_classification_report(
        self, y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str]
    ) -> None:
        report = classification_report(
            y_true, y_pred, target_names=class_names, output_dict=True
        )
        clsf_report = pd.DataFrame(report).transpose()
        clsf_report.to_markdown(
            self._output_dir / "classification_report.md", index=True
        )


@click.command(help="Evaluation.")
@click.option(
    "-t",
    "--train-data-path",
    type=click.STRING,
    required=False,
    default="data/input/processed/agnews-train-avg-word-model.npz",
)
@click.option(
    "-o",
    "--output-dir",
    type=click.STRING,
    required=False,
    default="data/output/",
)
def main(
    train_data_path: str,
    output_dir: str,
):
    LogisticRegressionEvaluation(
        train_data_path=train_data_path,
        output_dir=output_dir,
    ).run()


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
