import os
from unittest import TestCase, skip, main

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from src.opengood.py_ml_plot import setup_classification_plot


class TestClassificationPlot(TestCase):
    def test_logistic_regression_setup_classification_plot_with_shaded_regions(self):
        """Should set up a Logistic regression classification plot with classified regions shaded."""
        resource_path = os.path.join(
            os.path.dirname(__file__), "../resources", "data.csv"
        )
        dataset = pd.read_csv(resource_path)

        x = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, -1].values

        x_train, _, y_train, _ = train_test_split(x, y, test_size=0.2, random_state=0)

        sc = StandardScaler()
        x_train = sc.fit_transform(x_train)

        classifier = LogisticRegression(random_state=0)
        classifier.fit(x_train, y_train)

        result = setup_classification_plot(
            x=x_train,
            y=y_train,
            cmap=ListedColormap(("salmon", "dodgerblue")),
            title="Logistic Regression",
            x_label="Age",
            y_label="Estimated Salary",
            meshgrid={
                0: {"min": 10, "max": 10, "step": 0.25},
                1: {"min": 1000, "max": 1000, "step": 0.25},
            },
            feature_scale=lambda x_set, y_set: (sc.inverse_transform(x_set), y_set),
            predict=lambda x1, x2: classifier.predict(
                sc.transform(np.array([x1.ravel(), x2.ravel()]).T)
            ).reshape(x1.shape),
        )

        assert result is True, "Classification plot should be set up correctly"
        assert plt.gcf() is not None, "Classification plot should be created"

        axes = plt.gca()
        assert len(axes.collections) > 0, (
            "Classification plot should contain shaded regions"
        )
        assert axes.get_title() == "Logistic Regression", (
            "Classification plot title should be set"
        )
        assert axes.get_xlabel() == "Age", (
            "Classification plot x-axis label should be set"
        )
        assert axes.get_ylabel() == "Estimated Salary", (
            "Classification plot y-axis label should be set"
        )
        assert len(axes.collections) > 0 or len(axes.lines) > 0, (
            "Classification plot should contain either decision boundary lines or shaded regions"
        )

        plt.show()
        plt.close()

        self.assertIsNotNone(result)

    @skip(reason="Long running test, only used for local tests")
    def test_k_nearest_neighbor_setup_classification_plot_with_shaded_regions(self):
        """Should set up a K-Nearest Neighbor classification plot with classified regions shaded."""
        resource_path = os.path.join(
            os.path.dirname(__file__), "../resources", "data.csv"
        )
        dataset = pd.read_csv(resource_path)

        x = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, -1].values

        x_train, _, y_train, _ = train_test_split(x, y, test_size=0.2, random_state=0)

        sc = StandardScaler()
        x_train = sc.fit_transform(x_train)

        classifier = KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2)
        classifier.fit(x_train, y_train)

        result = setup_classification_plot(
            x=x_train,
            y=y_train,
            cmap=ListedColormap(("salmon", "dodgerblue")),
            title="K-Nearest Neighbor (K-NN)",
            x_label="Age",
            y_label="Estimated Salary",
            meshgrid={
                0: {"min": 10, "max": 10, "step": 0.25},
                1: {"min": 1000, "max": 1000, "step": 0.25},
            },
            feature_scale=lambda x_set, y_set: (sc.inverse_transform(x_set), y_set),
            predict=lambda x1, x2: classifier.predict(
                sc.transform(np.array([x1.ravel(), x2.ravel()]).T)
            ).reshape(x1.shape),
        )

        assert result is True, "Classification plot should be set up correctly"
        assert plt.gcf() is not None, "Classification plot should be created"

        axes = plt.gca()
        assert len(axes.collections) > 0, (
            "Classification plot should contain shaded regions"
        )
        assert axes.get_title() == "K-Nearest Neighbor (K-NN)", (
            "Classification plot title should be set"
        )
        assert axes.get_xlabel() == "Age", (
            "Classification plot x-axis label should be set"
        )
        assert axes.get_ylabel() == "Estimated Salary", (
            "Classification plot y-axis label should be set"
        )
        assert len(axes.collections) > 0 or len(axes.lines) > 0, (
            "Classification plot should contain either decision boundary lines or shaded regions"
        )

        plt.show()
        plt.close()

        self.assertIsNotNone(result)


if __name__ == "__main__":
    main()
