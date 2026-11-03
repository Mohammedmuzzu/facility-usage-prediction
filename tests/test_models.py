import joblib
import numpy as np


def test_training_serialization_and_train_only_preprocessing(examples, trained):
    X, _, splits, _ = examples
    bundle, root = trained
    loaded = joblib.load(root / "model_bundle.joblib")
    for name, model in loaded["models"].items():
        np.testing.assert_array_equal(model.predict(X.iloc[splits["test"]]), bundle["models"][name].predict(X.iloc[splits["test"]]))
        categories = model.named_steps["features"].named_transformers_["categorical"].categories_
        for column, observed in zip(X.select_dtypes(include="object").columns, categories):
            assert set(observed) == set(X.iloc[splits["train"]][column])
    assert loaded["metadata"]["test_used_for_fit_or_selection"] is False
