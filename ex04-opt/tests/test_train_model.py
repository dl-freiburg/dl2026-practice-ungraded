import numpy as np

from lib.training import create_and_train_model
from tests.dummy_data import create_dummy_data


def test_train_model():
    """Test model training."""
    # get some data
    x_train, x_val, y_train, y_val = create_dummy_data()

    # train and retrieve results
    results = create_and_train_model("sgd", {
        "lr": 1}, x_train, y_train, x_val, y_val, num_epochs=10)

    # compare with ground truth
    ground_truths = [
        np.array(
            [2.3070044, 2.30584583, 2.30702933, 2.30564029, 2.30462539, 2.30548307, 2.30779576, 2.30506142, 2.30550145,
             2.30483814]),
        np.array([0.086, 0.107, 0.104, 0.095, 0.109, 0.105, 0.102, 0.102, 0.094, 0.089]),
        np.array(
            [2.30287302, 2.30720076, 2.30847024, 2.3112411, 2.30994201, 2.30207942, 2.30203104, 2.29932165, 2.30083581,
             2.29728976]),
        np.array([0.085, 0.08, 0.085, 0.085, 0.085, 0.12, 0.12, 0.12, 0.12, 0.125])
    ]

    for result, ground_truth in zip(results, ground_truths):
        np.testing.assert_allclose(result, ground_truth, rtol=1e-5, err_msg="Model training not implemented correctly.")


if __name__ == '__main__':
    test_train_model()
    print('Test complete.')
