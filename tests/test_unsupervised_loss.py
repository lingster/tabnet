import numpy as np
import torch
import pytest
from pytorch_tabnet.metrics import UnsupervisedLoss, UnsupervisedLossNumpy


@pytest.mark.parametrize(
    "y_pred,embedded_x,obf_vars",
    [
        (
            np.random.uniform(low=-2, high=2, size=(20, 100)),
            np.random.uniform(low=-2, high=2, size=(20, 100)),
            np.random.choice([0, 1], size=(20, 100), replace=True)
        ),
        (
            np.random.uniform(low=-2, high=2, size=(30, 50)),
            np.ones((30, 50)),
            np.random.choice([0, 1], size=(30, 50), replace=True)
        )
    ]
)
def test_equal_losses(y_pred, embedded_x, obf_vars):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for GPU-only tests.")
    device = torch.device("cuda")
    numpy_loss = UnsupervisedLossNumpy(
        y_pred=y_pred,
        embedded_x=embedded_x,
        obf_vars=obf_vars
    )

    torch_loss = UnsupervisedLoss(
        y_pred=torch.tensor(y_pred, dtype=torch.float64, device=device),
        embedded_x=torch.tensor(embedded_x, dtype=torch.float64, device=device),
        obf_vars=torch.tensor(obf_vars, dtype=torch.float64, device=device)
    )

    assert np.isclose(numpy_loss, torch_loss.detach().cpu().numpy(), rtol=1e-5, atol=1e-7)
