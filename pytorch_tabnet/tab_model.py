import torch
import numpy as np
from scipy.special import softmax
from pytorch_tabnet.utils import (
    create_predict_dataloader,
    filter_weights,
    get_xp,
    to_cpu,
    CUPY_AVAILABLE,
)
from pytorch_tabnet.abstract_model import TabModel
from pytorch_tabnet.multiclass_utils import infer_output_dim, check_output_dim

try:
    import cupy as cp
except Exception:
    cp = None


class TabNetClassifier(TabModel):
    def __post_init__(self):
        super(TabNetClassifier, self).__post_init__()
        self._task = 'classification'
        self._default_loss = torch.nn.functional.cross_entropy
        self._default_metric = 'accuracy'

    def weight_updater(self, weights):
        """
        Updates weights dictionary according to target_mapper.

        Parameters
        ----------
        weights : bool or dict
            Given weights for balancing training.

        Returns
        -------
        bool or dict
            Same bool if weights are bool, updated dict otherwise.

        """
        if isinstance(weights, int):
            return weights
        elif isinstance(weights, dict):
            return {self.target_mapper[key]: value for key, value in weights.items()}
        else:
            return weights

    def prepare_target(self, y):
        return np.vectorize(self.target_mapper.get)(y)

    def compute_loss(self, y_pred, y_true):
        return self.loss_fn(y_pred, y_true.long())

    def update_fit_params(
        self,
        X_train,
        y_train,
        eval_set,
        weights,
    ):
        output_dim, train_labels = infer_output_dim(y_train)
        for X, y in eval_set:
            check_output_dim(train_labels, y)
        self.output_dim = output_dim
        self._default_metric = ('auc' if self.output_dim == 2 else 'accuracy')
        self.classes_ = train_labels
        self.target_mapper = {
            class_label: index for index, class_label in enumerate(self.classes_)
        }
        self.preds_mapper = {
            str(index): class_label for index, class_label in enumerate(self.classes_)
        }
        self.updated_weights = self.weight_updater(weights)

    def stack_batches(self, list_y_true, list_y_score):
        if torch.is_tensor(list_y_true[0]):
            y_true = torch.cat(list_y_true, dim=0)
            y_score = torch.cat(list_y_score, dim=0)
            y_score = torch.softmax(y_score, dim=1)
            return y_true, y_score
        xp = get_xp()
        y_true = xp.hstack(list_y_true)
        y_score = xp.vstack(list_y_score)
        # softmax requires CPU arrays
        y_score = softmax(to_cpu(y_score), axis=1)
        if CUPY_AVAILABLE:
            y_score = xp.asarray(y_score)
        return y_true, y_score

    def predict_func(self, outputs):
        xp = get_xp()
        outputs = xp.argmax(outputs, axis=1)
        return np.vectorize(self.preds_mapper.get)(to_cpu(outputs).astype(str))

    def predict_proba(self, X):
        """
        Make predictions for classification on a batch (valid)

        Parameters
        ----------
        X : a :tensor: `torch.Tensor` or matrix: `scipy.sparse.csr_matrix`
            Input data

        Returns
        -------
        res : np.ndarray

        """
        self.network.eval()

        dataloader = create_predict_dataloader(
            X,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            device=self.device,
        )

        xp = get_xp()
        results = []
        with torch.inference_mode():
            for batch_nb, data in enumerate(dataloader):
                data = data.to(self.device, non_blocking=True)
                if data.dtype != torch.float32:
                    data = data.float()
                with self._autocast_context():
                    output, _ = self.network(data)
                softmax_output = torch.nn.Softmax(dim=1)(output)
                if CUPY_AVAILABLE and self.device.type == 'cuda':
                    predictions = cp.from_dlpack(softmax_output.detach())
                else:
                    predictions = softmax_output.cpu().detach().numpy()
                results.append(predictions)
        res = xp.vstack(results)
        return res


class TabNetRegressor(TabModel):
    def __post_init__(self):
        super(TabNetRegressor, self).__post_init__()
        self._task = 'regression'
        self._default_loss = torch.nn.functional.mse_loss
        self._default_metric = 'mse'

    def prepare_target(self, y):
        return y

    def compute_loss(self, y_pred, y_true):
        return self.loss_fn(y_pred, y_true)

    def update_fit_params(
        self,
        X_train,
        y_train,
        eval_set,
        weights
    ):
        if len(y_train.shape) != 2:
            msg = "Targets should be 2D : (n_samples, n_regression) " + \
                  f"but y_train.shape={y_train.shape} given.\n" + \
                  "Use reshape(-1, 1) for single regression."
            raise ValueError(msg)
        self.output_dim = y_train.shape[1]
        self.preds_mapper = None

        self.updated_weights = weights
        filter_weights(self.updated_weights)

    def predict_func(self, outputs):
        return outputs

    def stack_batches(self, list_y_true, list_y_score):
        if torch.is_tensor(list_y_true[0]):
            y_true = torch.cat(list_y_true, dim=0)
            y_score = torch.cat(list_y_score, dim=0)
            return y_true, y_score
        xp = get_xp()
        y_true = xp.vstack(list_y_true)
        y_score = xp.vstack(list_y_score)
        return y_true, y_score
