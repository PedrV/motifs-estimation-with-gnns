"""
Unload a PyG loader into a simpler Pytorch loader to allow Ray to seamlessly use it.
"""
import torch.utils.data as data


class Unloader(data.Dataset):
    def __init__(self, pyg_loader, needs_sorting=False):
        super().__init__()

        self.needs_sorting = needs_sorting
        self.pyg_loader = pyg_loader
        self.xs = []
        self.ys = []
        self.edge_indexes = []
        self.batches_id = []

        self._unload()

    def _unload(self):
        for batch in self.pyg_loader:
            if self.needs_sorting:
                batch = batch.sort(sort_by_row=False)

            self.xs.append(batch.x)
            self.ys.append(batch.y)
            self.edge_indexes.append(batch.edge_index)
            self.batches_id.append(batch.batch)

    def __getitem__(self, index):
        return (
            self.edge_indexes[index],
            self.xs[index],
            self.ys[index],
            self.batches_id[index],
        )

    def __len__(self):
        return len(self.xs)


class EarlyStopper:
    def __init__(
        self, grace_period=1, local_patience=1, global_patience=1, min_delta=0
    ):
        self.local_patience = local_patience
        self.global_patience = global_patience
        self.min_delta = min_delta
        self.grace_period = grace_period
        self.counter_since_best = 0
        self.counter_since_last_decrease = 0
        self.min_validation_loss = float("inf")
        self.last_validation_loss = float("inf")

    def early_stop(self, validation_loss, epoch):
        if epoch < self.grace_period:
            return False
        return self._not_globally_improving(
            validation_loss
        )  # or self._not_locally_improving(validation_loss)

    def _not_locally_improving(self, validation_loss):
        """
        Stops if the validation loss does not decrease for `local_patience` iterations.
        """
        if validation_loss < self.last_validation_loss:
            self.counter_since_last_decrease = 0
        elif validation_loss >= (self.last_validation_loss + self.min_delta):
            self.counter_since_last_decrease += 1

        self.last_validation_loss = validation_loss
        if self.counter_since_last_decrease >= self.local_patience:
            return True

        return False

    def _not_globally_improving(self, validation_loss):
        """
        Stops if the validation loss does not go lower than the minimum seen over all
        the training loop for `global_patience` iterations.
        """
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter_since_best = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter_since_best += 1
            if self.counter_since_best >= self.global_patience:
                return True

        return False
