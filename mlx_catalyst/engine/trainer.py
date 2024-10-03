import time
from mlx import nn
import mlx.core as mx
from functools import partial

class Trainer:

    def __init__(
        self,
        model: nn.Module,
        optimizer,
        loss_fn,
        train_loader,
        dataset_keys = ["images", "labels"],
        val_loader=None,
        epochs=10,
        log_every=10,
        logger=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.dataset_keys = dataset_keys
        self.val_loader = val_loader
        self.epochs = epochs
        self.log_every = log_every
        self.logger = logger

    def train_epoch(self):
        state = [self.model.state, self.optimizer.state]

        def train_step(model: nn.Module, input: mx.array, target: mx.array):
            output = model(input)
            loss = mx.mean(self.loss_fn(output, target))
            acc = mx.mean(mx.argmax(output, axis=1) == target)
            return loss, acc

        @partial(mx.compile, inputs=state, outputs=state)
        def step(inp, tgt):
            train_step_fn = nn.value_and_grad(self.model, train_step)
            (loss, acc), grads = train_step_fn(self.model, inp, tgt)
            self.optimizer.update(self.model, grads)
            return loss, acc

        total_loss = 0

        for i, batch in enumerate(self.train_loader):
            # Forward pass
            x = mx.array(batch[self.dataset_keys[0]])
            y = mx.array(batch[self.dataset_keys[1]])
            loss, acc = step(x, y)
            mx.eval(state)
            total_loss += loss.item()

            if i % self.log_every == 0:
                print(f"Batch {i} - Loss: {loss.item()}")

        return total_loss / len(self.train_loader)

    def val_epoch(self):

        def eval_fn(model, inp, tgt):
            return mx.mean(mx.argmax(model(inp), axis=1) == tgt)

        accs = []
        for _, (x, y) in enumerate(self.val_loader):
            acc = eval_fn(self.model, x, y)
            acc_value = acc.item()
            accs.append(acc_value)
        mean_acc = mx.mean(mx.array(accs))
        return mean_acc

    def train(self):
        for epoch in range(self.epochs):
            start_time = time.time()

            # Train
            train_loss = self.train_epoch()
            print(f"Epoch {epoch + 1}/{self.epochs} - Train Loss: {train_loss:.4f}")

            # Validate (if validation loader is provided)
            if self.val_loader:
                val_loss = self.validate_epoch()
                print(f"Epoch {epoch + 1}/{self.epochs} - Validation Loss: {val_loss:.4f}")

            if self.logger and epoch % self.log_every == 0:
                self.logger.log({"Train_loss": train_loss, "Validation_loss": val_loss})

            print(f"Epoch {epoch + 1} completed in {time.time() - start_time:.2f}s")
