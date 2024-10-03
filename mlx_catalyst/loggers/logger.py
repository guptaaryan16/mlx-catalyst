

class Logger:
    def __init__(self):
        self.metrics = {}

    def log(self, key, value, step=None):
        if key not in self.metrics:
            self.metrics[key] = []
        self.metrics[key].append((step, value))

    def display_metrics(self):
        for key, values in self.metrics.items():
            print(f"Metrics for {key}:")
            for step, value in values:
                print(f"Step {step}: {value}")

    def log_epoch(self, epoch, train_loss, val_loss=None):
        print(f"Epoch {epoch}:")
        print(f"Train Loss: {train_loss}")
        if val_loss:
            print(f"Validation Loss: {val_loss}")
