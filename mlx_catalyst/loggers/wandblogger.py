"""Import WandB Logger integration with MLX Catalyst."""

class WandBLogger:
    def __init__(self, project_name, run_name, config=None):
        """
        Args:
            project_name (str): The name of the W&B project.
            run_name (str): A unique name for the run.
            config (dict, optional): A dictionary of hyperparameters and other configs to log.
        """
        try:
            import wandb
        except ModuleImportError:
            raise ModuleImportError("This function requires wandb installed on your device. Use `pip install wandb` to install the module")
        self.logger = wandb
        self.project_name = project_name
        self.run_name = run_name
        self.config = config

    def init(self):
        """Initialize the W&B run."""
        self.logger.init(project=self.project_name, name=self.run_name, config=self.config)

    def log(self, metrics, step=None):
        """
        Log metrics to W&B.
        
        Args:
            metrics (dict): A dictionary of metrics to log.
            step (int, optional): The training step or epoch number.
        """
        self.logger.log(metrics, step=step)

    def finish(self):
        """Complete the W&B run."""
        self.logger.finish()
