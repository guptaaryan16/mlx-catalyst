import mlx.core as mx
import mlx.nn as nn

class State:
    def __init__(self):
        self.iteration = 0
        self.epoch = 0
        self.params = None
        self.opt_state = None
        self.output = None

# Functional Trainer like pytorch ignite 
# Usable with both MLX and JAX
class Engine:
    def __init__(self, model: nn.Module):
        self.state = None
        self.handlers = {}

    def add_event_handler(self, event_name: str, handler: Callable):
        if event_name not in self.handlers:
            self.handlers[event_name] = []
        self.handlers[event_name].append(handler)

    def fire_event(self, event_name: str, *args, **kwargs):
        if event_name in self.handlers:
            for handler in self.handlers[event_name]:
                handler(*args, **kwargs)

    @staticmethod
    def create_supervised_trainer(model, optimizer, loss_fn):
        def update(params, opt_state, batch):
            def loss_fn(params, x, y):
                y_pred = model.apply(params, x)
                return loss_fn(y_pred, y)

            grad_fn = nn.value_and_grad(loss_fn)
            loss, grads = grad_fn(params, batch["x"], batch["y"])
            updates, new_opt_state = optimizer.update(grads, opt_state)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_opt_state, loss

        def train_step(engine, batch):
            engine.state.params, engine.state.opt_state, loss = update(
                engine.state.params, engine.state.opt_state, batch
            )
            return {"loss": loss}

        return Engine._create_engine(train_step)

    @staticmethod
    def create_supervised_evaluator(model, metrics=None):
        @jax.jit
        def inference(params, batch):
            return model.apply(params, batch["x"])

        def eval_step(engine, batch):
            y_pred = inference(engine.state.params, batch)
            return {"y_pred": y_pred, "y": batch["y"]}

        evaluator = FlexEngine._create_engine(eval_step)

        if metrics:
            for name, metric in metrics.items():
                evaluator.add_event_handler(
                    "iteration_completed", MetricsHandler(metric, name)
                )

        return evaluator

    @staticmethod
    def _create_engine(process_function):
        engine = Engine()

        def run(engine, data):
            engine.fire_event("started")
            engine.state = State()

            for batch in data:
                engine.fire_event("iteration_started")
                engine.state.output = process_function(engine, batch)
                engine.fire_event("iteration_completed")

            engine.fire_event("completed")
            return engine.state

        engine.run = run
        return engine
        
