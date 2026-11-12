from pinn_methods.architecture import ARCHITECTURES
from pinn_methods.early_stopping import EARLY_STOPPING_METHODS
from pinn_methods.loss_point_weights import LOSS_POINT_WEIGHTS
from pinn_methods.loss_term_weights import LOSS_TERM_WEIGHTS
from pinn_methods.optimizer import OPTIMIZERS

# Dict of all method families
METHOD_FAMILIES = {
    "optimizer": OPTIMIZERS,
    "architecture": ARCHITECTURES,
    "loss_term_weights": LOSS_TERM_WEIGHTS,
    "loss_point_weights": LOSS_POINT_WEIGHTS,
    "early_stopping": EARLY_STOPPING_METHODS,
}

# Tuple of all methods
METHODS = tuple(name for family in METHOD_FAMILIES.values() for name in family.methods)
