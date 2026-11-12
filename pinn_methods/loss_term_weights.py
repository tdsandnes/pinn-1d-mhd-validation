from pinn_methods.method_dataclasses import MethodFamily, PINNMethod

NO_LOSS_TERM_WEIGHTS = PINNMethod(
    id=0,
    extras={"WEIGHT_METHOD": "constant"},
)

# loss_function is the same as the one the loss terms use, so it is not set here.
W23 = PINNMethod(
    id=1,
    module=(
        "{{label: w23, "
        "method_name: adaptive_weights_w23, "
        "alpha: {alpha}, "
        "update_iter_frequency: {update_iter_frequency}, "
        "loss_function: {loss_function}}}"
    ),
    defaults={"alpha": 0.9, "update_iter_frequency": 1000},
    extras={"WEIGHT_METHOD": "adaptive_w23"},
)

LOSS_TERM_WEIGHTS = MethodFamily(
    default="no_loss_term_weights",
    methods={
        "no_loss_term_weights": NO_LOSS_TERM_WEIGHTS,
        "w23": W23,
    },
)
