from pinn_methods.method_dataclasses import MethodFamily, PINNMethod

# Used in every run, so this family has nothing to switch off to
EARLY_STOPPING = PINNMethod(
    id=1,
    module=(
        "{{label: early_stopping, "
        "method_name: early_stopping, "
        "patience: {patience}, "
        "frac_tol: {frac_tol}}}"
    ),
    defaults={"patience": 5000, "frac_tol": 0.01},
)

EARLY_STOPPING_METHODS = MethodFamily(
    default="early_stopping",
    methods={
        "early_stopping": EARLY_STOPPING,
    },
)
