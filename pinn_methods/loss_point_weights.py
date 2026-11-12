from pinn_methods.method_dataclasses import MethodFamily, PINNMethod

NO_LOSS_POINT_WEIGHTS = PINNMethod(
    id=0,
    extras={"ADAPTIVE_POINT_WEIGHTS": "false"},
)

RBA = PINNMethod(
    id=1,
    module=(
        "{{label: rba, "
        "method_name: residual_based_attention_a24, "
        "gamma: {gamma}, "
        "eta: {eta}}}"
    ),
    defaults={"gamma": 0.999, "eta": 0.01},
    extras={"ADAPTIVE_POINT_WEIGHTS": "true"},
)

LOSS_POINT_WEIGHTS = MethodFamily(
    default="no_loss_point_weights",
    methods={
        "no_loss_point_weights": NO_LOSS_POINT_WEIGHTS,
        "rba": RBA,
    },
)
