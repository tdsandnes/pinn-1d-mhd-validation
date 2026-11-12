from pinn_methods.method_dataclasses import MethodFamily, PINNMethod

ADAM = PINNMethod(
    id=0,
    module="{{optimizer: adam, learning_rate: {learning_rate}}}",
    defaults={"learning_rate": 0.001},
)

SOAP = PINNMethod(
    id=1,
    module=(
        "{{optimizer: soap, "
        "learning_rate: {learning_rate}, "
        "betas: {betas}, "
        "weight_decay: {weight_decay}, "
        "precondition_frequency: {precondition_frequency}}}"
    ),
    defaults={
        "learning_rate": 0.001,
        "betas": [0.99, 0.99],
        "weight_decay": 0.0,
        "precondition_frequency": 1,
    },
)

OPTIMIZERS = MethodFamily(
    default="adam",
    template_field="OPTIMIZER",
    methods={
        "adam": ADAM,
        "soap": SOAP,
    },
)
