from pinn_methods.method_dataclasses import MethodFamily, PINNMethod

MLP_TANH = PINNMethod(
    id=0,
    module=(
        "{{type: feedforward, "
        "weight_norm: {weight_norm}, "
        "layer_types: all_linear, "
        "activation_functions: all_tanh, "
        "init_weights: {init_weights}, "
        "init_biases: {init_biases}, "
        "input_transformation: {input_transformation}, "
        "hidden_layer_widths: {hidden_layer_widths}}}"
    ),
    defaults={
        "weight_norm": "false",
        "init_weights": "all_xavier",
        "init_biases": "all_zeros",
        "input_transformation": "scale_linear",
    },
)

SIREN = PINNMethod(
    id=1,
    module=(
        "{{type: feedforward, "
        "weight_norm: {weight_norm}, "
        "layer_types: siren, "
        "activation_functions: siren, "
        "init_weights: siren, "
        "init_biases: {init_biases}, "
        "input_transformation: {input_transformation}, "
        "hidden_layer_widths: {hidden_layer_widths}}}"
    ),
    defaults={
        "weight_norm": "false",
        "init_biases": "all_zeros",
        "input_transformation": "scale_linear",
    },
)

ARCHITECTURES = MethodFamily(
    default="mlp_tanh",
    template_field="NN_ARCHITECTURE",
    methods={
        "mlp_tanh": MLP_TANH,
        "siren": SIREN,
    },
)
