from collections.abc import Iterable
from typing import Any

from pinn_methods.methods import METHOD_FAMILIES, METHODS


def selected_methods(names: Iterable[str]) -> dict[str, str]:
    """The method each family uses, defaulting to the family's own default."""
    chosen = set(names)

    unknown = chosen - set(METHODS)
    if unknown:
        raise SystemExit(
            f"Unknown method(s) {', '.join(sorted(unknown))}. "
            f"Use any of {', '.join(METHODS)}."
        )

    selected = {}
    for family_name, family in METHOD_FAMILIES.items():
        picked = chosen & set(family.methods)
        if len(picked) > 1:
            raise SystemExit(
                f"Cannot use {', '.join(sorted(picked))} together. "
                f"Pick at most one of {', '.join(family.methods)}."
            )
        selected[family_name] = next(iter(picked), family.default)
    return selected


def method_str(names: Iterable[str]) -> str:
    """One choice of methods as the id of each family's method, e.g. 111101."""
    selected = selected_methods(names)
    return "".join(
        str(METHOD_FAMILIES[family_name].methods[method_name].id)
        for family_name, method_name in selected.items()
    )


def method_settings(
    names: Iterable[str],
    methods_config: dict[str, dict[str, Any]],
    **template_fields: Any,
) -> dict[str, str]:
    """The template fields one choice of training methods produces."""
    selected = selected_methods(names)

    unknown = methods_config.keys() - set(METHODS)
    if unknown:
        raise SystemExit(
            f"Cannot configure unknown method(s) {', '.join(sorted(unknown))}. "
            f"Use any of {', '.join(METHODS)}."
        )

    # loss function depends on combination of weighting methods used
    if selected["loss_point_weights"] == "no_loss_point_weights":
        loss_function = "mse"
    elif selected["loss_term_weights"] == "no_loss_term_weights":
        loss_function = "weighted_point_avg_mse"
    else:
        loss_function = "weighted_mse"

    fields: dict[str, str] = {"LOSS_FUNCTION": loss_function}
    trainer_modules = []

    for family_name, family in METHOD_FAMILIES.items():
        method_name = selected[family_name]
        method = family.methods[method_name]
        block = method.format_config(
            method_name,
            methods_config.get(method_name, {}),
            loss_function=loss_function,
            **template_fields,
        )

        # A method's extras are template fields it sets outright
        fields.update(method.extras)

        if family.template_field is not None:
            fields[family.template_field] = block
        elif block:
            trainer_modules.append(block)

    fields["TRAINER_MODULES"] = "[" + ", ".join(trainer_modules) + "]"
    return fields
