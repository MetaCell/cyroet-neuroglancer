"""Shader functions in GLSL for rendering in Neuroglancer."""


def build_shader(shader_parts: list[str]) -> str:
    """Builds a shader from a list of parts."""
    return "\n".join(shader_parts)


def _build_invlerp(
    name: str, contrast_limits: tuple[float, float], window_limits: tuple[float, float]
) -> str:
    return f"#uicontrol invlerp {name}(range=[{contrast_limits[0]}, {contrast_limits[1]}], window=[{window_limits[0]}, {window_limits[1]}], clamp=true)"


def _build_invertable_invlerp_getter(name: str):
    checkbox_part = f"#uicontrol bool invert_{name} checkbox(default=false)"
    data_value_getter = (
        f"float {name}_get() {{ return invert_{name} ? 1.0 - {name}() : {name}(); }}"
    )
    return checkbox_part, data_value_getter


def _build_volume_rendering_switch(if_vr: list[str], if_not_vr: list[str]):
    indented_if_vr = [f"    {line}" for line in if_vr]
    indented_if_not_vr = [f"    {line}" for line in if_not_vr]
    return [
        "  if (VOLUME_RENDERING) {",
        *indented_if_vr,
        "  } else {",
        *indented_if_not_vr,
        "  }",
    ]


def get_default_image_shader(
    contrast_limits: tuple[float, float], window_limits: tuple[float, float]
):
    """Generates a default image shader with contrast and window controls."""
    contrast_name = "contrast"
    threed_contrast_name = "contrast3D"
    shader_parts = [
        _build_invlerp(contrast_name, contrast_limits, window_limits),
        _build_invlerp(threed_contrast_name, contrast_limits, window_limits),
        *_build_invertable_invlerp_getter(contrast_name),
        *_build_invertable_invlerp_getter(threed_contrast_name),
        "",
        "void main() {",
        "  float outputValue;",
        *_build_volume_rendering_switch(
            [f"outputValue = {threed_contrast_name}_get();"],
            [f"outputValue = {contrast_name}_get();"],
        ),
        "  emitGrayscale(outputValue);",
        "  emitIntensity(outputValue);",
        "}",
    ]
    return build_shader(shader_parts)


if __name__ == "__main__":
    print(get_default_image_shader((0.0, 1.0), (0.0, 1.0)))
