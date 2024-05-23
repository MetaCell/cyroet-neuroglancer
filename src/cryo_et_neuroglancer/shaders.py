"""Shader functions in GLSL for rendering in Neuroglancer."""


def build_shader(shader_parts: list[str]) -> str:
    """Builds a shader from a list of parts."""
    return "\n".join(shader_parts)


def build_invlerp(
    name: str, contrast_limits: tuple[float, float], window_limits: tuple[float, float]
) -> str:
    """Builds an invlerp function for a shader."""
    return f"#uicontrol invlerp {name}(range=[{contrast_limits[0]}, {contrast_limits[1]}], window=[{window_limits[0]}, {window_limits[1]})"


def get_default_image_shader(
    contrast_limits: tuple[float, float], window_limits: tuple[float, float]
):
    shader_parts = [
        build_invlerp("contrast", contrast_limits, window_limits),
        "void main() {",
        "  emitGrayscale(contrast());",
        "}",
    ]
    return build_shader(shader_parts)
