from cryo_et_neuroglancer.shaders import get_default_image_shader


def test_get_default_image_shader():
    contrast_limits = (0.0, 1.0)
    window_limits = (0.0, 1.0)
    expected_shader = """
#uicontrol invlerp contrast(range=[0.0, 1.0], window=[0.0, 1.0])
#uicontrol bool invert_contrast checkbox(default=false)
float contrast_get() { return invert_contrast ? 1.0 - contrast() : contrast(); }
#uicontrol invlerp contrast3D(range=[0.0, 1.0], window=[0.0, 1.0])
#uicontrol bool invert_contrast3D checkbox(default=false)
float contrast3D_get() { return invert_contrast3D ? 1.0 - contrast3D() : contrast3D(); }

void main() {
  float outputValue;
  if (VOLUME_RENDERING) {
    outputValue = contrast3D_get();
    emitIntensity(outputValue);
  } else {
    outputValue = contrast_get();
  }
  emitGrayscale(outputValue);
}
"""
    actual_shader = get_default_image_shader(contrast_limits, window_limits)
    assert actual_shader.strip() == expected_shader.strip()
