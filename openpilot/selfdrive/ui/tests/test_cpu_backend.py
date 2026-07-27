import os
import subprocess
import sys
import textwrap

from openpilot.common.basedir import BASEDIR


def test_cpu_backend_api():
  code = textwrap.dedent("""
    import io
    import os
    import struct
    from PIL import Image
    from openpilot.system.ui.lib import raylib as rl
    from openpilot.system.ui.lib.cpu_backend import framebuffer, state

    rl.init_window(64, 48, "cpu-test")
    rl.clear_background(rl.BLACK)
    rl.begin_scissor_mode(8, 8, 16, 16)
    rl.draw_rectangle(0, 0, 64, 48, rl.RED)
    rl.end_scissor_mode()
    assert tuple(framebuffer()[4, 4]) == (0, 0, 0, 255)
    assert tuple(framebuffer()[12, 12]) == tuple(rl.RED)

    target = rl.load_render_texture(12, 10)
    rl.begin_texture_mode(target)
    rl.clear_background(rl.GREEN)
    rl.end_texture_mode()
    image = rl.load_image_from_texture(target.texture)
    assert (image.width, image.height) == (12, 10)
    rl.unload_image(image)
    rl.unload_render_texture(target)

    encoded = io.BytesIO()
    Image.new("RGBA", (3, 2), (10, 20, 30, 255)).save(encoded, "PNG")
    image = rl.load_image_from_memory(".png", encoded.getvalue(), len(encoded.getvalue()))
    texture = rl.load_texture_from_image(image)
    rl.draw_texture_ex(texture, (30, 20), 0, 2, rl.WHITE)
    rl.unload_texture(texture)
    rl.unload_image(image)

    rl.draw_circle_gradient((50, 12), 8, rl.WHITE, rl.BLANK)
    rl.draw_ring((50, 35), 4, 8, 90, 180, 16, rl.WHITE)
    rl.draw_line_ex((2, 45), (20, 30), 3, rl.WHITE)
    rl.draw_spline_linear([(1, 1), (10, 4), (20, 1)], 3, 2, rl.WHITE)

    # Full and quarter rings use the optimized horizontal-span rasterizer.
    # Check every pixel against the same inclusive radius/axis semantics as
    # the generic arc path.
    ring_cases = (
      (0, 360, lambda x, y: True),
      (180, 270, lambda x, y: x <= 0 and y <= 0),
      (270, 360, lambda x, y: x >= 0 and y <= 0),
      (0, 90, lambda x, y: x >= 0 and y >= 0),
      (90, 180, lambda x, y: x <= 0 and y >= 0),
    )
    center_x, center_y, inner, outer = 32, 24, 5, 10
    for start, end, in_quadrant in ring_cases:
      rl.clear_background(rl.BLANK)
      rl.draw_ring((center_x, center_y), inner, outer, start, end, 16, rl.WHITE)
      alpha = framebuffer()[:, :, 3]
      for y in range(48):
        for x in range(64):
          dx, dy = x - center_x, y - center_y
          expected = inner * inner <= dx * dx + dy * dy <= outer * outer and in_quadrant(dx, dy)
          assert (alpha[y, x] == 255) == expected, (start, end, x, y)

    burn_shader = rl.load_shader_from_memory("", "// highlight burn-in risk")
    burn_target = rl.load_render_texture(4, 1)
    rl.begin_texture_mode(burn_target)
    rl.clear_background(rl.Color(0, 0, 192, 255))
    rl.end_texture_mode()
    rl.begin_shader_mode(burn_shader)
    rl.draw_texture_pro(burn_target.texture, rl.Rectangle(0, 0, 4, -1),
                        rl.Rectangle(24, 1, 4, 1), rl.Vector2(0, 0), 0, rl.WHITE)
    rl.end_shader_mode()
    assert tuple(framebuffer()[1, 24]) == (255, 126, 0, 255)
    rl.unload_render_texture(burn_target)
    rl.unload_shader(burn_shader)

    read_fd, write_fd = os.pipe()
    os.set_blocking(read_fd, False)
    if state.touch_fd >= 0:
      os.close(state.touch_fd)
    state.touch_fd = read_fd
    state.touch_canonical_zero = False
    event = struct.Struct("llHHi")
    records = [
      event.pack(0, 0, 3, 0x2f, 0),
      event.pack(0, 0, 3, 0x39, 1),
      event.pack(0, 0, 3, 0x35, 10),
      event.pack(0, 0, 3, 0x36, 20),
      event.pack(0, 0, 0, 0, 0),
      event.pack(0, 0, 3, 0x39, -1),
      event.pack(0, 0, 0, 0, 0),
    ]
    os.write(write_fd, b"".join(records))
    rl.poll_input_events()
    assert rl.is_mouse_button_pressed(0)
    touch = rl.get_touch_position(0)
    assert (touch.x, touch.y) == (20, 38)
    rl.poll_input_events()
    assert rl.is_mouse_button_released(0)

    state.touch_canonical_zero = True
    os.write(write_fd, b"".join([
      event.pack(0, 0, 3, 0x39, 2),
      event.pack(0, 0, 3, 0x35, 10),
      event.pack(0, 0, 3, 0x36, 20),
      event.pack(0, 0, 0, 0, 0),
    ]))
    rl.poll_input_events()
    touch = rl.get_touch_position(0)
    assert (touch.x, touch.y) == (44, 10)
    os.close(write_fd)

    assert framebuffer().sum() > 0
    rl.close_window()
    rl.init_window(8, 8, "cpu-reinit")
    rl.clear_background(rl.GREEN)
    assert tuple(framebuffer()[0, 0]) == tuple(rl.GREEN)
    rl.close_window()
  """)
  env = os.environ.copy()
  env.update({
    "CPU_OFFSCREEN": "1",
    "PYTHONPATH": f"{BASEDIR}:{env.get('PYTHONPATH', '')}",
    "RAYLIB_BACKEND": "cpu",
  })
  subprocess.run([sys.executable, "-c", code], cwd=BASEDIR, env=env, check=True)
