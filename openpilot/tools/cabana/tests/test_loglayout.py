from pathlib import Path
import unittest

from openpilot.tools.cabana.import_loglayout import convert_layout


class TestLogLayoutImport(unittest.TestCase):
  def test_upstream_longitudinal_layout(self):
    path = Path(__file__).parents[2] / "plotjuggler/layouts/longitudinal.xml"
    layout = convert_layout(path.read_text())
    self.assertEqual(len(layout["plots"]), 4)
    self.assertEqual(layout["plots"][0][0]["signal"], "carState/aEgo")
    self.assertEqual(len(layout["plots"][0]), 4)

  def test_scale_and_offset(self):
    layout = convert_layout('''<root><plot mode="TimeSeries"><curve name="/carState/vEgo">
      <transform name="Scale/Offset"><options value_scale="3.6" value_offset="-1" time_offset="0"/>
      </transform></curve></plot></root>''')
    self.assertEqual(layout["plots"][0][0], {"signal": "carState/vEgo", "scale": 3.6, "offset": -1, "derivative": False})

  def test_unsupported_modes_are_rejected(self):
    for xml in ('<root><plot mode="XY"><curve name="/carState/vEgo"/></plot></root>',
                '<root><plot><curve name="custom math"/></plot></root>',
                '<root><plot><curve name="/x"><transform name="Derivative"/></curve></plot></root>',
                '<root><plot><curve name="/x"><transform name="Scale/Offset">'
                '<options time_offset="1"/></transform></curve></plot></root>',
                '<root><plot><curve name="/x"><transform name="Scale/Offset">'
                '<options value_scale="nan"/></transform></curve></plot></root>',
                '<root><plot><curve name="/"/></plot></root>', '<root/>'):
      with self.subTest(xml=xml), self.assertRaises(ValueError):
        convert_layout(xml)

  def test_custom_math_cannot_masquerade_as_raw_signal(self):
    xml = '<root><plot><curve name="/derived"/></plot><customMathEquations><snippet name="/derived"/></customMathEquations></root>'
    with self.assertRaises(ValueError):
      convert_layout(xml)


if __name__ == "__main__":
  unittest.main()
