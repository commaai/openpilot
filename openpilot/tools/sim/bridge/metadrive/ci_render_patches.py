"""Env-gated render simplifications for running the MetaDrive bridge on CI.

Free GitHub Actions runners have no GPU; Mesa LLVMpipe software rendering
cannot sustain the default MetaDrive pipeline in real time (issue #30693).
Each flag removes render work without changing what openpilot's camera sees
structurally (road layout, lane lines, horizon).

Must be called before the MetaDrive engine is created — Prc settings load at
engine init and the terrain patch replaces a method used during reset.
"""
import os


def apply_ci_render_patches():
  if os.environ.get("METADRIVE_NO_MSAA"):
    # metadrive's EngineCore forces 8x MSAA at import time; Prc settings are
    # last-write-wins, so this must load before the engine is created
    from panda3d.core import loadPrcFileData
    loadPrcFileData("", "framebuffer-multisample 0")
    loadPrcFileData("", "multisamples 0")

  if os.environ.get("METADRIVE_NO_SHADOWS"):
    # PSSM renders the scene into a 2-split shadow atlas every frame;
    # deactivate the shadow buffer but keep its shader inputs bound so the
    # stock terrain shader stays valid
    from metadrive.engine.core.pssm import PSSM
    pssm_init_orig = PSSM.init
    def pssm_init_no_render(self):
      pssm_init_orig(self)
      self.buffer.set_active(False)
      self.use_pssm = False
      self.engine.render.set_shader_inputs(use_pssm=False)
    PSSM.init = pssm_init_no_render

  if os.environ.get("METADRIVE_FLAT_TERRAIN_CARD"):
    # the PG-map terrain is flat (physics already uses a plane); replace the
    # chunked ShaderTerrainMesh — whose draw path dominates LLVMpipe frame
    # time — with a single flat quad carrying the same shader inputs
    from metadrive.constants import CameraTagStateKey, CamMask
    from metadrive.engine.core.terrain import Terrain
    from panda3d.core import Geom, GeomNode, GeomTriangles, GeomVertexData, GeomVertexFormat, GeomVertexWriter, Shader

    def _gen_card(self, size, heightfield, attribute_tex, target_triangle_width=10, engine=None):
      engine = engine or self.engine
      vdata = GeomVertexData("terrain_card", GeomVertexFormat.getV3t2(), Geom.UHStatic)
      vdata.setNumRows(4)
      vw = GeomVertexWriter(vdata, "vertex")
      uw = GeomVertexWriter(vdata, "texcoord")
      for x, y in ((0, 0), (1, 0), (1, 1), (0, 1)):
        vw.addData3(x, y, 0)
        uw.addData2(x, y)
      tris = GeomTriangles(Geom.UHStatic)
      tris.addVertices(0, 1, 2)
      tris.addVertices(0, 2, 3)
      geom = Geom(vdata)
      geom.addPrimitive(tris)
      node = GeomNode("terrain_card")
      node.addGeom(geom)
      self._mesh_terrain = self.origin.attach_new_node(node)
      self._mesh_terrain.setTwoSided(True)
      # the main window camera's RGB tag state applies the stock terrain
      # shader, which asserts on ShaderTerrainMesh.* inputs only the real
      # terrain node provides — hide the card from it (the openpilot sensor
      # cameras carry their own tag state with the card shaders)
      self._mesh_terrain.hide(CamMask.MainCam)
      here = os.path.dirname(os.path.abspath(__file__))
      self._mesh_terrain.set_shader(Shader.load(Shader.SL_GLSL,
                                                os.path.join(here, "terrain_card.vert.glsl"),
                                                os.path.join(here, "terrain_ci.frag.glsl")))
      self._mesh_terrain.setTag(CameraTagStateKey.Semantic, self.SEMANTIC_LABEL)
      self._mesh_terrain.setTag(CameraTagStateKey.RGB, self.SEMANTIC_LABEL)
      self._mesh_terrain.setTag(CameraTagStateKey.Depth, self.SEMANTIC_LABEL)
      self._terrain_shader_set = False  # rebind inputs to the new node
      self._set_terrain_shader(engine, attribute_tex)
      self._mesh_terrain.set_scale(size, size, 1)
      self._mesh_terrain.set_pos(-size / 2, -size / 2, 0)

    Terrain._generate_mesh_vis_terrain = _gen_card
