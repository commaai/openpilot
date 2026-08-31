"""Opt-in rendering reductions for CPU-only MetaDrive CI."""

import os


def apply_ci_render_patches():
  if os.environ.get("METADRIVE_NO_MSAA"):
    from panda3d.core import loadPrcFileData
    loadPrcFileData("", "framebuffer-multisample 0")
    loadPrcFileData("", "multisamples 0")

  if os.environ.get("METADRIVE_NO_SHADOWS"):
    from metadrive.engine.core.pssm import PSSM
    original_init = PSSM.init

    def init_without_rendering(self):
      original_init(self)
      self.buffer.set_active(False)
      self.use_pssm = False
      self.engine.render.set_shader_inputs(use_pssm=False)

    PSSM.init = init_without_rendering

  if os.environ.get("METADRIVE_FLAT_TERRAIN_CARD"):
    from metadrive.constants import CameraTagStateKey, CamMask
    from metadrive.engine.core.terrain import Terrain
    from panda3d.core import (Geom, GeomNode, GeomTriangles, GeomVertexData, GeomVertexFormat,
                              GeomVertexWriter, Shader)

    def generate_card(self, size, heightfield, attribute_tex, target_triangle_width=10, engine=None):
      engine = engine or self.engine
      vdata = GeomVertexData("terrain_card", GeomVertexFormat.getV3t2(), Geom.UHStatic)
      vdata.setNumRows(4)
      vertices = GeomVertexWriter(vdata, "vertex")
      texcoords = GeomVertexWriter(vdata, "texcoord")
      for x, y in ((0, 0), (1, 0), (1, 1), (0, 1)):
        vertices.addData3(x, y, 0)
        texcoords.addData2(x, y)
      triangles = GeomTriangles(Geom.UHStatic)
      triangles.addVertices(0, 1, 2)
      triangles.addVertices(0, 2, 3)
      geom = Geom(vdata)
      geom.addPrimitive(triangles)
      node = GeomNode("terrain_card")
      node.addGeom(geom)
      self._mesh_terrain = self.origin.attach_new_node(node)
      self._mesh_terrain.setTwoSided(True)
      self._mesh_terrain.hide(CamMask.MainCam)
      here = os.path.dirname(os.path.abspath(__file__))
      self._mesh_terrain.set_shader(Shader.load(Shader.SL_GLSL,
                                                os.path.join(here, "terrain_card.vert.glsl"),
                                                os.path.join(here, "terrain_ci.frag.glsl")))
      self._mesh_terrain.setTag(CameraTagStateKey.Semantic, self.SEMANTIC_LABEL)
      self._mesh_terrain.setTag(CameraTagStateKey.RGB, self.SEMANTIC_LABEL)
      self._mesh_terrain.setTag(CameraTagStateKey.Depth, self.SEMANTIC_LABEL)
      self._terrain_shader_set = False
      self._set_terrain_shader(engine, attribute_tex)
      self._mesh_terrain.set_scale(size, size, 1)
      # MetaDrive's roads sit almost on the terrain plane. Keep the simplified
      # two-triangle terrain slightly below them to avoid depth fighting that
      # can hide the road for several seconds and feed blank frames to modeld.
      self._mesh_terrain.set_pos(-size / 2, -size / 2, -0.2)

    Terrain._generate_mesh_vis_terrain = generate_card
