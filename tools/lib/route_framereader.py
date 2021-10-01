"""RouteFrameReader indexes and reads frames across routes, by frameId or segment indices."""
from tools.lib.framereader import FrameReader


class _FrameReaderDict(dict):
  def __init__(self, camera_paths, cache_paths, framereader_kwargs, *args, **kwargs):
    super(_FrameReaderDict, self).__init__(*args, **kwargs)

    if cache_paths is None:
      cache_paths = {}
    if not isinstance(cache_paths, dict):
      cache_paths = dict(enumerate(cache_paths))

    self._camera_paths = camera_paths
    self._cache_paths = cache_paths
    self._framereader_kwargs = framereader_kwargs

  def __missing__(self, key):
    if self._camera_paths.get(key, None) is not None:
      frame_reader = FrameReader(self._camera_paths[key],
                                 self._cache_paths.get(key), **self._framereader_kwargs)
      self[key] = frame_reader
      return frame_reader
    else:
      raise KeyError("Segment index out of bounds: {}".format(key))


class RouteFrameReader(object):
  """Reads frames across routes and route segments by frameId."""
  def __init__(self, camera_paths, cache_paths, frame_id_lookup, **kwargs):
    """Create a route framereader.

       Inputs:
        TODO

        kwargs: Forwarded to the FrameReader function. If cache_prefix is included, that path
                will also be used for frame position indices.
    """
    if not isinstance(camera_paths, dict):
      camera_paths = {int(k.split('?')[0].split('/')[-2]): k for k in camera_paths if k is not None}

    self._first_camera_idx = min(camera_paths.keys())
    self._frame_readers = _FrameReaderDict(camera_paths, cache_paths, kwargs)
    self._frame_id_lookup = frame_id_lookup

  @property
  def w(self):
    """Width of each frame in pixels."""
    return self._frame_readers[self._first_camera_idx].w

  @property
  def h(self):
    """Height of each frame in pixels."""
    return self._frame_readers[self._first_camera_idx].h

  def get(self, frame_id, **kwargs):
    """Get a frame for a route based on frameId.

       Inputs:
        frame_id: The frameId of the returned frame.
        kwargs: Forwarded to BaseFrameReader.get. "count" is not implemented.
    """
    segment_num, segment_id = self._frame_id_lookup.get(frame_id, (None, None))
    if segment_num is None or segment_num == -1 or segment_id == -1:
      return None
    else:
      return self.get_from_segment(segment_num, segment_id, **kwargs)

  def get_from_segment(self, segment_num, segment_id, **kwargs):
    """Get a frame from a specific segment with a specific index in that segment (segment_id).

       Inputs:
        segment_num: The number of the segment.
        segment_id: The index of the return frame within that segment.
        kwargs: Forwarded to BaseFrameReader.get. "count" is not implemented.
    """
    if "count" in kwargs:
      raise NotImplementedError("count")

    return self._frame_readers[segment_num].get(segment_id, **kwargs)[0]

  def close(self):
    frs = self._frame_readers
    self._frame_readers.clear()
    for fr in frs:
      fr.close()

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    self.close()
