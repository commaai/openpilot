# DO NOT PUSH!
import io
import base64


def takeSnapshot():
  from selfdrive.camerad.snapshot.snapshot import snapshot, jpeg_write
  ret = snapshot()
  if ret is not None:
    def b64jpeg(x):
      if x is not None:
        # f = io.BytesIO()
        with open('/data/snapshot.jpeg', 'wb') as f:
          jpeg_write(f, x)
        print('Wrote image!')
        # return base64.b64encode(f.getvalue()).decode("utf-8")
    b64jpeg(ret[0])
    # return {'jpegBack': b64jpeg(ret[0]),
    #         'jpegFront': b64jpeg(ret[1])}
  else:
    raise Exception("not available while camerad is started")


imgs = takeSnapshot()
# if imgs is not None:
#   with open('/data/snapshot.jpeg', 'wb') as f:
#     f.write(imgs['jpegBack'])
