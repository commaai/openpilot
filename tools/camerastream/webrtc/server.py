#!/usr/bin/env python
import argparse
import asyncio
import json
import logging
import os
import ssl
from typing import OrderedDict
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCRtpCodecCapability
from compressed_vipc_track import VisionIpcTrack

ROOT = os.path.dirname(__file__)

cams = ["roadEncodeData","wideRoadEncodeData","driverEncodeData"]

async def index(request):
    content = open(os.path.join(ROOT, "index.html"), "r").read()
    return web.Response(content_type="text/html", text=content)

async def javascript(request):
    content = open(os.path.join(ROOT, "client.js"), "r").read()
    return web.Response(content_type="application/javascript", text=content)


async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print("Connection state is %s" % pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    # TODO: stream the microphone
    audio = None

    video = VisionIpcTrack(cams[int(args.cam)], args.addr)

    video_sender = pc.addTrack(video)
    transceiver = next(t for t in pc.getTransceivers() if t.sender == video_sender)
    transceiver.setCodecPreferences(
        # [codec for codec in codecs if codec.mimeType == forced_codec]
        [RTCRtpCodecCapability(
                        mimeType="video/H264",
                        clockRate=90000,
                        channels=None,
                        parameters=OrderedDict([
                        ("packetization-mode", "1"),
                        ("level-asymmetry-allowed", "1"),
                        ("profile-level-id", "42001f"),
                        ])
                    )]
    )

    await pc.setRemoteDescription(offer)

    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ),
    )


pcs = set()


async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decode video streams and broadcast via WebRTC")
    parser.add_argument("addr", help="Address of comma three")

    # Not implemented (yet?). Geo already made the PoC for this, it should be possible.
    # parser.add_argument("--nvidia", action="store_true", help="Use nvidia instead of ffmpeg")

    parser.add_argument("--cam", default="0", help="Camera to stream")

    parser.add_argument("--cert-file", help="SSL certificate file (for HTTPS)")
    parser.add_argument("--key-file", help="SSL key file (for HTTPS)")
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host for HTTP server (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="Port for HTTP server (default: 8080)"
    )
    parser.add_argument("--verbose", "-v", action="count")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.cert_file:
        ssl_context = ssl.SSLContext()
        ssl_context.load_cert_chain(args.cert_file, args.key_file)
    else:
        ssl_context = None

    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/", index)
    app.router.add_get("/client.js", javascript)
    app.router.add_post("/offer", offer)
    web.run_app(app, host=args.host, port=args.port, ssl_context=ssl_context)