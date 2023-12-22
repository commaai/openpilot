// This example shows how to publish a foxglove.FrameTransform message using
// modelV2 events.
//
// You can visualize this message with the 3D panel
// https://docs.foxglove.dev/docs/visualization/panels/3d
import { Input, Message } from "./types";
import { FrameTransform } from "@foxglove/schemas";

type Output = FrameTransform;

export const inputs = ["modelV2"];
export const output = "/pose";

function toQuaternion(roll: number, pitch: number, yaw: number) // roll (x), pitch (y), yaw (z), angles are in radians
{
    // Abbreviations for the various angular functions

    const cr = Math.cos(roll * 0.5);
    const sr = Math.sin(roll * 0.5);
    const cp = Math.cos(pitch * 0.5);
    const sp = Math.sin(pitch * 0.5);
    const cy = Math.cos(yaw * 0.5);
    const sy = Math.sin(yaw * 0.5);

    var q = {w: 0, x: 0, y: 0, z: 0};
    q.w = cr * cp * cy + sr * sp * sy;
    q.x = sr * cp * cy - cr * sp * sy;
    q.y = cr * sp * cy + sr * cp * sy;
    q.z = cr * cp * sy - sr * sp * cy;

    return q;
}

export default function script(event: Input<"modelV2">): Output {
  const position = event.message.modelV2.temporalPose.transStd
  const orientation = event.message.modelV2.temporalPose.rotStd
  return {
    timestamp: event.receiveTime,
    parent_frame_id: (event.message.modelV2.frameId - 1).toString(),
    child_frame_id: event.message.modelV2.frameId.toString(),
    translation: {x:position[0], y: position[1], z: position[2]},
    rotation: toQuaternion(orientation[0], orientation[1], orientation[2])
  };
};
