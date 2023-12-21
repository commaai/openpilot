// This example shows how to publish a foxglove.LocationFix message using liveLocationKalman events.
//
// You can visualize this message with the Map panel
// https://docs.foxglove.dev/docs/visualization/panels/map

import { Input } from "./types";
import { LocationFix, PositionCovarianceType } from "@foxglove/schemas";

export const inputs = ["liveLocationKalman"];
export const output = "/gps/llk";

export default function script(
  event: Input<"liveLocationKalman">,
): LocationFix {
  return {
    timestamp: event.receiveTime,
    frame_id: event.message.logMonoTime,
    latitude: event.message.liveLocationKalman.positionGeodetic.value[0],
    longitude: event.message.liveLocationKalman.positionGeodetic.value[1],
    altitude: event.message.liveLocationKalman.positionGeodetic.value[2],
    position_covariance_type: PositionCovarianceType.APPROXIMATED,
    position_covariance: [0, 0, 0, 0, 0, 0, 0, 0, 0],
  };
}
