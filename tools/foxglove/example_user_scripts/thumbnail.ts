// This example shows how to publish a foxglove.CompressedImage message using
// thumbnail events.
//
// You can visualize this message with the Map panel
// https://docs.foxglove.dev/docs/visualization/panels/image
import { Input } from "./types";
import { CompressedImage } from "@foxglove/schemas";


type Output = CompressedImage;


export const inputs = ["thumbnail"];

export const output = "/thumbnail";

export default function script(event: Input<"thumbnail">): Output {
  return {
    timestamp: event.receiveTime,
    frame_id: event.message.thumbnail.frameId.toString(),
    format: "jpeg",
    data: event.message.thumbnail.thumbnail,
  };
};
