// This example shows how to publish a foxglove.Log message using
// errorLogMessage events.
//
// You can visualize this message with the Log panel
// https://docs.foxglove.dev/docs/visualization/panels/log
import { Input, Message } from "./types";
import { Log, LogLevel } from "@foxglove/schemas";

type Output = Log;

export const inputs = ["errorLogMessage"];
export const output = "/logs";

export default function script(event: Input<"errorLogMessage">): Output {
  const message = JSON.parse(event.message.errorLogMessage)
  var level = LogLevel.UNKNOWN
  if (message?.level == "ERROR") {
    level = LogLevel.ERROR
  }
  if (message?.level == "INFO") {
    level = LogLevel.INFO
  }
  if (message?.level == "WARNING") {
    level = LogLevel.WARNING
  }
  return {
    timestamp: event.receiveTime,
    level: level,
    message: event.message.errorLogMessage,
    name: message?.ctx?.daemon,
    file: message?.filename,
    line: message?.lineno,
  };
};
