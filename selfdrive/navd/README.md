# navigation

This directory contains two daemons, `navd` and `map_renderer`, which support navigation in the openpilot stack.

### navd

`navd` takes in a route through the `NavDestination` param and sends out two packets: `navRoute` and `navInstruction`. These packets contain the coordinates of the planned route and turn-by-turn instructions.

### map renderer

The map renderer listens for the `navRoute` and publishes a rendered map view over VisionIPC for the navigation model, which lives in `selfdrive/modeld/`. The rendered maps look like this:

![](https://i.imgur.com/oZLfmwq.png)

## development

Currently, [mapbox](https://www.mapbox.com/) is used for navigation.

* get an API token: https://docs.mapbox.com/help/glossary/access-token/
* set an API token using the `MAPBOX_TOKEN` environment variable
* routes/destinations are set through the `NavDestination` param
  * use `set_destination.py` for debugging
* edit the map: https://www.mapbox.com/contribute
* mapbox API playground: https://docs.mapbox.com/playground/
