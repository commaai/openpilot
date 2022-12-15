# navigation

This directory contains two daemons: `navd` and `map_renderer`.

### navd

The navd daemon sends two packets: `navRoute` and `navInstruction`.

### map renderer

The map renderer listens for the `navRoute` and publishes a rendered map view over VisionIPC for the navigation model, which is in `selfdrive/modeld/`. The rendered maps look like this:

![](https://i.imgur.com/oZLfmwq.png)

## development

Currently, [mapbox](https://www.mapbox.com/) is used for navigation.

* get an API token [mapbox](https://docs.mapbox.com/help/glossary/access-token/)
* set an API token using the `MAPBOX_TOKEN` environment variable
* routes/destinations are set through the `NavDestination` param
