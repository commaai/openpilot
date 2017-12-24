# Updating your panda

Panda should update automatically via the [Chffr](http://chffr.comma.ai/) app ([apple](https://itunes.apple.com/us/app/chffr-dash-cam-that-remembers/id1146683979) and [android](https://play.google.com/store/apps/details?id=ai.comma.chffr))

If it doesn't however,  you can use the following commands on linux or Mac OSX
 `sudo pip install --upgrade pandacan`
` PYTHONPATH="" sudo python -c "import panda; panda.flash_release()"`

(You'll need to have `pip` and `sudo` installed.)
