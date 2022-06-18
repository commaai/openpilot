# Updating your panda

Panda should update automatically via the [openpilot](http://openpilot.comma.ai/).

On Linux or Mac OSX, you can manually update it using:
```
sudo pip install --upgrade pandacan`
PYTHONPATH="" sudo python -c "import panda; panda.flash_release()"`
```
