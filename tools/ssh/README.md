# SSH

## Quick Start

In order to SSH into your device, you'll need a GitHub account with SSH keys. See this [GitHub article](https://docs.github.com/en/github/authenticating-to-github/connecting-to-github-with-ssh) for getting your account setup with SSH keys.

* Enable SSH in your device's settings
* Enter your GitHub username in the device's settings
* Connect to your device
  * Username: `root`
  * Port: `22` or `8022`

Here's an example command for connecting to your device using its tethered connection:
`ssh root@192.168.43.1`

For doing development work on device, it's recommended to use [SSH agent forwarding](https://docs.github.com/en/developers/overview/using-ssh-agent-forwarding).

## Notes

See the [community wiki](https://github.com/commaai/openpilot/wiki/SSH) for more detailed instructions and information.
