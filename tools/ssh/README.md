# SSH

## Quick Start

In order to SSH into your device, you'll need a GitHub account with SSH keys. See this [GitHub article](https://docs.github.com/en/github/authenticating-to-github/connecting-to-github-with-ssh) for getting your account setup with SSH keys.

* Enable SSH in your device's settings
* Enter your GitHub username in the device's settings
* Connect to your device
  * Username: `comma`
  * Port: `22` or `8022`

Here's an example command for connecting to your device using its tethered connection:<br />
`ssh comma@192.168.43.1`

For doing development work on device, it's recommended to use [SSH agent forwarding](https://docs.github.com/en/developers/overview/using-ssh-agent-forwarding).

## Notes

The public keys are only fetched from your GitHub account once. In order to update your device's authorized keys, you'll need to re-enter your GitHub username.

The `id_rsa` key in this directory only works while your device is in the setup state with no software installed. After installation, that default key will be removed.

See the [community wiki](https://github.com/commaai/openpilot/wiki/SSH) for more detailed instructions and information.

# Connecting to ssh.comma.ai
SSH into your comma device from anywhere with `ssh.comma.ai`. Requires a [comma prime subscription](https://comma.ai/connect).

## Setup

With software version 0.6.1 or newer, enter your GitHub username on your device under Developer Settings. Your GitHub authorized public keys will become your authorized SSH keys for `ssh.comma.ai`. You can add any additional keys in `/system/comma/home/.ssh/authorized_keys.persist`.

## Recommended .ssh/config

With the below SSH configuration, you can type `ssh comma-{dongleid}` to connect to your device through `ssh.comma.ai`.<br />
For example: `ssh comma-ffffffffffffffff`

```
Host comma-*
  Port 22
  User comma
  IdentityFile ~/.ssh/my_github_key
  ProxyCommand ssh %h@ssh.comma.ai -W %h:%p
Host ssh.comma.ai
  Hostname ssh.comma.ai
  Port 22
  IdentityFile ~/.ssh/my_github_key
```

## One-off connection

```
ssh -i ~/.ssh/my_github_key -o ProxyCommand="ssh -i ~/.ssh/my_github_key -W %h:%p -p %p %h@ssh.comma.ai" comma@ffffffffffffffff
```
(Replace `ffffffffffffffff` with your dongle_id)

## ssh.comma.ai host key fingerprint

```
Host key fingerprint is SHA256:X22GOmfjGb9J04IA2+egtdaJ7vW9Fbtmpz9/x8/W1X4
+---[RSA 4096]----+
|                 |
|                 |
|        .        |
|         +   o   |
|        S = + +..|
|         + @ = .=|
|        . B @ ++=|
|         o * B XE|
|         .o o OB/|
+----[SHA256]-----+
```
