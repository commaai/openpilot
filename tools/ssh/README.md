# Quick Start

SSH is disabled by default. SSH can be enabled in the Network section of the settings. The IP address can also be found in the settings.

* Username: root
* Port: 22 or 8022

# Generating an SSH key and adding it to your GitHub Account
An SSH key pair is like a padlock and its key, your public key is the padlock, and your private key holds the information of opening the padlock.

To create your SSH key pair, you will have to have OpenSSH client installed on your operating system, please consult your operating system manual on the procedure of accomplishing this.

Once OpenSSH Client is installed and operational, using command line console, you will have to type `ssh-keygen -t rsa -f ~/.ssh/id_rsa`, it will then prompt you for a password, you may ignore this and press the `enter` key; nevertheless, setting a password is always a good practice.

This will create two files in your `.ssh` folder, one file will be named `id_rsa` and another file will be named `id_rsa.pub`. This step is completed once these two files are present in your `.ssh` folder.

You may refer to [This GitHub Doc](https://docs.github.com/en/github/authenticating-to-github/adding-a-new-ssh-key-to-your-github-account) for more information regarding uploading your Public SSH Key.

# Register your public key on your comma device
In order for your comma device to obtain the key from your GitHub Account, it has to be connected to the internet and know your GitHub username.

1. Turn off your vehicle and wait till your comma device goes to `offroad` mode.
2. Navigate to `Settings → Developer → Enable SSH`, and ensure SSH has been enabled.
3. Navigate to `Settings → Developer → GitHub Username`.
4. Using the On-screen Keyboard, type in your GitHub username.

Your comma device will now obtain your Public Key from GitHub.

*Please kindly note that the obtaining process only occurs once, future keys added to your account will not be automatically fetched, and would require the user to repeat the obtaining process.*

# Connecting to your comma device using SSH
Now you may connect to your comma device using SSH, the comma device's default SSH port is `8022`.

You will need to find out your comma device's IP address either from your router or from the comma device itself. You may obtain your comma device's IP address by going to `Settings → WiFi → Open WiFi Settings → More Options → Three Dots Menu at the Top Left Corner → Advanced`, once you have scrolled to the bottom, you should be able to see its IP address.

We will assume the IP address to be `10.10.8.3`

In your command line console, type `ssh root@10.10.8.3 -p 8022` (replace `10.10.8.3` with the IP address you have obtained), your computer will now attempt to make connection with your comma device using SSH.

Once you see `root@localhost:/data/openpilot$`, you are connected and all set.
