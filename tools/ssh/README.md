![SSH into comma two](https://user-images.githubusercontent.com/37757984/82586797-0496cc00-9b4d-11ea-9e98-48d193cf38ff.jpg)
# Connecting To Your comma two Using SSH
This will walk you through the procedure of connecting to your comma two using SSH in order for you to make localised changes on your comma two.

Table of Contents
=================

* [Creating a GitHub Account](#create-github-account)
* [Generating an SSH Key](#generate-ssh-key)
* [Uploading the Public SSH Key To Your GitHub Account](#upload-public-key)
* [Register Your Public Key On Your comma two](#register-key-on-c2)
* [Connecting To Your comma two Using SSH](#connection)

# Creating a GitHub Account
In order to create a GitHub Account, you have to have your email address handy and ready to receive your confirmation email.
1. Go to the [GitHub Account Creation Page](https://github.com/join).
2. Enter your information, this includes your desired username, email address, and password.
3. Once your details are correct, you may click the `create account` button.
4. You will soon receive an account verification email from GitHub, once your account is verified, you may proceed to the next step.

For more information on email verification, please see the [GitHub Doc](https://docs.github.com/en/github/getting-started-with-github/verifying-your-email-address) regarding this topic.

# Generating an SSH Key
An SSH key pair is like a padlock and its key, your public key is the padlock, and your private key holds the information of opening the padlock.

To create your SSH key pair, you will have to have OpenSSH client installed on your operating system, please consult your operating system manual on the procedure of accomplishing this.

Once OpenSSH Client is installed and operational, using command line console, you will have to type `ssh-keygen -t rsa -f ~/.ssh/id_rsa`, it will then prompt you for a password, you may ignore this and press the `enter` key; nevertheless, setting a password is always a good practice.

This will create two files in your `.ssh` folder, one file will be named `id_rsa` and another file will be named `id_rsa.pub`. This step is completed once these two files are present in your `.ssh` folder.

# Uploading the Public SSH Key To Your GitHub Account
For your comma two to obtain your public key, it will have to be added to your GitHub Account.

1. Open the `id_rsa.pub` you just created in the text editor of your choice, copy its content as-is.
2. Login to your GitHub Account, on the top right corner, click your profile picture, this will open a menu, you shall now navigate to `Settings`.
3. You should be able to see an `SSH and GPG Keys` section under your Settings page, navigate to this page.
4. Using the `New SSH Key` function, enter a desired `Title` for your key, then paste the content of your *Public Key* (`id_rsa.pub`) into the field below.

This step is complete once GitHub accepts the Public Key.

You may refer to [This GitHub Doc](https://docs.github.com/en/github/authenticating-to-github/adding-a-new-ssh-key-to-your-github-account) for more information regarding uploading your Public SSH Key.

# Register Your Public Key On Your comma two
In order for your comma two to obtain the key from your GitHub Account, it has to be connected to the internet and know your GitHub username.

1. Turn off your vehicle and wait till your comma two goes to `offroad` mode.
2. Navigate to `Settings -> Developer -> Enable SSH`, and ensure SSH has been enabled.
3. Navigate to `Settings -> Developer -> GitHub Username`
4. Using the On-screen Keyboard, type in your GitHub username

Your comma two will now obtain your Public Key from GitHub.

# Connecting To Your comma two Using SSH
Now you may connect to your comma two using SSH, the comma two's default SSH port is `8022`.

You will need to find out your comma two's IP address either from your router or from the comma two itself. You may obtain your comma two's IP address by going to `Settings > WiFi > Open WiFi Settings > More Options > Three Dots in Top Left > Advanced`, once you have scrolled to the bottom, you should be able to see its IP address.

We will assume the IP address to be `10.10.8.3`

In your command line console, type `ssh root@10.10.8.3 -p 8022`, your computer will now attempt to make connection with your comma two using SSH.

Once you see `root@localhost:/data/openpilot$`, you are connected and all set.
