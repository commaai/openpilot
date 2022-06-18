When building the installer, please put the relevant vc_redist.x86.exe file into this folder.
Make sure that the uninstall registry key is correct in the panda_install.nsi file.

Here is a list of the VC runtime downloads: https://support.microsoft.com/en-us/help/2977003/the-latest-supported-visual-c-downloads
An list of the registry keys has been maintained here: https://stackoverflow.com/a/34209692/627525

Copy vscruntimeinfo.nsh.sample to vscruntimeinfo.nsh and edit it for your version of Visual Studio.