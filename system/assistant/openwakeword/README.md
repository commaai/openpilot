This code was copied from https://github.com/dscripka/openWakeWord

To test wake word detection on the comma device or PC, run wakeword.py and say "alexa".

Install google cloud cli on comma 3 with:

sudo apt-get update

curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg

echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list

sudo apt-get update && sudo apt-get install google-cloud-cli

other installation methods shown here https://cloud.google.com/sdk/docs/install#deb

then run gcloud init to login to your google account with speech api enabled and then follow the prompts to select your google project.

To create and download a service account key file from the Google Cloud Console, you can follow these steps:

    Sign in to Google Cloud Console: Go to the Google Cloud Console and sign in with your Google account.

    Select or Create a Project:
        If you already have a Google Cloud project, select it from the top of the page.
        If you don’t have a project, create a new one by clicking on the project selection dropdown and then on "New Project".

    Enable the Speech-to-Text API:
        Go to the "API & Services" dashboard.
        Click on "+ ENABLE APIS AND SERVICES" at the top.
        Search for "Speech-to-Text API", select it, and click "Enable".

    Create a Service Account:
        Navigate to the "IAM & Admin" section, then select "Service accounts".
        Click "Create Service Account" at the top.
        Enter a name and description for your service account.
        Click "Create".

    Grant Access to the Service Account (Optional):
        After creating the account, you can assign roles to it. For the Speech-to-Text API, roles like "Owner", "Editor", or "Speech-to-Text User" could be relevant, but for security best practices, it's advisable to grant the least privilege necessary.
        Click "Continue" after assigning roles.

    Generate the Key File:
        After creating the service account and assigning roles, click on "Done".
        Find the service account you just created in the list, and click on it.
        Go to the "Keys" tab.
        Click "Add Key" and select "Create new key".
        Choose "JSON" as the key type and click "Create".
        The JSON key file will be generated and downloaded to your computer. This file contains the credentials needed for your application to authenticate with Google Cloud services.

    Set the Environment Variable:
        After downloading the key file, you need to set an environment variable on your system that points to this file.
        export GOOGLE_APPLICATION_CREDENTIALS=<path/to/projectname.json>

then install pthe google speech python api with:

pip install google-cloud-speech