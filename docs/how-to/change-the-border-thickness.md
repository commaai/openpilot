# Change the border thickness

The goal of this guide is to make our first changes to the openpilot UI. **No previous knowledge of coding is required.** As long as you follow each step you will be able to change the thickness of the outer green border that is shown when openpilot is engaged.

### Requirements:
* Computer
* Visual Studio Code installed
* Internet connection

---

<br>

## Step 1: Clone openpilot

Open up terminal to copy and paste each line below. Press `enter` on your keyboard after each line to execute the command. Wait for each line to execute before moving on to the next line.

Wait for it to clone openpilot and install dependencies. A faster internet speed will make install quicker. If clone fails check internet.
````bash
curl -fsSL openpilot.comma.ai | bash
````
<br>

## Step 2: Setup environment

**2a.** Navigate to openpilot folder & activate a Python virtual environment
```bash
cd openpilot
source .venv/bin/activate
```

**2b.** Builds openpilot UI
```bash
scons -j8
```

If this is your first time using terminal, seeing something like below is totally normal. It should look like a bunch of seemingly random words to someone new. As openpilot is building you will see a lot of words being scrolled through.
```bash
common/libcommon.a -ljson11 cereal/libsocketmaster.a msgq_repo/libmsgq.a -lzmq -lcapnp -lkj cereal/libcereal.a msgq_repo/libvisionipc.a
-lm -lssl -lcrypto -lpthread -lqt_util -lQt5Widgets -lQt5Gui -lQt5Core -lQt5Network -lQt5Concurrent -lQt5Multimedia -lQt5Quick
-lQt5Qml -lQt5QuickWidgets -lQt5DBus -lQt5Xml -lGL -lOpenCL
scons: done building targets.
```

If you see `scons: done building targets.` at the bottom of the terminal that's a great sign! You successfully built openpilot!

**2c.** Check what the path is to the cloned openpilot in the terminal
```
pwd
```

You should see an output similar to below
```
/home/dcr/TEST/openpilot
```

This shows where my openpilot clone is. This is what we will need to find when we open up Visual Studio Code.

<br>

## Step 3: Locate the openpilot folder

Open up Visual Studio Code and open up the `openpilot` folder. Refer to the path from the previous step if you are trying to locate where your clone is.

If you are new, find the top toolbar and locate the button `File` then find the button `Open Folder`. Then locate the `openpilot` folder and click open.

<br>

## Step 4: Searching for the UI code

**4a.** Look for `border_size`<br>
Use the search tool. Click on the magnifying glass icon which will then show a search bar at the top left corner. Type `border_size` and press `enter`. You will then see many results with the keyword highlighted in the sidebar.

**4b.** Navigate to `ui.h` file<br>
Click on the result that shows `const int UI_BORDER_SIZE = 30;`. This should be in the `ui.h` file.

<br>

## Step 5: Change the thickness of the border

The code below changes how thick the openpilot border is. Increase the number for a thicker border, decrease it for a thinner border.
```
const int UI_BORDER_SIZE = 30;
```

I changed it to 60 for a thicker border.
```
const int UI_BORDER_SIZE = 60;
```

<br>

## Step 6: Save the changes

Make sure after making the changes to save them. If the changes are not saved when rebuilding the UI you will not see the expected change.

<br>

## Step 7: Run replay to see changes

**7a.** Open up 2 terminals<br>
Make sure each terminal is in the openpilot folder or else the below commands will not work. In each terminal copy and paste the different commands as shown below.

**7b.** Terminal 1 commands<br>
You will see your terminal transform into a different UI that shows information about the given route.
```bash
tools/replay/replay --demo
```

**7c.** Terminal 2 commands<br>
This will activate Python environment and also rebuild the UI. This could take a similar length of time to the first build we did in the beginning in step 2b.
```bash
source .venv/bin/activate
scons -j8 && selfdrive/ui/ui
```

<br>

## Step 8: Look at the UI change

After terminal 2 finishes another window should pop up that shows the openpilot UI replaying the demo route. You should then see a thicker or thinner border depending on your change. Nice job!

<br>

---

## Extra tips when adjusting UI code for beginners

- Your biggest friend is using the search tool from VSCode
- Don't forget to activate Python virtual environment
- Use VSCode source control to make sure you are not making any unintended changes
- Make sure to save after changes, then rebuild every time you make a UI change in openpilot to view your change
