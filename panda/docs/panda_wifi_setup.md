# Connecting to White Panda via Wi-Fi

1. First connect to your White Panda's Wi-Fi pairing network (this should be the Wi-Fi network WITH the "-pair" at the end)

2. Now in your favorite web browser go to this address **192.168.0.10** (this should open a web interface to interact with the White Panda)

3. Inside the web interface enable secured mode by clinking the **secure it** link/button (this should make the White Panda's Wi-Fi network visible)

   ### If you need your White Panda's Wi-Fi Password 

   * Run the **get_panda_password.py** script in found in **examples/** (Must have panda paw for this step because you need to connect White Panda via USB to retrive the Wi-Fi password)
   * Also ensure that you are connected to your White Panda's Wi-Fi pairing network 

4. Connect to your White Panda's default Wi-Fi network (this should be the Wi-Fi network WITHOUT the "-pair" at the end)

5. Your White Panda is now connected to Wi-Fi you can test this by running this line of code `python -c 'from panda import Panda; panda = Panda("WIFI")'` in your terminal of choice.