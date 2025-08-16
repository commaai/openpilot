# tinybox

Although these docs live in tinygrad, they pertain to deep learning hardware sold by the tiny corp. tinyboxes are used heavily in tinygrad's CI, and are the best tested platform to use tinygrad with. They appeared running tinygrad on [MLPerf Training 4.0](https://public.tableau.com/views/MLCommons-Training_16993769118290/MLCommons-Training)

If you don't have a tinybox and you want one, see [tinygrad.org](https://tinygrad.org). If you don't want one, that's okay too.

## Welcome

Welcome to your tinybox! The tinybox is the universal system purpose-built for all AI infrastructure and workloads, from training to inference. The red box includes six 7900XTX GPUs, and the green box includes six 4090 GPUs. Whether you bought a red one or a green one, we want you to love it.

We don't have a stupid cloud service, you don't have to create a tiny account to set it up, and we aren't tracking how you use the box. We're just happy you bought one. This petaflop is your petaflop.

## Plugging it in

tinybox has two 1600W PSUs, which together exceed the capacity of most 120V household circuits. Fortunately, it comes with two plugs. You'll want to plug each plug into a different circuit. You can verify that they are different circuits by flipping the breaker and seeing what turns off. If you have at least a 120V 30A or 220V 20A circuit, you are welcome to use only that one.

You'll also want to connect the Ethernet port without a rubber stopper to your home network.

While it's designed primarily for the home or office, the tinybox is 12U rack mountable using [these rails](https://rackmountmart.store.turbify.net/26slidrailfo.html).

## Power limiting the box

While a tinybox should ideally be run without power limits, there are cases where you might want to run the box off of a single outlet.

In such cases, it is possible to power limit the box using the provided `power-limit` script, which will power limit all of the GPUs to a specified wattage.

`sudo power-limit 150` should be good to run off of a single 120V 15A outlet.

## Connecting to the box

tinybox ships with a relatively basic install of Ubuntu 22.04. To do initial setup, you can either plug in a VGA monitor and keyboard, or you can connect remotely to the machine using the BMC. The BMC IP and password are displayed on the screen.

`ipmitool -H <BMC IP> -U admin -P <BMC PW> -I lanplus sol activate`

The default username is `tiny` and the default password is `tiny`. Once you are logged in, you can add an SSH key to authorized keys to connect over SSH (on the normal IP). Exit `ipmitool` with `~.` after a newline.

The BMC also has a web interface you can use if you find that easier.

## Changing the BMC password

It is recommended that you change the BMC password after setting up the box, as the password on the screen is only the initial password.

If you do decide to change the BMC password and no longer want the initial password to be displayed, remove the `/root/.bmc_password` file.
Reboot after making these changes or restart the `displayservice.service` service.

## What do I use it for?

The [default tinybox image](https://github.com/tinygrad/tinyos) ships with tinygrad and PyTorch. While we develop tinygrad, the box is universal hardware. Use whatever framework you desire, run notebooks, download demos, install more things, train, inference, live, laugh, love, you aren't paying per hour for this box so the only limit is your imagination.

## Building the OS image

The OS image is built using `ubuntu-image` from <https://github.com/tinygrad/tinyos>.

After cloning, run `make green` or `make red` to build a tinybox green or tinybox red image respectively.
