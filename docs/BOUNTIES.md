# Bounties

Get paid to improve openpilot!

## Rules

* code must be merged into openpilot master
* bounty eligibility is solely at our discretion
* once you open a PR, the bounty is locked to you until you stop working on it
* open a ticket at [comma.ai/support](https://comma.ai/support/shop-order) with links to your PRs to claim
* get an extra 20% if you redeem your bounty in [comma shop](https://comma.ai/shop) credit

New bounties can be proposed in the **#contributing** channel in Discord.

## Issue bounties

We've tagged bounty eligible issues across openpilot and the rest of our repos; check out all the open ones [here](https://github.com/search?q=org%3Acommaai+is%3Aissue+label%3Abounty++&type=issues&state=open). These bounties roughly work out like this:
* **$100** - a few hours of work for an experienced openpilot developer; a good intro for someone new to openpilot
* **$300** - a day of work for an experienced openpilot developer
* **$500** - a few days of work for an experienced openpilot developer
* **$1k+** - a week or two of work (could be less for the right person)

## Car bounties

The car bounties only apply to cars that have a path to ship in openpilot release, which excludes unsupportable cars (e.g. Fords with a steering lockout) or cars that require extra hardware (Honda Accord with serial steering).

#### Brand or platform port - $2000
Example PR: [commaai/openpilot#23331](https://github.com/commaai/openpilot/pull/23331)

This is for adding support for an entirely new brand or a substantially new ADAS platform within a brand (e.g. the Volkswagen PQ platform).

#### Model port - $250
Example PR: [commaai/openpilot#30245](https://github.com/commaai/openpilot/pull/30245)

This is for porting a new car model that runs on a platform openpilot already supports.
In the average case, this is a few hours of work for an experienced software developer.

This bounty also covers getting openpilot supported on a previously unsupported trim of an already supported car, e.g. the Chevy Bolt without ACC.

#### Reverse Engineering a new Actuation Message - $300

This is for cars that are already supported, and it has three components:
* reverse a new steering, adaptive cruise, or AEB message
* merge the DBC definitions to [opendbc](http://github.com/commaai/opendbc)
* merge the openpilot code to use it and post a demo route

The control doesn't have to be perfect, but it should generally do what it's supposed to do.

### Specific Cars

#### Rivian R1T or R1S - $3000

Get a Rivian driving with openpilot.
Requires a merged port with lateral control and at least a POC of longitudinal control.

#### Toyota SecOc - $5000

We're contributing $5k to the [community-organized bounty](https://github.com/commaai/openpilot/discussions/19932).

#### Chevy Bolt with SuperCruise - $2500

The Bolt is already supported on the trim with standard ACC. Get openpilot working on the trim with SuperCruise. It must be a normal install: no extra pandas or other hardware, no ECU reflashes, etc.
