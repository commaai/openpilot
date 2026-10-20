"""Relays the page's read of its referrals to billing, which only answers CORS for connect's origins. One fixed GET,
with the page's own Authorization: prime's checkout, portal and cancel are never relayed."""
from openpilot.tools.ycpe.daemon.athena import forward
from openpilot.tools.ycpe.daemon.core import module, route

BILLING = "https://billing.comma.ai"

module("billing", "1")


@route("GET", "/local/billing/referrals")
def referrals(req):
  return forward(req, "GET", f"{BILLING}/v1/referrals", name="billing")
