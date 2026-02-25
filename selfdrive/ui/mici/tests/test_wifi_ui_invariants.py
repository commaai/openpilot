import pyray as rl
rl.set_config_flags(rl.ConfigFlags.FLAG_WINDOW_HIDDEN)

import time
import unittest
import hypothesis.strategies as st
from hypothesis import given, settings, Phase

from openpilot.system.ui.lib.application import gui_app, MousePos, MouseEvent
from openpilot.system.ui.lib.wifi_manager import Network, SecurityType, WifiState, ConnectStatus, normalize_ssid
from openpilot.selfdrive.ui.mici.layouts.settings.network.wifi_ui import WifiUIMici, WifiButton
from openpilot.selfdrive.ui.mici.layouts.settings.network import WifiNetworkButton


class FakeWifiManager:
  def __init__(self):
    self.wifi_state = WifiState()
    self._networks: list[Network] = []
    self.ipv4_address: str = ""
    self._saved_ssids: set[str] = set()

  @property
  def networks(self) -> list[Network]:
    return self._networks

  @property
  def connecting_to_ssid(self) -> str | None:
    return self.wifi_state.ssid if self.wifi_state.status == ConnectStatus.CONNECTING else None

  @property
  def connected_ssid(self) -> str | None:
    return self.wifi_state.ssid if self.wifi_state.status == ConnectStatus.CONNECTED else None

  def is_connection_saved(self, ssid: str) -> bool:
    return ssid in self._saved_ssids

  def add_callbacks(self, **kwargs):
    pass

  def set_active(self, active: bool):
    pass

  def forget_connection(self, ssid: str):
    pass

  def connect_to_network(self, ssid: str, password: str, hidden: bool = False):
    self.wifi_state = WifiState(ssid=ssid, status=ConnectStatus.CONNECTING)

  def activate_connection(self, ssid: str, block: bool = False):
    self.wifi_state = WifiState(ssid=ssid, status=ConnectStatus.CONNECTING)


def _get_buttons(wifi_ui: WifiUIMici) -> dict[str, WifiButton]:
  return {btn.network.ssid: btn for btn in wifi_ui._scroller.items if isinstance(btn, WifiButton)}


def _check_disabled_invariant(buttons: dict[str, WifiButton]):
  for btn in buttons.values():
    should_disable = (btn._is_connecting or btn._is_connected or btn._network_missing
                      or btn._network_forgetting or btn.network.security_type == SecurityType.UNSUPPORTED)
    assert btn.enabled != should_disable, \
      f"'{btn.network.ssid}': enabled={btn.enabled}, connecting={btn._is_connecting}, " \
      f"connected={btn._is_connected}, missing={btn._network_missing}, " \
      f"forgetting={btn._network_forgetting}, security={btn.network.security_type}"


def _check_value_invariant(buttons: dict[str, WifiButton]):
  """After _update_state, button.value must match the state priority chain in _update_state."""
  for btn in buttons.values():
    # Priority: forgetting > connecting > connected > missing > unsupported > wrong_password > connect
    if btn._network_forgetting:
      expected = "forgetting..."
    elif btn._is_connecting:
      expected = "connecting..."
    elif btn._is_connected:
      expected = "connected"
    elif btn._network_missing:
      expected = "not in range"
    elif btn.network.security_type == SecurityType.UNSUPPORTED:
      expected = "unsupported"
    elif btn._wrong_password:
      expected = "wrong password"
    else:
      expected = "connect"

    assert btn.value == expected, \
      f"'{btn.network.ssid}': value={btn.value!r}, expected={expected!r} " \
      f"(forgetting={btn._network_forgetting}, connecting={btn._is_connecting}, " \
      f"connected={btn._is_connected}, missing={btn._network_missing}, " \
      f"security={btn.network.security_type}, wrong_pw={btn._wrong_password})"


def _check_net_btn_invariant(net_btn: WifiNetworkButton, wm):
  """WifiNetworkButton text and value must match wifi_state after _update_state."""
  ws = wm.wifi_state
  if ws.status == ConnectStatus.CONNECTING:
    expected_text = normalize_ssid(ws.ssid or "wi-fi")
    expected_value = "connecting..."
  elif ws.status == ConnectStatus.CONNECTED:
    expected_text = normalize_ssid(ws.ssid or "wi-fi")
    expected_value = wm.ipv4_address or "obtaining IP..."
  else:
    expected_text = "wi-fi"
    expected_value = "not connected"

  assert net_btn.text == expected_text, \
    f"net_btn.text={net_btn.text!r}, expected={expected_text!r} (state={ws})"
  assert net_btn.value == expected_value, \
    f"net_btn.value={net_btn.value!r}, expected={expected_value!r} (state={ws}, ipv4={wm.ipv4_address!r})"


def _check_forget_btn_invariant(buttons: dict[str, WifiButton]):
  """_show_forget_btn must match: (saved and not wrong_password) or connecting, unless tethering."""
  for btn in buttons.values():
    if btn.network.is_tethering:
      assert not btn._show_forget_btn, \
        f"'{btn.network.ssid}': tethering network should never show forget"
    else:
      expected = (btn._is_saved and not btn._wrong_password) or btn._is_connecting
      assert btn._show_forget_btn == expected, \
        f"'{btn.network.ssid}': _show_forget_btn={btn._show_forget_btn}, expected={expected} " \
        f"(saved={btn._is_saved}, wrong_pw={btn._wrong_password}, connecting={btn._is_connecting})"


# -- Hypothesis strategies --

SSID_ST = st.text(min_size=1, max_size=20)
NETWORK_ST = st.builds(Network, ssid=SSID_ST, strength=st.integers(0, 100),
                       security_type=st.sampled_from(list(SecurityType)), is_tethering=st.booleans())
NETWORKS_ST = st.lists(NETWORK_ST, max_size=8, unique_by=lambda n: n.ssid)


@st.composite
def WIFI_SCENARIOS(draw):
  networks = draw(NETWORKS_ST)
  ssids = [n.ssid for n in networks]
  status = draw(st.sampled_from(list(ConnectStatus)))
  ipv4 = draw(st.text(max_size=40))
  ssid = draw(st.one_of(st.sampled_from(ssids), SSID_ST, st.none())) if ssids else draw(st.one_of(SSID_ST, st.none()))
  saved = draw(st.frozensets(st.sampled_from(ssids) if ssids else st.nothing(), max_size=len(ssids)))
  return networks, WifiState(ssid=ssid, status=status), ipv4, saved


@st.composite
def EAGER_SCENARIOS(draw):
  initial = draw(st.lists(NETWORK_ST, min_size=1, max_size=6, unique_by=lambda n: n.ssid))
  known_ssids = [n.ssid for n in initial]
  actions = []

  for _ in range(draw(st.integers(1, 25))):
    action = draw(st.sampled_from([
      'network_update', 'need_auth', 'forgotten', 'forget_btn',
      'connect', 'disconnect', 'set_connected',
    ]))
    if action == 'network_update':
      nets = draw(st.lists(NETWORK_ST, max_size=6, unique_by=lambda n: n.ssid))
      actions.append(('network_update', nets))
      known_ssids = [n.ssid for n in nets]
    elif action == 'disconnect':
      actions.append(('disconnect',))
    elif known_ssids:
      actions.append((action, draw(st.sampled_from(known_ssids))))

  return initial, actions


RENDER_RECT = rl.Rectangle(0, 0, 1080, 500)

# Mouse event coordinates within the render area
MOUSE_X_ST = st.floats(0, 1080, allow_nan=False, allow_infinity=False)
MOUSE_Y_ST = st.floats(0, 500, allow_nan=False, allow_infinity=False)


@st.composite
def MOUSE_INPUT_SCENARIOS(draw):
  """Generate networks + sequences of mouse events interleaved with state changes."""
  initial = draw(st.lists(NETWORK_ST, min_size=1, max_size=6, unique_by=lambda n: n.ssid))
  known_ssids = [n.ssid for n in initial]
  steps = []

  ALL_ACTIONS = [
    'click', 'press_hold_release', 'drag', 'multi_touch',
    'render_frame', 'rapid_clicks', 'long_press',
    'network_update', 'connect', 'disconnect', 'set_connected',
    'show_hide', 'need_auth', 'forgotten',
    'change_saved', 'change_ipv4',
  ]

  for _ in range(draw(st.integers(3, 40))):
    action = draw(st.sampled_from(ALL_ACTIONS))

    if action == 'click':
      steps.append(('click', draw(MOUSE_X_ST), draw(MOUSE_Y_ST)))

    elif action == 'press_hold_release':
      x1, y1 = draw(MOUSE_X_ST), draw(MOUSE_Y_ST)
      x2, y2 = draw(MOUSE_X_ST), draw(MOUSE_Y_ST)
      steps.append(('press_hold_release', x1, y1, x2, y2))

    elif action == 'drag':
      n_points = draw(st.integers(2, 8))
      points = [(draw(MOUSE_X_ST), draw(MOUSE_Y_ST)) for _ in range(n_points)]
      steps.append(('drag', points))

    elif action == 'multi_touch':
      slot = draw(st.integers(0, 3))
      steps.append(('multi_touch', slot, draw(MOUSE_X_ST), draw(MOUSE_Y_ST), draw(st.booleans())))

    elif action == 'rapid_clicks':
      n_clicks = draw(st.integers(2, 6))
      clicks = [(draw(MOUSE_X_ST), draw(MOUSE_Y_ST)) for _ in range(n_clicks)]
      steps.append(('rapid_clicks', clicks))

    elif action == 'long_press':
      steps.append(('long_press', draw(MOUSE_X_ST), draw(MOUSE_Y_ST), draw(st.integers(2, 10))))

    elif action == 'render_frame':
      steps.append(('render_frame',))

    elif action == 'network_update':
      nets = draw(st.lists(NETWORK_ST, max_size=6, unique_by=lambda n: n.ssid))
      steps.append(('network_update', nets))
      known_ssids = [n.ssid for n in nets]

    elif action == 'connect' and known_ssids:
      steps.append(('connect', draw(st.sampled_from(known_ssids))))
    elif action == 'disconnect':
      steps.append(('disconnect',))
    elif action == 'set_connected' and known_ssids:
      steps.append(('set_connected', draw(st.sampled_from(known_ssids))))

    elif action == 'show_hide':
      steps.append(('show_hide',))
    elif action == 'need_auth' and known_ssids:
      steps.append(('need_auth', draw(st.sampled_from(known_ssids))))
    elif action == 'forgotten' and known_ssids:
      steps.append(('forgotten', draw(st.sampled_from(known_ssids))))

    elif action == 'change_saved':
      steps.append(('change_saved', draw(st.frozensets(
        st.sampled_from(known_ssids) if known_ssids else st.nothing(), max_size=len(known_ssids)))))
    elif action == 'change_ipv4':
      steps.append(('change_ipv4', draw(st.text(max_size=40))))

  return initial, steps


@st.composite
def CHAOS_SCENARIOS(draw):
  """Generate sequences mixing callbacks, user actions, lifecycle, and garbage inputs."""
  initial = draw(st.lists(NETWORK_ST, min_size=1, max_size=6, unique_by=lambda n: n.ssid))
  known_ssids = [n.ssid for n in initial]
  all_ssids = list(known_ssids)
  actions = []

  ALL_ACTIONS = [
    'network_update', 'need_auth', 'need_auth_new', 'forgotten', 'forget_btn',
    'connect', 'disconnect', 'set_connected',
    'user_click', 'user_click_phantom', 'connect_with_password',
    'change_saved', 'change_ipv4',
    'show_event', 'hide_event',
    'need_auth_phantom', 'forgotten_phantom',
    'double_forget', 'empty_network_update',
  ]

  for _ in range(draw(st.integers(1, 40))):
    action = draw(st.sampled_from(ALL_ACTIONS))

    if action == 'network_update':
      nets = draw(st.lists(NETWORK_ST, max_size=6, unique_by=lambda n: n.ssid))
      actions.append(('network_update', nets))
      known_ssids = [n.ssid for n in nets]
      all_ssids = list(set(all_ssids + known_ssids))
    elif action == 'empty_network_update':
      actions.append(('network_update', []))
      known_ssids = []
    elif action in ('disconnect', 'show_event', 'hide_event'):
      actions.append((action,))
    elif action == 'change_saved':
      actions.append(('change_saved', draw(st.frozensets(
        st.sampled_from(all_ssids) if all_ssids else st.nothing(), max_size=len(all_ssids)))))
    elif action == 'change_ipv4':
      actions.append(('change_ipv4', draw(st.text(max_size=40))))
    elif action == 'user_click' and known_ssids:
      actions.append(('user_click', draw(st.sampled_from(known_ssids))))
    elif action == 'user_click_phantom':
      actions.append(('user_click', draw(SSID_ST)))
    elif action == 'connect_with_password' and known_ssids:
      actions.append(('connect_with_password', draw(st.sampled_from(known_ssids)), draw(st.text(max_size=64))))
    elif action in ('need_auth_phantom', 'forgotten_phantom'):
      actions.append((action.replace('_phantom', ''), draw(SSID_ST)))
    elif action == 'need_auth_new' and known_ssids:
      actions.append(('need_auth_new', draw(st.sampled_from(known_ssids))))
    elif action == 'double_forget' and known_ssids:
      ssid = draw(st.sampled_from(known_ssids))
      actions.append(('forget_btn', ssid))
      actions.append(('forget_btn', ssid))
    elif known_ssids:
      actions.append((action, draw(st.sampled_from(known_ssids))))

  return initial, actions


class TestWifiUIInvariants(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    gui_app.init_window("test-wifi-invariants")
    cls.wm = FakeWifiManager()
    cls.net_btn = WifiNetworkButton(cls.wm)  # main settings page button, compared against WifiButtons inside wifi_ui
    cls.wifi_ui = WifiUIMici(cls.wm)

  @classmethod
  def tearDownClass(cls):
    gui_app.close()

  def _reset(self):
    self.wm.wifi_state = WifiState()
    self.wm._networks = []
    self.wm.ipv4_address = ""
    self.wm._saved_ssids = set()
    self.wifi_ui._scroller.items.clear()
    self.wifi_ui._networks.clear()
    self._complete_animations()

  def _complete_animations(self):
    """Simulate animation frames completing so move_item isn't blocked by stale state."""
    scroller = self.wifi_ui._scroller
    scroller._move_animations.clear()
    scroller._move_lift.clear()
    scroller._pending_move.clear()
    scroller._pending_lift.clear()

  @given(scenario=WIFI_SCENARIOS())
  @settings(max_examples=500, deadline=None, phases=(Phase.reuse, Phase.generate, Phase.shrink))
  def test_connection_status_consistent(self, scenario):
    """WifiNetworkButton and WifiButtons must agree on which network is active."""
    networks, wifi_state, ipv4, saved = scenario
    self._reset()

    self.wm.wifi_state = wifi_state
    self.wm._networks = networks
    self.wm.ipv4_address = ipv4
    self.wm._saved_ssids = set(saved)

    self.wifi_ui._on_network_updated(networks)
    self.net_btn._update_state()
    buttons = _get_buttons(self.wifi_ui)
    for btn in buttons.values():
      btn._update_state()

    active = wifi_state.ssid
    connecting = [b for b in buttons.values() if b._is_connecting]
    connected = [b for b in buttons.values() if b._is_connected]

    assert len(connecting) <= 1
    assert len(connected) <= 1
    for btn in buttons.values():
      assert not (btn._is_connecting and btn._is_connected)

    # Cross-widget: both must show the same ssid text
    if connecting:
      assert self.net_btn.text == connecting[0].text
    if connected:
      assert self.net_btn.text == connected[0].text

    # Converse: active ssid's button must reflect the status
    if active and active in buttons:
      if wifi_state.status == ConnectStatus.CONNECTING:
        assert buttons[active]._is_connecting
      elif wifi_state.status == ConnectStatus.CONNECTED:
        assert buttons[active]._is_connected

    # No non-active button should claim active
    for ssid, btn in buttons.items():
      if ssid != active:
        assert not btn._is_connecting, f"'{ssid}' claims connecting but active is '{active}'"
        assert not btn._is_connected, f"'{ssid}' claims connected but active is '{active}'"

    _check_disabled_invariant(buttons)
    _check_value_invariant(buttons)
    _check_net_btn_invariant(self.net_btn, self.wm)
    _check_forget_btn_invariant(buttons)

  @given(scenario=EAGER_SCENARIOS())
  @settings(max_examples=500, deadline=None, phases=(Phase.reuse, Phase.generate, Phase.shrink))
  def test_eager_state_interactions(self, scenario):
    """Eager state flags behave correctly through callback-driven sequences."""
    initial, actions = scenario
    self._reset()
    self.wifi_ui._on_network_updated(initial)

    for action_tuple in actions:
      action = action_tuple[0]
      buttons = _get_buttons(self.wifi_ui)

      if action == 'network_update':
        new_networks = action_tuple[1]
        new_ssids = {n.ssid for n in new_networks}
        wrong_pw_before = {s: b._wrong_password for s, b in buttons.items() if s in new_ssids}

        self.wifi_ui._on_network_updated(new_networks)

        for ssid, btn in _get_buttons(self.wifi_ui).items():
          if ssid in new_ssids:
            assert not btn._network_missing
            if btn._is_connected or btn._is_connecting:
              assert not btn._wrong_password
            elif ssid in wrong_pw_before:
              assert btn._wrong_password == wrong_pw_before[ssid]
          else:
            assert btn._network_missing

        ssids = [b.network.ssid for b in self.wifi_ui._scroller.items if isinstance(b, WifiButton)]
        assert len(ssids) == len(set(ssids)), f"Duplicate SSIDs: {ssids}"

      elif action == 'need_auth':
        btn = buttons.get(action_tuple[1])
        self.wifi_ui._on_need_auth(action_tuple[1])
        if btn is not None:
          assert btn._wrong_password

      elif action == 'forgotten':
        btn = buttons.get(action_tuple[1])
        self.wifi_ui._on_forgotten(action_tuple[1])
        if btn is not None:
          assert not btn._network_forgetting

      elif action == 'forget_btn':
        btn = buttons.get(action_tuple[1])
        if btn is not None:
          btn._forget_network()
          assert btn._network_forgetting

      elif action == 'connect':
        self.wm.wifi_state = WifiState(ssid=action_tuple[1], status=ConnectStatus.CONNECTING)

      elif action == 'disconnect':
        self.wm.wifi_state = WifiState()

      elif action == 'set_connected':
        self.wm.wifi_state = WifiState(ssid=action_tuple[1], status=ConnectStatus.CONNECTED)

      buttons = _get_buttons(self.wifi_ui)
      for btn in buttons.values():
        btn._update_state()
      _check_disabled_invariant(buttons)
      _check_value_invariant(buttons)
      _check_forget_btn_invariant(buttons)

  @given(scenario=CHAOS_SCENARIOS())
  @settings(max_examples=500, deadline=None, phases=(Phase.reuse, Phase.generate, Phase.shrink))
  def test_chaos_invariants(self, scenario):
    """Chaotic random sequences must never crash or violate behavioral invariants."""
    initial, actions = scenario
    self._reset()
    self.wifi_ui._on_network_updated(initial)

    for action_tuple in actions:
      action = action_tuple[0]
      buttons = _get_buttons(self.wifi_ui)

      if action == 'network_update':
        new_nets = action_tuple[1]
        new_ssids = {n.ssid for n in new_nets}
        self.wifi_ui._on_network_updated(new_nets)

        # Postcondition: missing flags must match scan results
        for ssid, btn in _get_buttons(self.wifi_ui).items():
          if ssid in new_ssids:
            assert not btn._network_missing, f"'{ssid}' should not be missing after scan update"
          else:
            assert btn._network_missing, f"'{ssid}' should be missing (not in scan)"

        # Postcondition: button's Network object must be the one from _networks
        for ssid, btn in _get_buttons(self.wifi_ui).items():
          if ssid in self.wifi_ui._networks:
            assert btn._network is self.wifi_ui._networks[ssid], \
              f"'{ssid}': button._network is stale, not synced with wifi_ui._networks"

        # Postcondition: active network (if present) should be at front
        active_ssid = self.wm.wifi_state.ssid
        if active_ssid and self.wm.wifi_state.status != ConnectStatus.DISCONNECTED:
          wifi_buttons = [b for b in self.wifi_ui._scroller.items if isinstance(b, WifiButton)]
          if wifi_buttons and any(b.network.ssid == active_ssid for b in wifi_buttons):
            assert wifi_buttons[0].network.ssid == active_ssid, \
              f"Active network '{active_ssid}' should be at front, got '{wifi_buttons[0].network.ssid}'"

      elif action == 'need_auth':
        ssid = action_tuple[1]
        btn = buttons.get(ssid)
        self.wifi_ui._on_need_auth(ssid)
        # Postcondition: button must have wrong_password set
        if btn is not None:
          assert btn._wrong_password, f"'{ssid}' should have _wrong_password after need_auth"

      elif action == 'need_auth_new' and len(action_tuple) > 1:
        self.wifi_ui._on_need_auth(action_tuple[1], incorrect_password=False)

      elif action == 'forgotten':
        ssid = action_tuple[1] if len(action_tuple) > 1 else 'nonexistent'
        btn = buttons.get(ssid)
        self.wifi_ui._on_forgotten(ssid)
        # Postcondition: forgetting flag must be cleared
        if btn is not None:
          assert not btn._network_forgetting, f"'{ssid}' should not be forgetting after on_forgotten"

      elif action == 'forget_btn':
        ssid = action_tuple[1] if len(action_tuple) > 1 else None
        btn = buttons.get(ssid)
        if btn is not None:
          was_forgetting = btn._network_forgetting
          btn._forget_network()
          # Postcondition: must be forgetting now
          assert btn._network_forgetting, f"'{ssid}' should be forgetting after _forget_network"
          # Idempotency: calling again shouldn't crash or change anything (guard in code)

      elif action == 'user_click':
        ssid = action_tuple[1]
        network = self.wifi_ui._networks.get(ssid)
        state_before = WifiState(ssid=self.wm.wifi_state.ssid, status=self.wm.wifi_state.status)
        self.wifi_ui._connect_to_network(ssid)

        if network is not None:
          if self.wm.is_connection_saved(ssid) or network.security_type == SecurityType.OPEN:
            # Postcondition: should now be connecting to this network
            assert self.wm.wifi_state.ssid == ssid, \
              f"After clicking saved/open '{ssid}', wifi_state.ssid={self.wm.wifi_state.ssid!r}"
            assert self.wm.wifi_state.status == ConnectStatus.CONNECTING, \
              f"After clicking saved/open '{ssid}', status={self.wm.wifi_state.status}"
          elif network.security_type != SecurityType.UNSUPPORTED:
            # Secured + not saved: should NOT change wifi_state (just opens dialog)
            assert self.wm.wifi_state.ssid == state_before.ssid, \
              f"Clicking secured unsaved '{ssid}' shouldn't change wifi_state"
            assert self.wm.wifi_state.status == state_before.status
        else:
          # Unknown ssid: state shouldn't change
          assert self.wm.wifi_state.ssid == state_before.ssid
          assert self.wm.wifi_state.status == state_before.status

      elif action == 'connect_with_password':
        ssid, password = action_tuple[1], action_tuple[2]
        self.wifi_ui._connect_with_password(ssid, password)
        # Postcondition: should be connecting now
        assert self.wm.wifi_state.ssid == ssid, \
          f"After connect_with_password '{ssid}', wifi_state.ssid={self.wm.wifi_state.ssid!r}"
        assert self.wm.wifi_state.status == ConnectStatus.CONNECTING

      elif action == 'connect':
        self.wm.wifi_state = WifiState(ssid=action_tuple[1], status=ConnectStatus.CONNECTING)
      elif action == 'disconnect':
        self.wm.wifi_state = WifiState()
      elif action == 'set_connected':
        self.wm.wifi_state = WifiState(ssid=action_tuple[1], status=ConnectStatus.CONNECTED)
      elif action == 'change_saved':
        self.wm._saved_ssids = set(action_tuple[1])
      elif action == 'change_ipv4':
        self.wm.ipv4_address = action_tuple[1]
      elif action == 'show_event':
        self.wifi_ui.hide_event()
        self._complete_animations()
        self.wifi_ui.show_event()
        # Postcondition: all buttons should have fresh eager state (recreated)
        for btn in _get_buttons(self.wifi_ui).values():
          assert not btn._wrong_password, f"'{btn.network.ssid}' has stale _wrong_password after show_event"
          assert not btn._network_forgetting, f"'{btn.network.ssid}' has stale _network_forgetting after show_event"
      elif action == 'hide_event':
        pass

      # Simulate frames rendered between actions (clears animation locks)
      self._complete_animations()

      self.net_btn._update_state()
      buttons = _get_buttons(self.wifi_ui)
      for btn in buttons.values():
        btn._update_state()

      # -- Global behavioral invariants that must hold after ANY operation --

      connecting = [b for b in buttons.values() if b._is_connecting]
      connected = [b for b in buttons.values() if b._is_connected]

      # No duplicate SSIDs
      ssids = [b.network.ssid for b in self.wifi_ui._scroller.items if isinstance(b, WifiButton)]
      assert len(ssids) == len(set(ssids)), f"Duplicate SSIDs: {ssids}"

      assert len(connecting) <= 1
      assert len(connected) <= 1
      for btn in buttons.values():
        assert not (btn._is_connecting and btn._is_connected)

      if connecting:
        assert self.net_btn.text == connecting[0].text, \
          f"net_btn.text={self.net_btn.text!r} != connecting btn text={connecting[0].text!r}"
      if connected:
        assert self.net_btn.text == connected[0].text, \
          f"net_btn.text={self.net_btn.text!r} != connected btn text={connected[0].text!r}"

      active = self.wm.wifi_state.ssid
      for ssid, btn in buttons.items():
        if ssid != active:
          assert not btn._is_connecting, f"'{ssid}' claims connecting but active is '{active}'"
          assert not btn._is_connected, f"'{ssid}' claims connected but active is '{active}'"

      _check_disabled_invariant(buttons)
      _check_value_invariant(buttons)
      _check_net_btn_invariant(self.net_btn, self.wm)
      _check_forget_btn_invariant(buttons)

  def _render_frame(self, mouse_events=None):
    """Render one frame with optional mouse events injected."""
    gui_app._mouse_events = mouse_events or []
    if mouse_events:
      gui_app._last_mouse_event = mouse_events[-1]
    rl.begin_drawing()
    try:
      self.wifi_ui.render(RENDER_RECT)
    finally:
      rl.end_drawing()
      gui_app._mouse_events = []

  @given(scenario=MOUSE_INPUT_SCENARIOS())
  @settings(max_examples=500, deadline=None, phases=(Phase.reuse, Phase.generate, Phase.shrink))
  def test_fuzzed_mouse_no_lockup(self, scenario):
    """Random mouse inputs interleaved with state changes must not crash or lock up."""
    initial, steps = scenario
    self._reset()
    self.wifi_ui._on_network_updated(initial)
    self.wifi_ui.set_rect(RENDER_RECT)

    # Prevent dialogs piling up on nav stack from button clicks
    pushed_widgets = []
    original_push = gui_app.push_widget
    gui_app.push_widget = lambda w: pushed_widgets.append(w)

    try:
      for step in steps:
        action = step[0]
        t = time.monotonic()

        if action == 'click':
          _, x, y = step
          pos = MousePos(x, y)
          self._render_frame([MouseEvent(pos, 0, True, False, True, t)])
          self._render_frame([MouseEvent(pos, 0, False, True, False, t + 0.016)])

        elif action == 'press_hold_release':
          _, x1, y1, x2, y2 = step
          self._render_frame([MouseEvent(MousePos(x1, y1), 0, True, False, True, t)])
          self._render_frame([MouseEvent(MousePos(x2, y2), 0, False, False, True, t + 0.05)])
          self._render_frame([MouseEvent(MousePos(x2, y2), 0, False, True, False, t + 0.1)])

        elif action == 'drag':
          points = step[1]
          self._render_frame([MouseEvent(MousePos(*points[0]), 0, True, False, True, t)])
          for i, (px, py) in enumerate(points[1:-1], 1):
            self._render_frame([MouseEvent(MousePos(px, py), 0, False, False, True, t + i * 0.016)])
          self._render_frame([MouseEvent(MousePos(*points[-1]), 0, False, True, False, t + len(points) * 0.016)])

        elif action == 'multi_touch':
          _, slot, x, y, pressed = step
          pos = MousePos(x, y)
          self._render_frame([MouseEvent(pos, slot, pressed, not pressed, pressed, t)])

        elif action == 'rapid_clicks':
          for i, (cx, cy) in enumerate(step[1]):
            pos = MousePos(cx, cy)
            self._render_frame([MouseEvent(pos, 0, True, False, True, t + i * 0.03)])
            self._render_frame([MouseEvent(pos, 0, False, True, False, t + i * 0.03 + 0.01)])

        elif action == 'long_press':
          _, x, y, hold_frames = step
          pos = MousePos(x, y)
          self._render_frame([MouseEvent(pos, 0, True, False, True, t)])
          for i in range(hold_frames):
            self._render_frame([MouseEvent(pos, 0, False, False, True, t + (i + 1) * 0.016)])
          self._render_frame([MouseEvent(pos, 0, False, True, False, t + (hold_frames + 1) * 0.016)])

        elif action == 'render_frame':
          self._render_frame()

        elif action == 'network_update':
          self.wifi_ui._on_network_updated(step[1])
          self._complete_animations()
        elif action == 'connect':
          self.wm.wifi_state = WifiState(ssid=step[1], status=ConnectStatus.CONNECTING)
        elif action == 'disconnect':
          self.wm.wifi_state = WifiState()
        elif action == 'set_connected':
          self.wm.wifi_state = WifiState(ssid=step[1], status=ConnectStatus.CONNECTED)
        elif action == 'show_hide':
          self.wifi_ui.hide_event()
          self._complete_animations()
          self.wifi_ui.show_event()
        elif action == 'need_auth':
          self.wifi_ui._on_need_auth(step[1])
        elif action == 'forgotten':
          self.wifi_ui._on_forgotten(step[1])
        elif action == 'change_saved':
          self.wm._saved_ssids = set(step[1])
        elif action == 'change_ipv4':
          self.wm.ipv4_address = step[1]

        pushed_widgets.clear()

      # Final invariant check: no duplicate SSIDs, button states consistent
      self._complete_animations()
      self._render_frame()

      buttons = _get_buttons(self.wifi_ui)
      for btn in buttons.values():
        btn._update_state()

      ssids = [b.network.ssid for b in self.wifi_ui._scroller.items if isinstance(b, WifiButton)]
      assert len(ssids) == len(set(ssids)), f"Duplicate SSIDs after mouse fuzz: {ssids}"

      _check_disabled_invariant(buttons)
      _check_value_invariant(buttons)

      # No permanently stuck animation state
      scroller = self.wifi_ui._scroller
      assert not scroller.moving_items, "Scroller stuck in move animation after test"

    finally:
      gui_app.push_widget = original_push
      gui_app._mouse_events = []

  def test_connect_dismiss_reopen_state(self):
    """Connecting, dismissing, and immediately reopening should not trigger a move animation.

    BUG: show_event clears items and rebuilds from scratch via _update_buttons.
    _update_buttons adds items in _networks dict order, then calls _move_network_to_front
    which triggers move_item — starting a full animation (lift, overlay dim, move, drop).
    Since buttons were just created with rect.x=0, the animation starts from wrong positions,
    causing items to slide in from the left and a dark overlay to briefly (or permanently) appear.
    Fix: build items in correct order during fresh rebuild, or skip animation."""
    self._reset()
    nets = [
      Network(ssid='Alpha', strength=80, security_type=SecurityType.OPEN, is_tethering=False),
      Network(ssid='Bravo', strength=60, security_type=SecurityType.OPEN, is_tethering=False),
      Network(ssid='Charlie', strength=40, security_type=SecurityType.OPEN, is_tethering=False),
    ]
    self.wm._networks = nets
    self.wifi_ui._on_network_updated(nets)
    self._complete_animations()

    # User taps "Bravo" to connect (open network, so it calls connect_to_network directly)
    self.wifi_ui._connect_to_network('Bravo')
    # Now wifi_state is CONNECTING for Bravo, move animation started

    # Verify move started: Bravo should be at front of items list (move_item does immediate reorder)
    wifi_buttons = [b for b in self.wifi_ui._scroller.items if isinstance(b, WifiButton)]
    assert wifi_buttons[0].network.ssid == 'Bravo'

    # User immediately dismisses
    self.wifi_ui.hide_event()

    # Scroller animation state should be fully cleared
    scroller = self.wifi_ui._scroller
    assert not scroller.moving_items, "hide_event should clear all move animation state"
    assert scroller._overlay_filter.x == 0.0, "hide_event should reset overlay"
    assert scroller._scrolling_to == (None, False), "hide_event should cancel scroll animation"

    # User immediately re-opens
    self.wifi_ui.show_event()

    # After show_event, check the scroller state WITHOUT clearing animations
    wifi_buttons = [b for b in self.wifi_ui._scroller.items if isinstance(b, WifiButton)]

    # Connecting network should be at front (Bravo is still CONNECTING)
    assert wifi_buttons[0].network.ssid == 'Bravo', \
      f"Connecting network should be at front after reopen, got '{wifi_buttons[0].network.ssid}'"

    # Scroll position should be at start
    assert scroller.scroll_panel.get_offset() == 0.0, \
      f"Scroll should be at start after reopen, got {scroller.scroll_panel.get_offset()}"

    # show_event should NOT trigger a move animation that causes the overlay to appear.
    # When rebuilding buttons from scratch, the connecting network should be placed at front
    # during _update_buttons WITHOUT needing a move animation.
    assert not scroller.moving_items, \
      "show_event should not start a move animation — items should be built in correct order"
    assert len(scroller._pending_move) == 0, \
      f"Stale pending_move after reopen: {len(scroller._pending_move)} items"
    assert scroller._overlay_filter.x == 0.0, \
      f"Overlay should not be active after reopen, got {scroller._overlay_filter.x}"

    # No stale pending animations referencing old widget objects
    for item in scroller._pending_move:
      assert item in scroller._items, "pending_move references widget not in scroller"
    for item in scroller._pending_lift:
      assert item in scroller._items, "pending_lift references widget not in scroller"
    for item in scroller._move_animations:
      assert item in scroller._items, "move_animations references widget not in scroller"
    for item in scroller._move_lift:
      assert item in scroller._items, "move_lift references widget not in scroller"

  def test_wrong_password_should_reprompt(self):
    """After wrong password, tapping a saved network should prompt for a new password, not retry saved creds."""
    self._reset()
    net = Network(ssid='TestWPA2', strength=80, security_type=SecurityType.WPA2, is_tethering=False)
    self.wm._networks = [net]
    self.wifi_ui._on_network_updated([net])
    self._complete_animations()

    # Simulate: connect_to_network created an NM profile, then auth failed
    self.wm._saved_ssids = {'TestWPA2'}
    self.wifi_ui._on_need_auth('TestWPA2')
    btn = _get_buttons(self.wifi_ui)['TestWPA2']
    btn._update_state()
    assert btn._wrong_password
    assert btn.value == "wrong password"
    assert btn.enabled

    # User taps the button again
    state_before = WifiState(ssid=self.wm.wifi_state.ssid, status=self.wm.wifi_state.status)
    self.wifi_ui._connect_to_network('TestWPA2')

    # BUG: _connect_to_network sees is_connection_saved=True and calls activate_connection,
    # which retries with the same wrong password. It should prompt for a new password instead.
    assert self.wm.wifi_state.status == state_before.status, \
      f"Should prompt for new password, not activate saved connection with wrong creds. " \
      f"wifi_state changed to {self.wm.wifi_state.status}"

  def test_forgotten_clears_wrong_password(self):
    """After forgetting a network, _wrong_password should be cleared."""
    self._reset()
    net = Network(ssid='TestNet', strength=80, security_type=SecurityType.WPA2, is_tethering=False)
    self.wm._networks = [net]
    self.wifi_ui._on_network_updated([net])
    self._complete_animations()

    # Set wrong password, then forget the network
    self.wifi_ui._on_need_auth('TestNet')
    btn = _get_buttons(self.wifi_ui)['TestNet']
    assert btn._wrong_password

    btn._forget_network()
    assert btn._network_forgetting

    # NM completes the forget
    self.wifi_ui._on_forgotten('TestNet')
    assert not btn._network_forgetting

    # BUG: on_forgotten doesn't clear _wrong_password. The network was just wiped,
    # showing "wrong password" for a fresh network makes no sense.
    btn._update_state()
    assert not btn._wrong_password, \
      "After forgetting a network, _wrong_password should be cleared"

  def test_forget_visible_with_wrong_password(self):
    """Saved network with wrong password should still show forget button."""
    self._reset()
    net = Network(ssid='TestNet', strength=80, security_type=SecurityType.WPA2, is_tethering=False)
    self.wm._networks = [net]
    self.wm._saved_ssids = {'TestNet'}
    self.wifi_ui._on_network_updated([net])
    self._complete_animations()

    self.wifi_ui._on_need_auth('TestNet')
    btn = _get_buttons(self.wifi_ui)['TestNet']
    btn._update_state()
    assert btn._wrong_password
    assert btn._is_saved

    # BUG: _show_forget_btn = (saved and not wrong_password) = (True and False) = False
    # The user has no way to forget the bad saved credentials. Combined with
    # test_wrong_password_should_reprompt, this creates a dead end where the user is stuck
    # retrying wrong credentials with no escape.
    assert btn._show_forget_btn, \
      "Saved network with wrong password should show forget button so user can clear bad credentials"


if __name__ == "__main__":
  unittest.main()
