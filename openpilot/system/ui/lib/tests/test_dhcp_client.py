import os
import subprocess
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import MagicMock, patch

from openpilot.system.ui.lib import dhcp_client as dhcp_client_module
from openpilot.system.ui.lib.dhcp_client import DhcpClient


class TestDhcpClient(TestCase):
  def setUp(self):
    self.enterContext(patch.object(dhcp_client_module.os, "access", return_value=True))

  def test_adopt_existing_udhcpc_without_restarting_it(self):
    client = DhcpClient()
    with (
      patch.object(client, "_owned_pid", return_value=123),
      patch.object(dhcp_client_module.subprocess, "Popen") as popen,
      patch.object(dhcp_client_module.threading, "Thread") as thread,
    ):
      assert client.adopt()

      popen.assert_not_called()
      thread.assert_called_once_with(target=client._monitor_client, daemon=True)
      thread.return_value.start.assert_called_once()

  def test_pidfile_rejects_foreign_udhcpc_command(self):
    client = DhcpClient()
    command = b"udhcpc\0-i\0wlan1\0-p\0/run/foreign.pid\0-s\0/tmp/script\0"
    with (
      patch.object(Path, "read_text", return_value="123"),
      patch.object(Path, "read_bytes", return_value=command),
    ):
      assert client._owned_pid() is None

  def test_start_flushes_stale_lease_and_detaches_udhcpc_from_ui_session(self):
    client = DhcpClient()
    events = []
    with (
      patch.object(dhcp_client_module.subprocess, "run", side_effect=lambda command, **_: events.append(command)) as run,
      patch.object(dhcp_client_module.subprocess, "Popen", side_effect=lambda *_, **__: events.append("spawn") or MagicMock()) as popen,
      patch.object(dhcp_client_module.threading, "Thread") as thread,
    ):
      client.start()

      assert [call.args[0] for call in run.call_args_list] == [
        ["sudo", "rm", "-f", client._pid_file],
        ["sudo", "ip", "-4", "route", "flush", "dev", "wlan0"],
        ["sudo", "ip", "-4", "addr", "flush", "dev", "wlan0"],
        ["sudo", "install", "-d", "-o", "root", "-g", "root", "-m", "755", dhcp_client_module.DHCP_RUNTIME_DIR],
        ["sudo", "rm", "-f", client._pid_file],
      ]
      assert events == [*([call.args[0] for call in run.call_args_list]), "spawn"]
      popen.assert_called_once_with(
        [
          "sudo", "udhcpc", "-i", "wlan0", "-f", "-t", "5", "-T", "3",
          "-p", client._pid_file, "-s", dhcp_client_module.DHCP_SCRIPT,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
      )
      thread.assert_called_once_with(target=client._monitor_client, daemon=True)
      thread.return_value.start.assert_called_once()

  def test_failed_launch_starts_client_monitor(self):
    client = DhcpClient()
    with (
      patch.object(client, "stop"),
      patch.object(dhcp_client_module.subprocess, "run"),
      patch.object(dhcp_client_module.subprocess, "Popen", side_effect=OSError("exec failed")),
      patch.object(dhcp_client_module.threading, "Thread") as thread,
    ):
      client.start()

    targets = [item.kwargs["target"] for item in thread.call_args_list]
    assert client._monitor_client in targets

  def test_missing_default_script_is_reported_before_launch(self):
    client = DhcpClient()
    with (
      patch.object(dhcp_client_module.os, "access", return_value=False),
      patch.object(dhcp_client_module.cloudlog, "error") as error,
      patch.object(dhcp_client_module.subprocess, "Popen") as popen,
    ):
      assert not client._spawn()

    error.assert_called_once_with(f"udhcpc default script is not executable: {dhcp_client_module.DHCP_DEFAULT_SCRIPT}")
    popen.assert_not_called()

  def test_exited_client_is_restarted(self):
    client = DhcpClient()
    client._proc = MagicMock()
    client._proc.poll.return_value = 1

    with (
      patch.object(client._client_stop, "wait", side_effect=[False, True]),
      patch.object(
        dhcp_client_module.subprocess,
        "run",
        return_value=MagicMock(returncode=1),
      ),
      patch.object(client, "_flush_address") as flush_address,
      patch.object(client, "_spawn", return_value=True) as spawn,
    ):
      client._monitor_client()

    flush_address.assert_called_once()
    spawn.assert_called_once()

  def test_dhcp_script_applies_metric_after_default_script(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      root = Path(temp_dir)
      trace = root / "trace"
      default_script = root / "default.script"
      default_script.write_text('#!/bin/sh\nprintf "default %s\\n" "$1" >> "$TRACE"\n')
      default_script.chmod(0o755)
      ip = root / "ip"
      ip.write_text("""#!/bin/sh
printf "ip %s\\n" "$*" >> "$TRACE"
[ "$*" = "-4 route show default dev wlan0" ] && printf "default via 192.168.1.1 dev wlan0 metric 600\\n"
exit 0
""")
      ip.chmod(0o755)
      env = {
        **os.environ,
        "PATH": f"{root}:{os.environ['PATH']}",
        "TRACE": str(trace),
        "UDHCPC_DEFAULT_SCRIPT": str(default_script),
        "interface": "wlan0",
        "router": "192.168.1.1 192.168.1.2",
      }

      subprocess.run([dhcp_client_module.DHCP_SCRIPT, "renew"], env=env, check=True)

      assert trace.read_text().splitlines() == [
        "default renew",
        "ip -4 route flush default dev wlan0",
        "ip -4 route replace default via 192.168.1.1 dev wlan0 metric 600",
        "ip -4 route show default dev wlan0",
      ]
  def test_dhcp_script_propagates_route_install_failure(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      root = Path(temp_dir)
      default_script = root / "default.script"
      default_script.write_text("#!/bin/sh\nexit 0\n")
      default_script.chmod(0o755)
      ip = root / "ip"
      ip.write_text("#!/bin/sh\nexit 1\n")
      ip.chmod(0o755)
      env = {
        **os.environ,
        "PATH": f"{root}:{os.environ['PATH']}",
        "UDHCPC_DEFAULT_SCRIPT": str(default_script),
        "interface": "wlan0",
        "router": "192.168.1.1",
      }

      result = subprocess.run([dhcp_client_module.DHCP_SCRIPT, "bound"], env=env, check=False)
      assert result.returncode != 0

  def test_dhcp_script_rejects_noncanonical_default_routes(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      root = Path(temp_dir)
      default_script = root / "default.script"
      default_script.write_text("#!/bin/sh\nexit 0\n")
      default_script.chmod(0o755)
      ip = root / "ip"
      ip.write_text("""\
#!/bin/sh
if [ "$*" = "-4 route show default dev wlan0" ]; then
  printf "default via 192.168.1.1 dev wlan0 metric 600\\ndefault via 192.168.1.2 dev wlan0 metric 0\\n"
fi
""")
      ip.chmod(0o755)
      env = {
        **os.environ,
        "PATH": f"{root}:{os.environ['PATH']}",
        "UDHCPC_DEFAULT_SCRIPT": str(default_script),
        "interface": "wlan0",
        "router": "192.168.1.1",
      }

      result = subprocess.run([dhcp_client_module.DHCP_SCRIPT, "renew"], env=env, check=False)
      assert result.returncode != 0


  def test_stop_cleans_wlan_dhcp_state_with_or_without_process_handle(self):
    for proc in (MagicMock(), None):
      with self.subTest(has_process_handle=proc is not None):
        client = DhcpClient()
        client._proc = proc
        with patch.object(dhcp_client_module.subprocess, "run") as run:
          client.stop()

        assert client._proc is None
        assert [call.args[0] for call in run.call_args_list] == [
          ["sudo", "rm", "-f", client._pid_file],
          ["sudo", "ip", "-4", "route", "flush", "dev", "wlan0"],
          ["sudo", "ip", "-4", "addr", "flush", "dev", "wlan0"],
        ]

  def test_stop_kills_only_pidfile_owned_adopted_client(self):
    client = DhcpClient()
    with (
      patch.object(client, "_owned_pid", return_value=123),
      patch.object(dhcp_client_module.subprocess, "run") as run,
    ):
      client.stop()

    assert [call.args[0] for call in run.call_args_list] == [
      ["sudo", "kill", "123"],
      ["sudo", "rm", "-f", client._pid_file],
      ["sudo", "ip", "-4", "route", "flush", "dev", "wlan0"],
      ["sudo", "ip", "-4", "addr", "flush", "dev", "wlan0"],
    ]

  def test_clear_ipv6_state_cleans_global_addresses_and_routes(self):
    client = DhcpClient()
    with patch.object(dhcp_client_module.subprocess, "run", return_value=MagicMock(returncode=0)) as run:
      client.clear_ipv6_state()

    assert [item.args[0] for item in run.call_args_list] == [
      ["sudo", "ip", "-6", "addr", "flush", "dev", "wlan0", "scope", "global"],
      ["sudo", "ip", "-6", "route", "del", "default", "dev", "wlan0"],
      ["sudo", "ip", "-6", "route", "flush", "dev", "wlan0"],
    ]

  def test_clear_ipv6_state_ignores_absent_default_route(self):
    client = DhcpClient()
    results = (
      MagicMock(returncode=0, stderr=b""),
      MagicMock(returncode=2, stderr=b"RTNETLINK answers: No such process\n"),
      MagicMock(returncode=0, stderr=b""),
    )
    with (
      patch.object(dhcp_client_module.subprocess, "run", side_effect=results),
      patch.object(dhcp_client_module.cloudlog, "warning") as warning,
    ):
      client.clear_ipv6_state()

    warning.assert_not_called()

  def test_ipv6_policy_is_applied_once_per_value(self):
    client = DhcpClient()
    with (
      patch.object(dhcp_client_module.subprocess, "run") as run,
      patch.object(Path, "read_text", return_value="1\n"),
      patch.object(client, "clear_ipv6_state") as clear_ipv6_state,
    ):
      client.set_ipv6_enabled(False)
      client.set_ipv6_enabled(False)

    run.assert_called_once_with(["sudo", "sysctl", "net.ipv6.conf.wlan0.disable_ipv6=1"], check=True)
    clear_ipv6_state.assert_called_once()

  def test_ipv6_policy_verifies_kernel_state(self):
    client = DhcpClient()
    with (
      patch.object(dhcp_client_module.subprocess, "run"),
      patch.object(Path, "read_text", return_value="1\n"),
    ):
      with self.assertRaisesRegex(RuntimeError, "actual='1'"):
        client.set_ipv6_enabled(True)
