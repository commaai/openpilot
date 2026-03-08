#!/usr/bin/env python3
"""Fetch CI results from GitHub Actions and Jenkins."""

import argparse
import json
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime

JENKINS_URL = "https://jenkins.comma.life"
DEFAULT_TIMEOUT = 1800  # 30 minutes
POLL_INTERVAL = 30  # seconds
LOG_TAIL_LINES = 10  # lines of log to include for failed jobs


def get_git_info():
  branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip()
  commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
  return branch, commit


def get_github_actions_status(commit_sha):
  result = subprocess.run(
    ["gh", "run", "list", "--commit", commit_sha, "--workflow", "tests.yaml", "--json", "databaseId,status,conclusion"],
    capture_output=True, text=True, check=True
  )
  runs = json.loads(result.stdout)
  if not runs:
    return None, None

  run_id = runs[0]["databaseId"]
  result = subprocess.run(
    ["gh", "run", "view", str(run_id), "--json", "jobs"],
    capture_output=True, text=True, check=True
  )
  data = json.loads(result.stdout)
  jobs = {job["name"]: {"status": job["status"], "conclusion": job["conclusion"],
                        "duration": format_duration(job) if job["conclusion"] not in ("skipped", None) and job.get("startedAt") else "",
                        "id": job["databaseId"]}
          for job in data.get("jobs", [])}
  return jobs, run_id


def get_github_job_log(run_id, job_id):
  result = subprocess.run(
    ["gh", "run", "view", str(run_id), "--job", str(job_id), "--log-failed"],
    capture_output=True, text=True
  )
  lines = result.stdout.strip().split('\n')
  return '\n'.join(lines[-LOG_TAIL_LINES:]) if len(lines) > LOG_TAIL_LINES else result.stdout.strip()


def format_duration(job):
  start = datetime.fromisoformat(job["startedAt"].replace("Z", "+00:00"))
  end = datetime.fromisoformat(job["completedAt"].replace("Z", "+00:00"))
  secs = int((end - start).total_seconds())
  return f"{secs // 60}m {secs % 60}s"


def get_jenkins_status(branch, commit_sha):
  base_url = f"{JENKINS_URL}/job/openpilot/job/{branch}"
  try:
    # Get list of recent builds
    with urllib.request.urlopen(f"{base_url}/api/json?tree=builds[number,url]", timeout=10) as resp:
      builds = json.loads(resp.read().decode()).get("builds", [])

    # Find build matching commit
    for build in builds[:20]:  # check last 20 builds
      with urllib.request.urlopen(f"{build['url']}api/json", timeout=10) as resp:
        data = json.loads(resp.read().decode())
        for action in data.get("actions", []):
          if action.get("_class") == "hudson.plugins.git.util.BuildData":
            build_sha = action.get("lastBuiltRevision", {}).get("SHA1", "")
            if build_sha.startswith(commit_sha) or commit_sha.startswith(build_sha):
              # Get stages info
              stages = []
              try:
                with urllib.request.urlopen(f"{build['url']}wfapi/describe", timeout=10) as resp2:
                  wf_data = json.loads(resp2.read().decode())
                  stages = [{"name": s["name"], "status": s["status"]} for s in wf_data.get("stages", [])]
              except urllib.error.HTTPError:
                pass
              return {
                "number": data["number"],
                "in_progress": data.get("inProgress", False),
                "result": data.get("result"),
                "url": data.get("url", ""),
                "stages": stages,
              }
    return None  # no build found for this commit
  except urllib.error.HTTPError:
    return None  # branch doesn't exist on Jenkins


def get_jenkins_log(build_url):
  url = f"{build_url}consoleText"
  with urllib.request.urlopen(url, timeout=30) as resp:
    text = resp.read().decode(errors='replace')
    lines = text.strip().split('\n')
    return '\n'.join(lines[-LOG_TAIL_LINES:]) if len(lines) > LOG_TAIL_LINES else text.strip()


def is_complete(gh_status, jenkins_status):
  gh_done = gh_status is None or all(j["status"] == "completed" for j in gh_status.values())
  jenkins_done = jenkins_status is None or not jenkins_status.get("in_progress", True)
  return gh_done and jenkins_done


def status_icon(status, conclusion=None):
  if status == "completed":
    return ":white_check_mark:" if conclusion == "success" else ":x:"
  return ":hourglass:" if status == "in_progress" else ":grey_question:"


def format_markdown(gh_status, gh_run_id, jenkins_status, commit_sha, branch):
  lines = ["# CI Results", "",
           f"**Branch**: {branch}",
           f"**Commit**: {commit_sha[:7]}",
           f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ""]

  lines.extend(["## GitHub Actions", "", "| Job | Status | Duration |", "|-----|--------|----------|"])
  failed_gh_jobs = []
  if gh_status:
    for job_name, job in gh_status.items():
      icon = status_icon(job["status"], job.get("conclusion"))
      conclusion = job.get("conclusion") or job["status"]
      lines.append(f"| {job_name} | {icon} {conclusion} | {job.get('duration', '')} |")
      if job.get("conclusion") == "failure":
        failed_gh_jobs.append((job_name, job.get("id")))
  else:
    lines.append("| - | No workflow runs found | |")

  lines.extend(["", "## Jenkins", "", "| Stage | Status |", "|-------|--------|"])
  failed_jenkins_stages = []
  if jenkins_status:
    stages = jenkins_status.get("stages", [])
    if stages:
      for stage in stages:
        icon = ":white_check_mark:" if stage["status"] == "SUCCESS" else (
          ":x:" if stage["status"] == "FAILED" else ":hourglass:")
        lines.append(f"| {stage['name']} | {icon} {stage['status'].lower()} |")
        if stage["status"] == "FAILED":
          failed_jenkins_stages.append(stage["name"])
      # Show overall build status if still in progress
      if jenkins_status["in_progress"]:
        lines.append("| (build in progress) | :hourglass: in_progress |")
    else:
      icon = ":hourglass:" if jenkins_status["in_progress"] else (
        ":white_check_mark:" if jenkins_status["result"] == "SUCCESS" else ":x:")
      status = "in progress" if jenkins_status["in_progress"] else (jenkins_status["result"] or "unknown")
      lines.append(f"| #{jenkins_status['number']} | {icon} {status.lower()} |")
    if jenkins_status.get("url"):
      lines.append(f"\n[View build]({jenkins_status['url']})")
  else:
    lines.append("| - | No builds found for branch |")

  if failed_gh_jobs or failed_jenkins_stages:
    lines.extend(["", "## Failure Logs", ""])

  for job_name, job_id in failed_gh_jobs:
    lines.append(f"### GitHub Actions: {job_name}")
    log = get_github_job_log(gh_run_id, job_id)
    lines.extend(["", "```", log, "```", ""])

  for stage_name in failed_jenkins_stages:
    lines.append(f"### Jenkins: {stage_name}")
    log = get_jenkins_log(jenkins_status["url"])
    lines.extend(["", "```", log, "```", ""])

  return "\n".join(lines) + "\n"


def main():
  parser = argparse.ArgumentParser(description="Fetch CI results from GitHub Actions and Jenkins")
  parser.add_argument("--wait", action="store_true", help="Wait for CI to complete")
  parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="Timeout in seconds (default: 1800)")
  parser.add_argument("-o", "--output", default="ci_results.md", help="Output file (default: ci_results.md)")
  parser.add_argument("--branch", help="Branch to check (default: current branch)")
  parser.add_argument("--commit", help="Commit SHA to check (default: HEAD)")
  args = parser.parse_args()

  branch, commit = get_git_info()
  branch = args.branch or branch
  commit = args.commit or commit
  print(f"Fetching CI results for {branch} @ {commit[:7]}")

  start_time = time.monotonic()
  while True:
    gh_status, gh_run_id = get_github_actions_status(commit)
    jenkins_status = get_jenkins_status(branch, commit) if branch != "HEAD" else None

    if not args.wait or is_complete(gh_status, jenkins_status):
      break

    elapsed = time.monotonic() - start_time
    if elapsed >= args.timeout:
      print(f"Timeout after {int(elapsed)}s")
      break

    print(f"CI still running, waiting {POLL_INTERVAL}s... ({int(elapsed)}s elapsed)")
    time.sleep(POLL_INTERVAL)

  content = format_markdown(gh_status, gh_run_id, jenkins_status, commit, branch)
  with open(args.output, "w") as f:
    f.write(content)
  print(f"Results written to {args.output}")


if __name__ == "__main__":
  main()
