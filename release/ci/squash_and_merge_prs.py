#!/usr/bin/env python3

import subprocess
import sys
import os
import argparse
import json
from datetime import datetime

TRUST_FORK_LABEL = "trust-fork-pr"

def setup_argument_parser():
  parser = argparse.ArgumentParser(description='Process and squash GitHub PRs')
  parser.add_argument('--pr-data', type=str, help='PR data in JSON format')
  parser.add_argument('--source-branch', type=str, default='master-new',
                      help='Source branch for merging')
  parser.add_argument('--target-branch', type=str, default='master-dev-c3-new-test',
                      help='Target branch for merging')
  parser.add_argument('--squash-script-path', type=str, required=True,
                      help='Path to the squash_and_merge.py script')
  return parser


def validate_squash_script(script_path):
  if not os.path.isfile(script_path):
    raise FileNotFoundError(f"Squash script not found at: {script_path}")
  if not os.access(script_path, os.X_OK):
    raise PermissionError(f"Squash script is not executable: {script_path}")


def sort_prs_by_creation(pr_data):
  """Sort PRs by creation date"""
  nodes = (pr_data.get('data', {}).get('search', {}).get('nodes', []))

  return sorted(
    nodes,
    key=lambda x: datetime.fromisoformat(x.get('createdAt', '').replace('Z', '+00:00'))
  )


def add_pr_comments(pr_number, comments: list[str]):
  """Adds or updates a comment with multiple comments to a PR using gh cli"""
  comment = "\n___\n".join(comments)
  _add_pr_comment(pr_number, comment)


def _add_pr_comment(pr_number, comment):
  """Add or update a comment to a PR using gh cli"""
  title = "## Squash and Merge"

  try:
    full_comment = f"{title}\n\n{comment}"
    subprocess.run(
      ['gh', 'pr', 'comment', '--edit-last', '--create-if-none', f"#{pr_number}", '--body', full_comment],
      check=True,
      capture_output=True,
      text=True
    )

  except subprocess.CalledProcessError as e:
    print(f"Failed to add/update comment on PR #{pr_number}: {e.stderr}")
  except json.JSONDecodeError:
    print(f"Failed to parse comments data for PR #{pr_number}")


def validate_pr(pr):
  """Validate a PR and return (is_valid, skip_reason)"""
  pr_number = pr.get('number', 'UNKNOWN')
  branch = pr.get('headRefName', '')

  if not branch:
    return False, f"missing branch name for PR #{pr_number}"

  # Check if checks have passed
  commits = pr.get('commits', {}).get('nodes', [])
  if not commits:
    return False, "no commit data found"

  status = commits[0].get('commit', {}).get('statusCheckRollup', {})
  if not status or status.get('state') != 'SUCCESS':
    return False, "not all checks have passed"

  # Check for merge conflicts
  merge_status = subprocess.run(['gh', 'pr', 'view', str(pr_number), '--json', 'mergeable,mergeStateStatus'],
                                capture_output=True, text=True)
  merge_data = json.loads(merge_status.stdout)
  if not merge_data.get('mergeable'):
    return False, "merge conflicts detected"

  # if (mergeStateStatus := merge_data.get('mergeStateStatus')) == "BEHIND":
  #   return False, f"branch is `{mergeStateStatus}`"

  return True, None


def process_pr(pr_data, source_branch, target_branch, squash_script_path):
  try:
    nodes = sort_prs_by_creation(pr_data)
    if not nodes:
      print("No PRs to squash")
      return 0

    print(f"Deleting target branch {target_branch}")
    subprocess.run(['git', 'branch', '-D', target_branch], check=False)
    subprocess.run(['git', 'branch', target_branch, f'origin/{source_branch}'], check=True)
    success_count = 0
    for pr in nodes:
      pr_comments = []
      try:
        pr_number = pr.get('number', 'UNKNOWN')
        branch = pr.get('headRefName', '')
        title = pr.get('title', '')
        head_repository = pr.get('headRepository', {})
        pr_labels = pr.get('labels', {}).get('nodes', [])
        is_fork = head_repository.get('isFork', False)
        trust_fork = any(label.get('name') == TRUST_FORK_LABEL for label in pr_labels)
        is_valid, skip_reason = validate_pr(pr)
        origin = "origin" if not head_repository.get('isFork', False) else head_repository.get('nameWithOwner', 'origin')

        if is_fork and trust_fork:
          print(f"Adding remote {origin} for PR #{pr_number}")
          subprocess.run(['git', 'remote', 'add', origin, head_repository.get('url')], check=False)

        if not is_valid:
          print(f"Warning: {skip_reason} for PR #{pr_number}, skipping")
          pr_comments.append(f"⚠️ This PR was skipped in the automated `{target_branch}` squash because **{skip_reason}**.")
          continue

        # Fetch PR branch
        subprocess.run(['git', 'fetch', origin, branch], check=True)
        # Delete branch if it exists (ignore errors if it doesn't)
        subprocess.run(['git', 'branch', '-D', branch], check=False)
        # Create new branch pointing to origin's branch
        subprocess.run(['git', 'branch', branch, f'{origin}/{branch}'], check=True)

        # Run squash script
        result = subprocess.run([
          squash_script_path,
          '--target', target_branch,
          '--base', source_branch,
          '--title', f"{title} (PR-{pr_number})",
          branch,
        ], capture_output=True, text=True)

        print(result.stdout)
        if result.returncode == 0:
          print(f"Successfully processed PR #{pr_number}")
          success_count += 1
          continue

        print(f"Error processing PR #{pr_number}:")
        print(f"Command failed with exit code {result.returncode}")
        output = result.stdout
        print(f"Error output: {output}")
        pr_comments.append(f"⚠️ Error during automated `{target_branch}` squash:\n```\n{output}\n```")
        subprocess.run(['git', 'reset', '--hard'], check=True)
        continue
      except Exception as e:
        print(f"Unexpected error processing PR #{pr_number}: {str(e)}")
        pr_comments.append(f"⚠️ Unexpected error during automated `{target_branch}` squash:\n```\n{str(e)}\n```")
        subprocess.run(['git', 'reset', '--hard'], check=True)
        continue
      finally:
        if pr_comments:
          add_pr_comments(pr_number, pr_comments)  # This "commits" all the comments generated on this run before leaving loop on continue.

    return success_count

  except Exception as e:
    import traceback
    print(f"Error in process_pr: {str(e)}")
    print("Full traceback:")
    print(traceback.format_exc())
    return 0


def main():
  parser = setup_argument_parser()
  try:
    args = parser.parse_args()
    validate_squash_script(args.squash_script_path)
    pr_data_json = json.loads(args.pr_data)

    # Process the PRs
    success_count = process_pr(pr_data_json, args.source_branch, args.target_branch, args.squash_script_path)
    print(f"Successfully processed {success_count} PRs")

  except Exception as e:
    print(f"Fatal error: {str(e)}", file=sys.stderr)
    return 1

  return 0


if __name__ == "__main__":
  sys.exit(main())
