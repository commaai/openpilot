#!/usr/bin/env python3

import subprocess
import sys
import os
import argparse
import json
from datetime import datetime


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


def add_pr_comment(pr_number, comment):
    """Add a comment to a PR using gh cli"""
    try:
        subprocess.run(
            ['gh', 'pr', 'comment', str(pr_number), '--body', comment],
            check=True,
            capture_output=True,
            text=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Failed to add comment to PR #{pr_number}: {e.stderr}")


def validate_pr(pr):
    """Validate a PR and return (is_valid, skip_reason)"""
    pr_number = pr.get('number', 'UNKNOWN')
    branch = pr.get('headRefName', '')

    if not branch:
        return False, f"Missing branch name for PR #{pr_number}"

    # Check if checks have passed
    commits = pr.get('commits', {}).get('nodes', [])
    if not commits:
        return False, "No commit data found"

    status = commits[0].get('commit', {}).get('statusCheckRollup', {})
    if not status or status.get('state') != 'SUCCESS':
        return False, "Not all checks have passed"

    # Check for merge conflicts
    merge_status = subprocess.run(['gh', 'pr', 'view', str(pr_number), '--json', 'mergeable,mergeStateStatus'], capture_output=True, text=True)
    merge_data = json.loads(merge_status.stdout)
    if not merge_data.get('mergeable'):
        return False, "Merge conflicts detected"

    if (mergeStateStatus := merge_data.get('mergeStateStatus')) == "BEHIND":
        return False, f"Branch is `{mergeStateStatus}`"

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
            pr_number = pr.get('number', 'UNKNOWN')
            branch = pr.get('headRefName', '')
            title = pr.get('title', '')
            is_valid, skip_reason = validate_pr(pr)

            if not is_valid:
                print(f"Warning: {skip_reason} for PR #{pr_number}, skipping")
                add_pr_comment(pr_number, f"⚠️ This PR was skipped in the automated `{target_branch}` squash because {skip_reason}.")
                continue

            try:
                # Fetch PR branch
                subprocess.run(['git', 'fetch', 'origin', branch], check=True)
                # Delete branch if it exists (ignore errors if it doesn't)
                subprocess.run(['git', 'branch', '-D', branch], check=False)
                # Create new branch pointing to origin's branch
                subprocess.run(['git', 'branch', branch, f'origin/{branch}'], check=True)

                # Run squash script
                subprocess.run([
                    squash_script_path,
                    '--target', target_branch,
                    '--source', branch,
                    '--title', f"{title} (#{pr_number})",
                ], check=True)

                print(f"Successfully processed PR #{pr_number}")
                success_count += 1

            except subprocess.CalledProcessError as e:
                print(f"Error processing PR #{pr_number}:")
                print(f"Command failed with exit code {e.returncode}")
                error_output = getattr(e, 'stderr', 'No error output available')
                print(f"Error output: {error_output}")
                add_pr_comment(pr_number, f"⚠️ Error during automated {target_branch} squash:\n```\n{error_output}\n```")
                continue
            except Exception as e:
                print(f"Unexpected error processing PR #{pr_number}: {str(e)}")
                continue

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
