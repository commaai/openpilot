#!/usr/bin/env python3
import subprocess
import sys
import argparse


def run_git_command(command, check=True):
  """
  Runs a git command and returns the trimmed stdout output.
  Exits the script if the command fails.
  """
  print(f"Running: {' '.join(command)}")
  result = subprocess.run(command, capture_output=True, text=True)
  if check and result.returncode != 0:
    print(result.stdout.strip())
    print(result.stderr.strip())
    sys.exit(result.returncode)
  return result.stdout.strip()


def main():
  parser = argparse.ArgumentParser(description="Merge multiple branches with squash merges.")
  parser.add_argument("--base", required=True, help="The base branch name from which the target branch will be created.")
  parser.add_argument("--target", required=True, help="The target branch name to merge into.")
  parser.add_argument("--title", required=False, help="Title for the commit")

  parser.add_argument("branches", nargs="+", help="List of branch names to merge into the target branch.")
  args = parser.parse_args()

  # Checkout the base branch to ensure a common starting point.
  run_git_command(["git", "checkout", args.base])

  # Check if the target branch exists. If not, create it from the base branch.
  branch_list = run_git_command(["git", "branch"], check=False)
  branch_names = [line.strip("* ").strip() for line in branch_list.splitlines()]
  if args.target in branch_names:
    run_git_command(["git", "checkout", args.target])
  else:
    run_git_command(["git", "checkout", "-b", args.target])

  # Iterate over each branch, merging it with a squash merge.
  for branch in args.branches:
    print(f"Merging branch '{branch}' with a squash merge.")
    # Merge the branch without creating a merge commit.
    run_git_command(["git", "merge", "--squash", branch])
    # Commit the squashed changes with an appropriate message.
    commit_message = args.title or f"Squashed merge of branch '{branch}'"
    run_git_command(["git", "commit", "-m", commit_message])

  print(f"All branches have been merged with squashed commits into '{args.target}'.")


if __name__ == "__main__":
  main()
