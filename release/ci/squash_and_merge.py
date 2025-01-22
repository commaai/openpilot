#!/usr/bin/env python3
import argparse
import subprocess
import sys
import shutil
import signal
import contextlib
import tempfile
import os


def run_command(command: str) -> tuple[int, str, str]:
  """Run a shell command and return exit code, stdout, and stderr."""
  process = subprocess.Popen(
    command,
    shell=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
  )
  stdout, stderr = process.communicate()
  return process.returncode, stdout.strip(), stderr.strip()


def is_gh_available() -> bool:
  """Check if GitHub CLI is available."""
  return shutil.which('gh') is not None


def get_current_branch() -> str | None:
  """Get the name of the current git branch."""
  code, output, error = run_command("git rev-parse --abbrev-ref HEAD")
  if code != 0:
    print(f"Error getting current branch: {error}")
    return None
  return output


def backup_branch(branch_name: str) -> bool:
  """Create a backup of the current branch."""
  backup_name = f"{branch_name}-backup-$(date +%Y%m%d_%H%M%S)"
  code, _, error = run_command(f"git branch {backup_name}")
  if code != 0:
    print(f"Error creating backup branch: {error}")
    return False
  print(f"Created backup branch: {backup_name}")
  return True


def get_commit_messages(source_branch: str, target_branch: str) -> list[str] | None:
  """Get all commit messages between source and target branches."""
  code, output, error = run_command(f"git log {target_branch}..{source_branch} --format=%B")
  if code != 0:
    print(f"Error getting commit messages: {error}")
    return None
  return [msg.strip() for msg in output.splitlines() if msg and not msg.startswith('Merge')]


def get_pr_info(branch_name: str) -> str | None:
  """Get PR title using GitHub CLI."""
  if not is_gh_available():
    print("Warning: GitHub CLI not found. Install it to auto-fetch PR titles:")
    print("  https://cli.github.com/")
    return None

  # Try to get PR info using gh cli
  code, output, error = run_command(f"gh pr view --json title --jq .title {branch_name}")
  if code != 0:
    print(f"No open PR found for branch '{branch_name}'")
    return None

  return output


def create_squash_message(pr_title: str | None, commit_messages: list[str], source_branch: str) -> str:
  """Create a squash commit message from PR title and commit messages."""
  parts = []

  # Add PR title if provided
  if pr_title:
    parts.append(pr_title)
  else:
    parts.append(f"Squashed changes from {source_branch}")
  parts.append("")  # Empty line after title

  # Add original commits section
  if commit_messages:
    parts.append("Original commits:")
    parts.append("")  # Empty line before list
    parts.extend(f"* {msg}" for msg in commit_messages)

  return '\n'.join(parts)


def prompt_for_title() -> str:
  """Prompt user for a commit title."""
  return input("Enter commit title (or press Enter to use default): ").strip()


@contextlib.contextmanager
def workspace_manager(original_branch: str):
  """Context manager to handle workspace state and cleanup."""
  stash_created = False
  stash_restored = False
  temp_branch: str | None = None

  def cleanup_handler(signum=None, frame=None):
    """Clean up workspace state."""
    nonlocal temp_branch, stash_created, stash_restored
    try:
      if signum and stash_restored:
        # If we're handling Ctrl+C but stash was already restored,
        # just clean up branches and exit
        current = get_current_branch()
        if current and current != original_branch:
          run_command(f"git checkout {original_branch}")
        if temp_branch:
          run_command(f"git branch -D {temp_branch}")
        print("\nOperation interrupted, but changes were already restored.")
        sys.exit(3)

      # First, switch back to original branch
      current = get_current_branch()
      if current and current != original_branch:
        run_command(f"git checkout {original_branch}")

      # Then clean up temp branch
      if temp_branch:
        run_command(f"git branch -D {temp_branch}")

      # Finally, restore stash if needed - AFTER switching branches
      if stash_created and not stash_restored:
        print("Restoring your uncommitted changes...")
        code, stash_list, _ = run_command("git stash list")
        if code == 0 and "Automatic stash by squash script" in stash_list:
          run_command("git stash pop")
          stash_restored = True
          stash_created = False

      if signum:
        print("\nOperation interrupted. Cleaned up and restored original state.")
        sys.exit(4)

    except Exception as e:
      print(f"Error during cleanup: {e}")
      if signum:
        sys.exit(5)

  try:
    # Set up signal handlers
    signal.signal(signal.SIGINT, cleanup_handler)
    signal.signal(signal.SIGTERM, cleanup_handler)

    # Check for changes (including untracked files)
    code, output, _ = run_command("git status --porcelain")
    if output:
      print("Stashing uncommitted changes...")
      run_command("git stash push -u -m 'Automatic stash by squash script'")
      stash_created = True

    yield lambda x: setattr(x, 'temp_branch', temp_branch)

  except Exception as e:
    print(f"\nError occurred: {str(e)}")
    cleanup_handler()
    raise
  finally:
    cleanup_handler()


def create_commit_with_message(message: str) -> bool:
  """Create a commit with the given message using a temporary file."""
  try:
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
      f.write(message)
      temp_path = f.name

    # Use the temporary file for the commit message
    code, _, error = run_command(f"git commit -F {temp_path}")
    os.unlink(temp_path)  # Clean up the temp file

    if code != 0:
      print(f"Error creating commit: {error}")
      return False
    return True
  except Exception as e:
    print(f"Error handling commit message: {e}")
    if os.path.exists(temp_path):
      os.unlink(temp_path)
    return False


def squash_and_merge(source_branch: str, target_branch: str, manual_title: str | None, backup: bool = False, push: bool = False) -> bool:
  """
  Squash the source branch and merge into target branch.
  """
  # Get original branch right away
  original_branch = get_current_branch()
  if not original_branch:
    return False

  class State:
    temp_branch: str | None = None

  state = State()

  with workspace_manager(original_branch) as set_temp_branch:
    # Validate source branch exists
    code, _, error = run_command(f"git rev-parse --verify {source_branch}")
    if code != 0:
      print(f"Error: Source branch {source_branch} not found")
      return False

    if source_branch == target_branch:
      print(f"Error: Source and target branches cannot be the same ({source_branch})")
      return False

    # Ensure target branch exists
    code, _, error = run_command(f"git rev-parse --verify {target_branch}")
    if code != 0:
      print(f"Error: Target branch {target_branch} not found")
      return False

    # Find merge base
    code, merge_base, error = run_command(f"git merge-base {target_branch} {source_branch}")
    if code != 0:
      print(f"Error finding merge base: {error}")
      return False

    # Create backup unless explicitly skipped
    if backup and not backup_branch(source_branch):
      return False

    # Get commit messages
    commit_messages = get_commit_messages(source_branch, target_branch)
    if commit_messages is None:
      return False

    # Get title (priority: manual title > PR title > prompt user)
    title = manual_title
    if not title:
      title = get_pr_info(source_branch)
    if not title:
      title = prompt_for_title()

    try:
      # Create and switch to temporary branch
      temp_branch = f"temp-squash-{source_branch}"
      state.temp_branch = temp_branch
      set_temp_branch(state)

      print(f"\nCreating temporary branch {temp_branch}...")
      code, _, error = run_command(f"git checkout -b {temp_branch} {source_branch}")
      if code != 0:
        print(f"Error creating temp branch: {error}")
        return False

      print("Preparing squash by resetting temporary branch to merge base...")
      code, _, error = run_command(f"git reset --soft {merge_base}")
      if code != 0:
        print(f"Error resetting for squash: {error}")
        return False

      # Create commit with message
      print("Creating squash commit...")
      squash_message = create_squash_message(title, commit_messages, source_branch)
      if not create_commit_with_message(squash_message):
        return False

      # Switch to target and try merge
      print(f"\nSwitching to target branch {target_branch}...")
      code, _, error = run_command(f"git checkout {target_branch}")
      if code != 0:
        print(f"Error checking out target branch: {error}")
        return False

      print(f"Attempting to merge changes from {temp_branch}...")
      code, _, error = run_command(f"git rebase {temp_branch}")

      if code != 0:
        print(f"\nMerge failed with error: {error}")
        print("\nThe squash was successful, and your changes are preserved in the temporary branch.")
        print("To complete the merge manually, follow these steps:")
        print(f"\n1. Your squashed changes are in branch: '{temp_branch}'")
        print(f"2. The target branch is: '{target_branch}'")
        print("\nTo resolve the conflicts:")
        print(f"   git checkout {target_branch}")
        print(f"   git merge {temp_branch}")
        print("   # resolve conflicts in your editor")
        print("   git add <resolved-files>")
        print("   git commit")
        print(f"   git push origin {target_branch}  # when ready to push")
        print("\nTo clean up after successful merge:")
        print(f"   git branch -D {temp_branch}")

        # Make sure to abort the merge
        print("\nAborting current merge attempt...")
        run_command("git merge --abort")

        # Return to original branch, but keep temp branch
        print(f"Returning to {original_branch}...")
        run_command(f"git checkout {original_branch}")
        return False

      # Clean up temp branch on success
      run_command(f"git branch -D {temp_branch}")

      # Push if requested
      if push:
        code, _, error = run_command(f"git push origin {target_branch}")
        if code != 0:
          print(f"Error pushing to {target_branch}: {error}")
          return False
        print(f"Successfully pushed to {target_branch}")
      else:
        print(f"Changes squashed and merged into {target_branch} locally")
        print(f"To push the changes: git push origin {target_branch}")

      # Return to original branch
      code, _, error = run_command(f"git checkout {original_branch}")
      if code != 0:
        print(f"Warning: Failed to return to original branch: {error}")
        return False

      return True

    except Exception as e:
      print(f"Error during squash process: {e}")
      return False


def main():
  parser = argparse.ArgumentParser(
    description='Squash branch and merge into target branch'
  )
  parser.add_argument('--target', '-t', required=True,
                      help='Target branch to merge changes into')
  parser.add_argument('--source', '-s',
                      help='Source branch to squash (default: current branch)')
  parser.add_argument('--title', '-m',
                      help='Optional manual title (overrides PR title)')
  parser.add_argument('--backup', action='store_true',
                      help='Creates a backup branch for the source branch')
  parser.add_argument('--push', action='store_true',
                      help='Push changes to remote after squashing')

  args, unknown = parser.parse_known_args()

  # Determine source branch early
  source_branch = args.source
  if not source_branch:
    source_branch = get_current_branch()
    if not source_branch:
      sys.exit(1)

  if not squash_and_merge(source_branch, args.target, args.title, args.backup, args.push):
    sys.exit(2)


if __name__ == "__main__":
  main()
