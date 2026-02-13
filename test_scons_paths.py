from SCons.Script import *
import os

env = Environment()

def build_action(target, source, env):
    print(f"Action CWD: {os.getcwd()}")
    return None

# Test nodes
root_dir = Dir('#')
sub_dir = Dir('selfdrive/ui')
sub_file = File('selfdrive/ui/SConscript')

print(f"Root Dir Path: [{root_dir.path}]")
print(f"Sub Dir Path: [{sub_dir.path}]")
print(f"Sub File Path: [{sub_file.path}]")

# Test command expansion
env.Command('test_target', sub_file, f"echo SOURCE: $SOURCE, TARGET: $TARGET, ROOT: {root_dir.path}")

# Run a dummy build to see the expansion
# (Usually we'd run 'scons -n' on this file)
