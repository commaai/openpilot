error_handler() {
    echo failed at ${BASH_LINENO[0]}
}
trap error_handler ERR

create_rootfs_diff() {
    sudo rm -f "$CACHE_ROOTFS_TARBALL_PATH" # remove the old diff tarball from previous run, if exists
    cd /upper
    sudo tar -cf "$CACHE_ROOTFS_TARBALL_PATH" .
    cd
}

apply_rootfs_diff() {
    cd /
    sudo tar -xf "$CACHE_ROOTFS_TARBALL_PATH" 2>/dev/null || true
    cd
}

prepare_build() {
    # create and mount the required volumes where they're expected
    mkdir -p /tmp/openpilot
    sudo mount --bind "$REPO" /tmp/openpilot

    # needed for the unit tests not to fail
    sudo chmod 755 /sys/fs/pstore
}

post_commit_root() {
    # we have the diff tarball, now let's remove the folder too
    sudo rm -rf /upper

    # now we apply it straight away
    apply_rootfs_diff

    # before the next tasks are run, finalize the environment for them
    prepare_build
}

# warning: this function initiates a somewhat complicated program flow, follow carefully
# (even despite this part was made sure to not be too relevant for the rest of the job)
commit_root() {
    # if that's a first execution
    if ! [ -e /root_committed ]
    then
        # prepare directories
        sudo mkdir -p /base /newroot /upper /work

        # re-execute the main script (causing it to go straight to `build_inside_namespace`), but
        # inside the newly created namespace, in a way which would cause all mounts
        # created to automatically umount before it exits
        set +e
        sudo unshare -f --kill-child -m "$SELF_PATH" build_inside_namespace
        ec=$?

        # after it exited, remove the created directories (except the one containing created diff)
        sudo rm -rf /base /newroot /work

        # finally, create the rootfs diff tarball (to be pushed into the CI native cache)
        create_rootfs_diff

        # after creating the rootfs diff, bring the system into a state as if it was restored from cache
        post_commit_root

        exit $ec
    fi
}

reexecute() {
    set +e
    touch /root_committed
    sudo -u runner "$SELF_PATH"
    ec=$?
    exit $ec
}

# that's where the script goes after being re-executed for the first time
build_inside_namespace() {
    # initialize the mounts namespace on overlayfs to be able to prepare the rootfs diff
    mount --bind / /base
    mount -t overlay overlay -o lowerdir=/base,upperdir=/upper,workdir=/work /newroot

    # apply the current DNS config (beware: systemd often symlinks /etc/resolv.conf, that's why it's needed)
    rm -f /newroot/etc/resolv.conf
    touch /newroot/etc/resolv.conf
    cat /etc/resolv.conf > /newroot/etc/resolv.conf

    # switch the namespace's root mount to the newly created one
    mkdir -p /newroot/old
    cd /newroot
    pivot_root . old

    # initialize basic required POSIX-standard additional mounts
    mount -t proc proc /proc
    mount -t devtmpfs devtmpfs /dev
    mkdir -p /dev/pts
    mount -t devpts devpts /dev/pts
    mount -t proc proc /proc
    mount -t sysfs sysfs /sys

    # re-execute the main script for the 2nd time, causing it to go back to the main flow
    # (but this time already inside the newly created namespace)
    reexecute

    # after the main flow terminates and the namespace exist, post_commit_root is executed - be sure to look at it
}

if [ "$1" = "build_inside_namespace" ]
then
    build_inside_namespace
    exit
fi
