# Model and asset storage

The ONNX models in `openpilot/selfdrive/modeld/models/` and the other files
selected by `.gitattributes` use Git LFS pointer files. Their bytes are stored
in the public Hugging Face model repository
[`commaai/openpilot-lfs`](https://huggingface.co/commaai/openpilot-lfs),
under `sha256/<LFS object ID>`.

`tools/op.sh setup` installs a Git LFS standalone transfer agent using
[xet-core](https://github.com/huggingface/xet-core)'s `hf-xet` bindings through
`huggingface_hub`. Its dependencies are pinned separately in
`tools/xet.py.lock` and installed by uv. Both uploads and downloads use Xet;
upstream `git-xet` currently implements Xet uploads only.

The normal `git add`, `git commit`, `git push`, and `git lfs pull` commands
continue to work. Git LFS still manages pointers, hooks, exclusions, and its
local object cache. The adapter verifies each object's size and SHA-256.
Xet handles chunk deduplication and parallel transfers within each file.
Assets in prebuilt release branches are still bundled in Git.

## Setup and publishing

After installing uv and Git LFS, the adapter can also be installed alone:

```bash
python3 tools/xet.py install
git lfs pull
```

Downloads from the public repository need no login. Uploads require a Hugging
Face token with write access to the storage repository. Set `HF_TOKEN`, or log
in with `hf auth login`. Unset `HF_HUB_DISABLE_XET` if it was configured for an
HTTP cache; the adapter fails instead of silently disabling Xet.

A fork can choose its own Hugging Face model repository:

```bash
git config --local xet.repo YOUR_ACCOUNT/openpilot-lfs
```

Populate it with the inherited objects using the procedure below before
switching downloads to it. Give its `sha256/**` files the same LFS attributes.
The override applies to both uploads and downloads, regardless of Git remote.

## Storage migration before merge

The endpoint change must stay in draft until the destination is populated and
a clean checkout can download every required object. Previously these objects
were hosted at `https://gitlab.com/commaai/openpilot-lfs.git/info/lfs`.
No model pointers or Git history need to be rewritten.

1. Create the public Hugging Face model repository `commaai/openpilot-lfs`.
   Commit a `.gitattributes` file containing
   `sha256/** filter=lfs diff=lfs merge=lfs -text` to it **before uploading**,
   so even small assets use Xet storage.
2. In a full openpilot clone with every supported branch/tag fetched, download
   the historical objects from GitLab. Explicitly bypass the new adapter and
   endpoint for this command:

   ```bash
   git -c lfs.standalonetransferagent= \
       -c lfs.url=https://gitlab.com/commaai/openpilot-lfs.git/info/lfs \
       lfs fetch --all origin
   ```

3. Install the adapter, authenticate to Hugging Face, and copy all local refs'
   objects to the new repository. Existing matching objects are skipped, so
   interrupted migrations can be resumed:

   ```bash
   python3 tools/xet.py install
   git lfs push --all origin
   ```

4. From a separate clone with an empty LFS and Hugging Face cache, run setup and
   `git lfs pull --exclude=''`. Verify `git lfs fsck`, including the big driving
   model, and test a new model upload. Validate Linux x86_64, macOS, and device
   ARM64 setup before merging.
5. Retain the GitLab store for old commits that still reference it.

The migration commands publish to the configured storage repository. Opening
this PR alone does not create that repository or copy any assets.
