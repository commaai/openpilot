# openpilot releases

```
## release checklist

**Go to `devel-staging`**
- [ ] update RELEASES.md
- [ ] update `devel-staging`: `git reset --hard origin/master-ci`
- [ ] open a pull request from `devel-staging` to `devel`
- [ ] post on Discord

**Go to `devel`**
- [ ] bump version on master: `common/version.h` and `RELEASES.md`
- [ ] before merging the pull request
  - [ ] update from previous release -> new release
  - [ ] update from new release -> previous release
  - [ ] fresh install with `openpilot-test.comma.ai`
  - [ ] drive on fresh install
  - [ ] no submodules or LFS
  - [ ] check sentry, MTBF, etc.

**Go to `release3`**
- [ ] publish the blog post
- [ ] `git reset --hard origin/release3-staging`
- [ ] tag the release: `git tag v0.X.X <commit-hash> && git push origin v0.X.X`
- [ ] create GitHub release
- [ ] final test install on `openpilot.comma.ai`
- [ ] update factory provisioning
- [ ] close out milestone
- [ ] post on Discord, X, etc.
```
