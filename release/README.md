# openpilot releases

## release checklist

**Go to `devel-staging`**
- [ ] update `devel-staging`: `git reset --hard origin/master-ci`
- [ ] open a pull request from `devel-staging` to `devel`

**Go to `devel`**
- [ ] update RELEASES.md
- [ ] close out milestone
- [ ] post on Discord dev channel
- [ ] bump version on master: `common/version.h` and `RELEASES.md`
- [ ] merge the pull request

tests:
- [ ] update from previous release -> new release
- [ ] update from new release -> previous release
- [ ] fresh install with `openpilot-test.comma.ai`
- [ ] drive on fresh install
- [ ] comma body test
- [ ] no submodules or LFS
- [ ] check sentry, MTBF, etc.

**Go to `release3`**
- [ ] publish the blog post
- [ ] `git reset --hard origin/release3-staging`
- [ ] tag the release
```
git tag v0.X.X <commit-hash>
git push origin v0.X.X
```
- [ ] create GitHub release
- [ ] final test install on `openpilot.comma.ai`
- [ ] update production
- [ ] Post on Discord, X, etc.
