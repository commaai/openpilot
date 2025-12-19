# openpilot releases

```
## release checklist

### Go to staging
- [ ] make a GitHub issue to track release
- [ ] create release master branch
- [ ] update RELEASES.md
- [ ] bump version on master: `common/version.h` and `RELEASES.md`
- [ ] build new userdata partition from `release3-staging`
- [ ] post on Discord, tag `@release crew`

Updating staging:
1. either rebase on master or cherry-pick changes
2. run this to update: `BRANCH=devel-staging release/build_devel.sh`
3. build new userdata partition from `release3-staging`

### Go to release
- [ ] before going to release, test the following:
  - [ ] update from previous release -> new release
  - [ ] update from new release -> previous release
  - [ ] fresh install with `openpilot-test.comma.ai`
  - [ ] drive on fresh install
  - [ ] no submodules or LFS
  - [ ] check sentry, MTBF, etc.
  - [ ] stress test passes in production
- [ ] publish the blog post
- [ ] `git reset --hard origin/release3-staging`
- [ ] tag the release: `git tag v0.X.X <commit-hash> && git push origin v0.X.X`
- [ ] create GitHub release
- [ ] final test install on `openpilot.comma.ai`
- [ ] update factory provisioning
- [ ] close out milestone and issue
- [ ] post on Discord, X, etc.
```
