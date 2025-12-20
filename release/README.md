# openpilot releases

```
## release checklist

### Go to staging
- [ ] make a GitHub issue to track release with this checklist
- [ ] create release master branch
  - [ ] create a branch from upstream master named `zerotentwo` for release `v0.10.2`
  - [ ] revert risky commits (double check with autonomy team)
  - [ ] push the new branch
- [ ] push to staging:
  - [ ] make sure you are on the newly created release master branch (`zerotentwo`)
  - [ ] run `BRANCH=devel-staging release/build_stripped.sh`. Jenkins will then automatically build staging on device, run `test_onroad` and update the staging branch
- [ ] bump version on master: `common/version.h` and `RELEASES.md`
- [ ] post on Discord, tag `@release crew`

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
- [ ] `git reset --hard origin/release-mici-staging`
- [ ] tag the release: `git tag v0.X.X <commit-hash> && git push origin v0.X.X`
- [ ] create GitHub release
- [ ] final test install on `openpilot.comma.ai`
- [ ] update factory provisioning
- [ ] close out milestone and issue
- [ ] post on Discord, X, etc.
```
