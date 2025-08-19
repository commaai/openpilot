# openpilot releases

```
## release checklist

**Go to `devel-staging`**
- [ ] make issue to track release
- [ ] update RELEASES.md
- [ ] trigger new nightly build: https://github.com/commaai/openpilot/actions/workflows/release.yaml
- [ ] update `devel-staging`: `git reset --hard origin/__nightly`
- [ ] build new userdata partition from `release3-staging`
- [ ] open a pull request from `devel-staging` to `devel`
- [ ] post on Discord, tag `@release crew`

**Go to `devel`**
- [ ] bump version on master: `common/version.h` and `RELEASES.md`
- [ ] before merging the pull request, test the following:
  - [ ] update from previous release -> new release
  - [ ] update from new release -> previous release
  - [ ] fresh install with `openpilot-test.comma.ai`
  - [ ] drive on fresh install
  - [ ] no submodules or LFS
  - [ ] check sentry, MTBF, etc.
  - [ ] stress test in production

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
