# CI Setup Optimization - Testing & Implementation Plan

## 🎯 Goal
Validate the CI setup optimization that reduces setup time from ~64s to <20s (main bounty) / <40s (sub-bounty).

## 📋 Testing Strategy

### Phase 1: Local Validation ✅
- [x] Native run script works correctly
- [x] Environment variables set properly (CI=1, PYTHONPATH, etc.)
- [x] Cache directories created
- [x] Basic command execution functional

### Phase 2: Fork & GitHub Testing

#### A. Create Your Fork
1. Go to https://github.com/commaai/openpilot
2. Click "Fork" to create your own copy
3. Clone your fork locally
4. Add the optimization branch

#### B. Set Up Testing Repository
```bash
# Add your fork as remote
git remote add fork https://github.com/YOUR_USERNAME/openpilot.git

# Push the optimization branch to your fork
git push fork ci-setup-optimization-test

# Create PR in your fork to trigger workflows
```

#### C. Enable GitHub Actions
- Go to your fork's "Actions" tab
- Click "I understand my workflows, go ahead and enable them"
- This allows the test workflows to run

### Phase 3: Performance Validation

#### Expected Results from test_native_setup.yaml:
- **Cold Cache (first run)**: 25-35s ✅ Sub-bounty (<40s)
- **Warm Cache (subsequent)**: 8-15s ✅ Main bounty (<20s)
- **Functionality Tests**: All packages and tools available

#### Validation Criteria:
- ✅ Setup completes in <40s (sub-bounty)
- ✅ Setup completes in <20s with cache (main bounty)
- ✅ All Python packages install correctly
- ✅ System tools (clang, git) available
- ✅ SCons build works

### Phase 4: Gradual Migration Strategy

#### 4.1 Start with Non-Critical Jobs
**Recommended first migrations:**
- `docs.yaml` - Documentation building
- `stale.yaml` - Issue management
- `badges.yaml` - Badge generation
- `repo-maintenance.yaml` - Repository maintenance

**Migration Process:**
1. Copy existing workflow file
2. Replace `setup-with-retry` → `setup-with-retry-native`
3. Replace Docker RUN commands with `NATIVE_RUN`
4. Test in fork first
5. Submit PR when validated

#### 4.2 Medium Priority Jobs
- `selfdrive_tests.yaml` → Use `selfdrive_tests_native.yaml`
- Unit test workflows
- Linting workflows

#### 4.3 Critical Jobs (Final Phase)
- Core integration tests
- Release workflows
- Model training/testing

### Phase 5: Performance Monitoring

#### Metrics to Track:
- **Setup Time**: Target <20s (warm) / <40s (cold)
- **Cache Hit Rate**: Should be >80% for most workflows
- **Failure Rate**: Should be ≤ current Docker setup
- **Resource Usage**: Memory/CPU usage comparison

#### Monitoring Tools:
- GitHub Actions timing logs
- Cache hit/miss rates
- Job success/failure rates
- Performance dashboard (can be added)

## 🔧 Local Testing Commands

### Test Basic Functionality
```bash
# Test environment setup
./tools/native_run.sh "echo 'CI='\$CI', PYTHONPATH='\$PYTHONPATH"

# Test Python packages (if available)
./tools/native_run.sh "python3 -c 'import sys; print(sys.path)'"

# Test cache directory creation
./tools/native_run.sh "ls -la .ci_cache/"
```

### Test SCons Integration
```bash
# Test SCons dry run (if SCons is installed)
./tools/native_run.sh "which scons && scons --help" || echo "SCons not installed locally"
```

## 📊 Success Criteria

### Technical Requirements
- [x] Native setup action created
- [x] Retry wrapper implemented
- [x] Performance test workflow ready
- [x] Native run helper functional
- [x] Comprehensive caching strategy
- [x] Documentation complete

### Performance Requirements
- [ ] Setup time <40s (sub-bounty) - **Needs GitHub testing**
- [ ] Setup time <20s with cache (main bounty) - **Needs GitHub testing**
- [ ] Functionality parity with Docker setup
- [ ] Cache efficiency >80% hit rate

### Integration Requirements
- [ ] All existing workflows can be migrated
- [ ] No breaking changes to build process
- [ ] Backward compatibility maintained

## 🚧 Known Limitations & Mitigations

### Local Testing Limitations
- **Can't test APT package installation** (requires Ubuntu 24.04)
- **Can't test GitHub Actions cache** (requires GitHub environment)
- **Can't test parallel performance** (requires multiple runners)

### Mitigation Strategy
- **Fork testing**: Full GitHub Actions environment
- **Gradual rollout**: Start with low-risk workflows
- **Monitoring**: Track performance metrics closely
- **Fallback**: Keep Docker setup as backup initially

## 🎯 Next Immediate Steps

1. **Create Fork** - Fork the openpilot repository
2. **Push Branch** - Upload optimization to your fork
3. **Enable Actions** - Allow workflows to run in fork
4. **Run Tests** - Execute `test_native_setup.yaml`
5. **Analyze Results** - Validate performance targets
6. **Create PR** - Submit to main repository when validated

## 📈 Expected Timeline

- **Week 1**: Fork testing & validation
- **Week 2**: Gradual migration of non-critical jobs
- **Week 3**: Migration of medium-priority jobs
- **Week 4**: Full rollout & monitoring

This phased approach ensures we meet the bounty requirements while minimizing risk to the CI system.