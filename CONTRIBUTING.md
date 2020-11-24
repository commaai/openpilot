# How to contribute

Our software is open source so you can solve your own problems without needing help from others. And if you solve a problem and are so kind, you can upstream it for the rest of the world to use.

Most open source development activity is coordinated through our [Discord](https://discord.comma.ai). A lot of documentation is available on our [medium](https://medium.com/@comma_ai/).

## Getting Started

 * Join our [Discord](https://discord.comma.ai)
 * Make sure you have a [GitHub account](https://github.com/signup/free)
 * Fork [our repositories](https://github.com/commaai) on GitHub

## Testing

### Local Testing

You can test your changes on your machine by running `run_docker_tests.sh`. This will run some automated tests in docker against your code.

### Automated Testing

All PRs and commits are automatically checked by Github Actions. Check out `.github/workflows/` for what Github Actions runs. Any new tests should be added to Github Actions.

### Code Style and Linting

Code is automatically checked for style by Github Actions as part of the automated tests. You can also run these tests yourself by running `pre-commit run --all`.

## Car Ports (openpilot)

We've released a [Model Port guide](https://medium.com/@comma_ai/openpilot-port-guide-for-toyota-models-e5467f4b5fe6) for porting to Toyota/Lexus models.

If you port openpilot to a substantially new car brand, see this more generic [Brand Port guide](https://medium.com/@comma_ai/how-to-write-a-car-port-for-openpilot-7ce0785eda84). You might also be eligible for a bounty.

## Pull Requests

Pull requests should be against the master branch. Before running master on in-car hardware, you'll need to clone the submodules too. That can be done by recursively cloning the repository:
```
git clone https://github.com/commaai/openpilot.git --recursive
```
Or alternatively, when on the master branch:
```
git submodule init
git submodule update
```
The reasons for having submodules on a dedicated repository and our new development philosophy can be found in our [post about externalization](https://medium.com/@comma_ai/a-2020-theme-externalization-13b33326d8b3).
Modules that are in seperate repositories include:
* apks
* cereal
* laika
* opendbc
* panda
* rednose
