# AutoDiVer: Automatic Differential Verification

A tool for verifying differential characteristics.


## Installation with Docker

The easiest way to get started is to use the provided `./run_docker.sh` script.
It builds a docker image which includes all dependencies like espresso, ApproxMC, and CryptoMiniSAT.

Inside of the Docker, the current working directory is available in `/mnt`.
The tool is already installed and can be run by using `autodiver -h`.


# Usage

You can start our tool as follows:
```
autodiver <cipher_name> <characteristic_file> <command>
```

The characteristic file usually ends in `.npz` (except for GIFT).
Samples are located in the `trails/` directory.

The following commands are supported:

| Command                   | Description                                                                            |
| ------------------------- | -------------------------------------------------------------------------------------- |
| `count-prob`              | Estimate the probability using ApproxMC                                                |
| `count-tweakeys`          | Count valid tweakeys using ApproxMC                                                    |
| `count-tweakeys-lin`      | Find the affine hull of the set of valid tweakeys                                      |
| `count-tweakeys-combined` | Find the affine hull and verify the remaining keyspace experimentally with SAT solvers |
| `count-tweakeys-sat`      | Estimate size of valid tweakey space experimentally with SAT solvers                   |
| `find-conflicts`          | List S-boxes which lead to a contradiction                                             |
| `solve`                   | Find a satisfying pair for the characteristic                                          |
| `embed`                   | Launch an interactive IPython shell                                                    |
| `write-cnf`               | Write the CNF to a file                                                                |

For example, to verify analyze the MIDORI-64 characteristic the following is possible:

```bash
# calculate the probability averaged over all 2^128 keys
autodiver midori64 trails/midori64/midori64_zhww_r5_1.npz count-prob

# find affine conditions on the key
autodiver midori64 trails/midori64/midori64_zhww_r5_1.npz count-tweakeys-lin

# use ApproxMC to count the number of keys
autodiver midori64 trails/midori64/midori64_zhww_r5_1.npz count-tweakeys

# count the number of keys experimentally
autodiver midori64 trails/midori64/midori64_zhww_r5_1.npz count-tweakeys-sat
```

# Licensing

The source code of AutoDiVer in `src/` is licensed under the MIT license.

The test cases in `tests/` are licensed under GPL to provide compatibility with some of the cipher implementations.
This includes the cipher implementations which are originally distributed under various licenses.
See [tests/src/autodiver_ciphers/README.md](tests/src/autodiver_ciphers/README.md) for details.


# Running the tests

To run the tests, you first need to install the GPL licensed cipher implementations:

```bash
pip install ./tests
```

Then, you can run all tests by running

```bash
pytest
```


# Manual Installation

For manual installation, install [espresso](https://github.com/classabbyamp/espresso-logic), [CryptoMiniSAT v5.11.21](), [ApproxMC v4.1.24](https://github.com/meelgroup/approxmc), and [Arjun v2.5.4](https://github.com/meelgroup/arjun).
Please follow the guides in the README documents to do so.
Make sure `cryptominisat5`, `approxmc`, and `espresso` are available inside your `$PATH` environment variable.

Then, create a new virtual environment, activate it and install the current directory:
```bash
python3 -m venv venv/
source venv/bin/activate
pip install --editable .
```
