# AutoDiVer: Automatic Differential Verification

A tool for verifying differential characteristics.


## Installation with Docker

The easiest way to get started is to use the provided `./run_docker.sh` script.
It builds a docker image which includes all dependencies like espresso, ApproxMC, and CryptoMiniSAT.

Inside of the Docker, the current working directory is available in `/mnt`.
The tool is already installed and can be run by using `verify-characteristic -h`.


## Usage

You can start our tool as follows:
```
verify-characteristic <cipher_name> <characteristic_file> <command>
```

For example, to verify analyze the MIDORI-64 characteristic the following is possible:

```bash
# calculate the probability averaged over all 2^128 keys
verify-characteristic midori64 trails/midori64/midori64_zhww_r5_1.npz count-prob

# find affine conditions on the key
verify-characteristic midori64 trails/midori64/midori64_zhww_r5_1.npz count-tweakeys-lin

# use ApproxMC to count the number of keys
verify-characteristic midori64 trails/midori64/midori64_zhww_r5_1.npz count-tweakeys

# count the number of keys experimentally
verify-characteristic midori64 trails/midori64/midori64_zhww_r5_1.npz count-tweakeys-sat
```


## Running the tests

Running the tests can be done by starting `pytest` in his directory.


## Manual Installation

For manual installation, install [espresso](https://github.com/classabbyamp/espresso-logic), [CryptoMiniSAT v5.11.21](), [ApproxMC v4.1.24](https://github.com/meelgroup/approxmc), and [Arjun v2.5.4](https://github.com/meelgroup/arjun).
Please follow the guides in the README documents to do so.
Make sure `cryptominisat5`, `approxmc`, and `espresso` are available inside your `$PATH` environment variable.

Then, create a new virtual environment, activate it and install the current directory:
```bash
python3 -m venv venv/
source venv/bin/activate
pip install --editable .
```
