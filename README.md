# Tool for Verifying Differential Characteristics
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
verify-characteristic midori64 trails/midori64/midori64_zhww_r5_1.npz count-keys-lin
# use ApproxMC to count the number of keys
verify-characteristic midori64 trails/midori64/midori64_zhww_r5_1.npz count-keys
# count the number of keys experimentally
verify-characteristic midori64 trails/midori64/midori64_zhww_r5_1.npz count-keys-sat
```
## Manual Installation
```bash
# install espresso
# create and activeate a virtual environment
python3 -m venv venv/
source venv/bin/activate
# install espresso-logic
git clone https://github.com/classabbyamp/espresso-logic.git
cd espresso-logic/espresso-src
make
cd ../..
cp espresso-logic/bin/espresso venv/bin/
# install the tool in editable mode
pip install -e .
# run the tool
verify-characteristic gift64 trails/gift64/gift64_lwzz19_r9_table_2.txt count-keys-sat embed
```