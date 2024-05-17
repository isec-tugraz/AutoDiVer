#!/usr/bin/env python3
from setuptools import Extension, setup, find_packages
from Cython.Build import cythonize
setup(
    ext_modules = cythonize([
        Extension("differential_verification.skinny.skinny", ["src/differential_verification/skinny/skinny.pyx"]),
        Extension("differential_verification.gift64.gift_cipher", ["src/differential_verification/gift64/gift_cipher.pyx"]),
        Extension("differential_verification.gift128.gift_cipher", ["src/differential_verification/gift128/gift_cipher.pyx"]),
        Extension("differential_verification.midori64.midori_cipher", ["src/differential_verification/midori64/midori_cipher.pyx"]),
        Extension("differential_verification.midori128.midori_cipher", ["src/differential_verification/midori128/midori_cipher.pyx"]),
        Extension("differential_verification.warp128.warp_cipher", ["src/differential_verification/warp128/warp_cipher.pyx"]),
        Extension("differential_verification.speedy192.speedy_cipher", ["src/differential_verification/speedy192/speedy_cipher.pyx"]),
    ]),
)