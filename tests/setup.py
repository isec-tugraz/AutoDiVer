#!/usr/bin/env python3
from setuptools import Extension, setup, find_packages
from Cython.Build import cythonize


setup(
    ext_modules = cythonize([
        Extension("autodiver_ciphers.skinny.skinny", ["src/autodiver_ciphers/skinny/skinny.pyx"]),
        Extension("autodiver_ciphers.gift.gift64_cipher", ["src/autodiver_ciphers/gift/gift64_cipher.pyx"]),
        Extension("autodiver_ciphers.gift.gift128_cipher", ["src/autodiver_ciphers/gift/gift128_cipher.pyx"]),
        Extension("autodiver_ciphers.midori64.midori_cipher", ["src/autodiver_ciphers/midori64/midori_cipher.pyx"]),
        Extension("autodiver_ciphers.midori128.midori_cipher", ["src/autodiver_ciphers/midori128/midori_cipher.pyx"]),
        Extension("autodiver_ciphers.warp128.warp_cipher", ["src/autodiver_ciphers/warp128/warp_cipher.pyx"]),
        Extension("autodiver_ciphers.speedy192.speedy_cipher", ["src/autodiver_ciphers/speedy192/speedy_cipher.pyx"]),
        Extension("autodiver_ciphers.rectangle128.rectangle_cipher", ["src/autodiver_ciphers/rectangle128/rectangle_cipher.pyx"]),
    ]),
)
