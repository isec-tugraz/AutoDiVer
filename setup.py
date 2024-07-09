#!/usr/bin/env python3
from setuptools import Extension, setup, find_packages
from Cython.Build import cythonize
setup(
    ext_modules = cythonize([
        Extension("autodiver.skinny.skinny", ["src/autodiver/skinny/skinny.pyx"]),
        Extension("autodiver.gift64.gift_cipher", ["src/autodiver/gift64/gift_cipher.pyx"]),
        Extension("autodiver.gift128.gift_cipher", ["src/autodiver/gift128/gift_cipher.pyx"]),
        Extension("autodiver.midori64.midori_cipher", ["src/autodiver/midori64/midori_cipher.pyx"]),
        Extension("autodiver.midori128.midori_cipher", ["src/autodiver/midori128/midori_cipher.pyx"]),
        Extension("autodiver.warp128.warp_cipher", ["src/autodiver/warp128/warp_cipher.pyx"]),
        Extension("autodiver.speedy192.speedy_cipher", ["src/autodiver/speedy192/speedy_cipher.pyx"]),
        Extension("autodiver.rectangle128.rectangle_cipher", ["src/autodiver/rectangle128/rectangle_cipher.pyx"]),
    ]),
)
