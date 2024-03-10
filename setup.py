#!/usr/bin/env python3
from setuptools import Extension, setup
from Cython.Build import cythonize
setup(
    ext_modules = cythonize([
        Extension("skinny.skinny", ["skinny/skinny.pyx"]),
        Extension("skinny._util", ["skinny/_util.pyx"]),
        Extension("gift64.gift_cipher", ["gift64/gift_cipher.pyx"]),
        Extension("midori64.midori_cipher", ["midori64/midori_cipher.pyx"]),
        Extension("midori128.midori_cipher", ["midori128/midori_cipher.pyx"]),
        # Extension("skinny/lin_util", ["skinny/lin_util.pyx"]),
    ]),
)