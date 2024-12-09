from setuptools import setup

setup(
    name="algorithms",
    version="0.1.0",
    packages=[
        "algorithms"
    ],
    package_dir={
        "algorithms":"src/algorithms"
    },
    include_package_data=False,
)