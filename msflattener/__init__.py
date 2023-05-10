from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("msflattener")
except PackageNotFoundError:
    # package is not installed
    pass
