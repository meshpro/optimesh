try:
    # Python 3.8
    from importlib import metadata
except ImportError:
    try:
        import importlib_metadata as metadata
    except ImportError:
        pass

try:
    __version__ = metadata.version("optimesh")
except Exception:
    __version__ = "unknown"
