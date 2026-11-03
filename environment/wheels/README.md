# Gensim 4.4.0 for Python 3.14 on Windows

`gensim-4.4.0-cp314-cp314-win_amd64.whl` is the locally built wheel already used by the course's Python 3.14 environment. It is **not** an official upstream Windows Python 3.14 wheel. The PyPI 4.4.0 release was checked during packaging and did not provide a `cp314-win_amd64` wheel.

SHA-256:

```text
afb49c171b31d2dcff43536ff156b9cb9b309bd0dc67b089707d05315cac917b
```

The build used the Gensim 4.4.0 source, Visual Studio 2022 C++ Build Tools/Windows SDK, and Cython 3.3.0 to regenerate C extensions for Python 3.14. The missing `gensim/similarities/fastss.pyx` was restored from the upstream 4.4.0 source. The saved build log reports the same wheel hash. The archive `gensim-4.4.0-cp314-source.tar.gz` contains the retained build source, including generated C files and `COPYING`, excluding build outputs and caches. The wheel also includes its license metadata.

The source archive is supplied for inspection and rebuilding; environment setup installs only the wheel. The build log, upstream source URL, and source archive hash are recorded in `provenance.json`. Gensim is distributed under its upstream LGPL license; see `COPYING` inside the source archive.

Upstream: [Gensim 4.4.0](https://pypi.org/project/gensim/4.4.0/), [source tag](https://github.com/piskvorky/gensim/tree/4.4.0).
