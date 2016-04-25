# How to Release

Here's a quick step-by-step for cutting a new release of mst_clustering.

## Pre-release

1. update version in ``mst_clustering/__init__.py`` to, e.g. "0.1"

2. create a release tag; e.g.
   ```
   $ git tag -a v0.1 -m 'version 0.1 release'
   ```

3. push the commits and tag to github

4. confirm that CI tests pass on github

5. under "tags" on github, update the release notes


## Publishing the Release

1. push the new release to PyPI (requires jakevdp's permissions)
   ```
   $ python setup.py sdist upload
   ```

## Post-release

1. update version in ``mst_clustering/__init__.py`` to next version; e.g. '0.2.dev0'

2. push changes to github
