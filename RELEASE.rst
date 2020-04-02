Release Instructions
====================

1. Prepare master branch for release (make sure all PRs are merged and tests pass).

2. Ensure that submodules (e.g. fv3config) point to a tagged commit on master. Tagged
   versions of fv3gfs-python should depend on tagged versions of dependencies.

2. In the project root, run `bumpversion <major/minor/patch>`. This will create a new tagged commit,
   having updated all version references to the new, higher version.

3. Run `git push && git push origin --tags` to push the version reference commit and
   then all local tags to master.

Circle CI will automatically push a tagged image to GCR when pushing tags to master.
