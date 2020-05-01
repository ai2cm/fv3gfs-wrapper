Release Instructions
====================

Versions should take the form "v<major>.<minor>.patch". For example, "v0.3.0" is a valid
version, while "v1" is not and "0.3.0" is not.

1. Make sure all PRs are merged and tests pass.

2. Prepare a release branch with `git checkout -b release/<version>`.

3. Ensure that submodules (e.g. fv3config) point to a tagged commit on master. Tagged
   versions of fv3gfs-python should depend on tagged versions of dependencies.

4. Update the HISTORY.md, replacing the "latest" version heading with the new version.

5. Commit your changes so far to the release branch.

6. In the project root, run `bumpversion <major/minor/patch>`. This will create a new commit.

7. `git push -u origin release/<version>` and create a new pull request in Github.

8. When the pull request is merged to master, `git checkout master` and `git pull`,
   followed by `git tag <version>` and 

3. Run `git push origin --tags` to push all local tags to Github.

Circle CI will automatically push a tagged image to GCR when you pushing tags to master.
