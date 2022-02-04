Release Instructions
====================

Versions should take the form "v<major>.<minor>.patch". For example, "v0.3.0" is a valid
version, while "v1" is not and "0.3.0" is not.

1. Make sure all PRs are merged and tests pass.

2. Prepare a release branch with `git checkout -b release/<version>`.

3. Run `git submodule status` and ensure fv3config shows a tagged version at the end its line, such as in f4862a7bbcccd0602d54c77aede935e4cba36a72 fv3config (v0.4.0). If it does not, you must first create a version-tagged commit by releasing fv3config, and then update the submodule reference. Then re-do this check.

4. Update the HISTORY.md, replacing the "latest" version heading with the new version.

5. Commit your changes so far to the release branch.

6. In the project root, run `bumpversion <major/minor/patch>`. This will create a new commit.

7. Repeat steps 4-6 inside `external/pace-util` to create a new version of pace-util.

8. `git push -u origin release/<version>` and create a new pull request in Github.

9. When the pull request is merged to master, `git checkout master` and `git pull`, followed by `git tag <version>`. This creates a local tag. It is crucial that you pull the latest master after the pull request is merged and before you create the tag.

10. Run `git push origin --tags` to push all local tags to Github.

Circle CI will automatically push a tagged image to GCR when you pushing tags to master.
