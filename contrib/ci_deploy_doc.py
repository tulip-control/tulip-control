"""Build documentation, push results to remote `git` repository.

The generated documentation is pushed only if the HTML or
other files change.

The push replaces the previous commit on branch `docs_pages`,
so that binaries do not accumulate on GitHub.

This script requires:
    `gitpython`
"""
import argparse
import os
from pathlib import Path
import shutil
import subprocess

import git


GIT_ORIGIN = 'origin'
DOCS_COMMIT_MESSAGE = '''\
BIN: built documentation files

committed by GitHub Actions.
'''


def _main():
    """Entry point."""
    args = _parse_args()
    docs_src = args.docs_src_path
    docs_out = args.docs_out_path
    docs_cp = args.docs_cp_path
    bin_branch = args.binaries_branch
    docs_branch = args.docs_branch
    repo = git.Repo('.')
    # checkout autogenerated image files
    dir_stack = list()
    _pushd(docs_src, dir_stack)
    repo.git.fetch(GIT_ORIGIN)
    # TODO: compute which PNG files need
    # to be checked-out
    #
    # TODO: check that, for each autogenerated PNG,
    # the latest commit that changed this PNG is
    # newer than the latest commit that changed the
    # corresponding SVG file on the source branch.
    #
    # The correspondence between PNG and SVG files
    # is inferred by the stem of the basename
    # (i.e., with the filename extension omitted).
    #
    # If this condition does not hold,
    # then raise a `RuntimeError`.
    #
    # Otherwise, print a message that informs
    # that the check passed, but that the test
    # does not ensure that the autogenerated PNG
    # files were created using the latest
    # corresponding SVG files.
    repo.git(
        'restore',
        f'--source={GIT_ORIGIN}/{bin_branch}',
        '*.png')
    # build the documentation
    cmd = ['make', 'html']
    retcode = subprocess.call(cmd)
    if retcode != 0:
        raise RuntimeError(
            f'running {cmd} returned '
            f'with: {retcode}')
    _popd(dir_stack)
    # push the built docs to `git` branch
    _config_git(repo)
    _commit_built_docs(
        docs_out, docs_cp, docs_branch, repo)
    remote_branch_exists = _fetch_branch(
        docs_out, repo)
    if remote_branch_exists:
        changed = _diff_built_docs_between_branches(
            docs_branch, docs_cp, repo)
    else:
        changed = True
    _push_built_docs(changed, repo)


def _config_git(repo):
    """Configure `git` operations."""
    repo.git.config('--global',
        'user.name', 'GitHub Actions')
    repo.git.config('--global',
        'user.email', "'<>'")


def _commit_built_docs(
        docs_out, docs_cp, docs_branch, repo):
    """Commit built documentation to `docs_branch`.

    Copies the directory tree that is rooted at
    `docs_out` to `docs_cp`, then commits the
    copied directory tree to branch `docs_branch`.
    """
    # create branch `docs_out` for committing
    # the newly built docs
    repo.git.checkout('-b', docs_branch)
    status = repo.git('status')
    print(f'`git status` returned:\n{status}')
    # copy the newly built docs to
    # the deployment directory
    os.mkdir(docs_cp)
    shutil.copytree(docs_out, docs_cp)
    Path(f'{docs_cp}/.nojekyll').touch()
    # commit the built docs
    repo.git.add('-f', docs_cp)
    # TODO: consider creating a root commit
    repo.git.commit('-m', DOCS_COMMIT_MESSAGE)


def _fetch_branch(docs_branch, repo):
    """Fetch `docs_branch` if it exists at remote.

    Return `True` if `docs_branch` exists at remote,
    and is successfully fetched, `False` otherwise.
    """
    repo.git.fetch(GIT_ORIGIN, docs_branch)
    remote_branch_exists = True
    remote_branch_name = (
        f'remotes/{GIT_ORIGIN}/{docs_branch}')
    try:
        repo.git.show_branch(
            remote_brach_name)
    except git.GitCommandError:
        remote_branch_exists = False
    return remote_branch_exists


def _diff_built_docs_between_branches(
        docs_branch, docs_cp, repo):
    """Return `True` if `docs_cp` differs on `docs_branch`.

    If any file in the directory tree rooted
    at `docs_cp` has changed, then return `True`.
    Otherwise, return `False`.
    """
    # the footer of the `sphinx`-generated
    # documentation includes a date
    # "Last updated on 05 October 2021",
    # which will trigger a build every month.
    #
    # I think that it is good to keep this timestamp
    # in the documentation. To avoid new pushes to
    # the documentation when the sources are unchanged,
    # a diff modulo timestamps needs to be implemented
    # for the generated documentation files.
    raise NotImplementedError()  # TODO
    remote_branch = f'{origin}/{docs_branch}'
    changed = repo.diff(
        '--exit-code', '--quiet',
        docs_branch, remote_branch,
        '--', docs_cp)


def _push_built_docs(
        changed, docs_branch, repo):
    """If `changed`, then push `docs_branch`.

    Also, print an informational message.
    """
    if not changed:
        print(
            'The built documentation did not change.\n'
            'No `git` pushing to do.')
        return
    print(
        'The built documentation changed, '
        'will now push the changes.')
    repo.git.push('-f', GIT_ORIGIN, docs_branch)


def _pushd(path, stack):
    """Append pwd to `stack`, then `cd path`."""
    pwd = os.getcwd()
    stack.append(pwd)
    os.chdir(path)


def _popd(stack):
    """Pop from `stack`, `cd` to popped path."""
    path = stack.pop()
    os.chdir(path)


def _parse_args():
    """Return parsed command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--docs-src-path',
        help='path to the root of the '
            'directory tree that contains the '
            'source code of the documentation')
    parser.add_argument(
        '--docs-out-path',
        help='path to the root of the directory tree '
            'that contains the built documentation')
    parser.add_argument(
        '--docs-cp-path',
        help='path to which the built documentation is '
            'copied to, i.e., where the root of the '
            'directory tree of the built documentation '
            'is placed when copying the directory tree '
            'of the built documentation')
    parser.add_argument(
        '--binaries-branch',
        help='name of branch from where additional '
            'binary assets are checked out, '
            'before building the documentation')
    parser.add_argument(
        '--docs-branch',
        help='name of branch to which to push '
            'the built documentation')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    _main()