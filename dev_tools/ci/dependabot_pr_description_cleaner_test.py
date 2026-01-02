# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: disable=line-too-long

import dependabot_pr_description_cleaner as cleaner
import pytest

# Sample 1
SAMPLE_1 = r"""Bumps [raven-actions/actionlint](https://github.com/raven-actions/actionlint) from 2.0.1 to 2.1.0.
<details>
<summary>Release notes</summary>
<p><em>Sourced from <a href="https://github.com/raven-actions/actionlint/releases">raven-actions/actionlint's releases</a>.</em></p>
<blockquote>
<h2>v2.1.0</h2>
<h2>🔄️ What's Changed</h2>
<ul>
<li>update action versions in workflows and action metadata <a href="https://github.com/DariuszPorowski"><code>@​DariuszPorowski</code></a> (<a href="https://redirect.github.com/raven-actions/actionlint/issues/46">#46</a>)</li>
</ul>
<h2>👥 Contributors</h2>
<p><a href="https://github.com/DariuszPorowski"><code>@​DariuszPorowski</code></a></p>
<p>See details of all code changes: <a href="https://github.com/raven-actions/actionlint/compare/v2.0.2...v2.1.0">https://github.com/raven-actions/actionlint/compare/v2.0.2...v2.1.0</a> since previous release.</p>
<h2>v2.0.2</h2>
<h2>🔄️ What's Changed</h2>
<ul>
<li>ci(deps): bump actions/checkout from 5 to 6 in the all group @<a href="https://github.com/apps/dependabot">dependabot[bot]</a> (<a href="https://redirect.github.com/raven-actions/actionlint/issues/45">#45</a>)</li>
<li>fix: don't interfere with repo package.json <a href="https://github.com/allejo"><code>@​allejo</code></a> (<a href="https://redirect.github.com/raven-actions/actionlint/issues/43">#43</a>)</li>
<li>ci(deps): bump actions/cache from 4.2.4 to 4.3.0 in the all group @<a href="https://github.com/apps/dependabot">dependabot[bot]</a> (<a href="https://redirect.github.com/raven-actions/actionlint/issues/42">#42</a>)</li>
<li>ci(deps): bump actions/github-script from 7.0.1 to 8.0.0 in the all group @<a href="https://github.com/apps/dependabot">dependabot[bot]</a> (<a href="https://redirect.github.com/raven-actions/actionlint/issues/41">#41</a>)</li>
<li>ci(deps): bump the all group across 1 directory with 2 updates @<a href="https://github.com/apps/dependabot">dependabot[bot]</a> (<a href="https://redirect.github.com/raven-actions/actionlint/issues/40">#40</a>)</li>
<li>chore: synced file(s) with raven-actions/.workflows @<a href="https://github.com/apps/raven-actions-sync">raven-actions-sync[bot]</a> (<a href="https://redirect.github.com/raven-actions/actionlint/issues/38">#38</a>)</li>
<li>chore: synced file(s) with raven-actions/.workflows @<a href="https://github.com/apps/raven-actions-sync">raven-actions-sync[bot]</a> (<a href="https://redirect.github.com/raven-actions/actionlint/issues/37">#37</a>)</li>
<li>chore: synced file(s) with raven-actions/.workflows @<a href="https://github.com/apps/raven-actions-sync">raven-actions-sync[bot]</a> (<a href="https://redirect.github.com/raven-actions/actionlint/issues/36">#36</a>)</li>
<li>chore: synced file(s) with raven-actions/.workflows @<a href="https://github.com/apps/raven-actions-sync">raven-actions-sync[bot]</a> (<a href="https://redirect.github.com/raven-actions/actionlint/issues/35">#35</a>)</li>
<li>chore: synced file(s) with raven-actions/.workflows @<a href="https://github.com/apps/raven-actions-sync">raven-actions-sync[bot]</a> (<a href="https://redirect.github.com/raven-actions/actionlint/issues/34">#34</a>)</li>
<li>chopre: delete .pre-commit-config.yaml <a href="https://github.com/DariuszPorowski"><code>@​DariuszPorowski</code></a> (<a href="https://redirect.github.com/raven-actions/actionlint/issues/33">#33</a>)</li>
<li>chore: synced file(s) with raven-actions/.workflows @<a href="https://github.com/apps/raven-actions-sync">raven-actions-sync[bot]</a> (<a href="https://redirect.github.com/raven-actions/actionlint/issues/32">#32</a>)</li>
</ul>
<h2>👥 Contributors</h2>
<p><a href="https://github.com/DariuszPorowski"><code>@​DariuszPorowski</code></a>, <a href="https://github.com/allejo"><code>@​allejo</code></a>, <a href="https://github.com/dependabot"><code>@​dependabot</code></a>[bot], <a href="https://github.com/raven-actions-sync"><code>@​raven-actions-sync</code></a>[bot], <a href="https://github.com/apps/dependabot">dependabot[bot]</a> and <a href="https://github.com/apps/raven-actions-sync">raven-actions-sync[bot]</a></p>
<p>See details of all code changes: <a href="https://github.com/raven-actions/actionlint/compare/v2.0.1...v2.0.2">https://github.com/raven-actions/actionlint/compare/v2.0.1...v2.0.2</a> since previous release.</p>
</blockquote>
</details>
<details>
<summary>Commits</summary>
<ul>
<li><a href="https://github.com/raven-actions/actionlint/commit/963d4779ef039e217e5d0e6fd73ce9ab7764e493"><code>963d477</code></a> ci: update action versions in workflows and action metadata (<a href="https://redirect.github.com/raven-actions/actionlint/issues/46">#46</a>)</li>
<li><a href="https://github.com/raven-actions/actionlint/commit/591b1dfdbabb809cef4c22b7b93f37cd994d9368"><code>591b1df</code></a> ci(deps): bump actions/checkout from 5 to 6 in the all group (<a href="https://redirect.github.com/raven-actions/actionlint/issues/45">#45</a>)</li>
<li><a href="https://github.com/raven-actions/actionlint/commit/fd1b743b251ce06a07f899ad2e53e238e94df30d"><code>fd1b743</code></a> fix: don't interfere with repo package.json (<a href="https://redirect.github.com/raven-actions/actionlint/issues/43">#43</a>)</li>
<li><a href="https://github.com/raven-actions/actionlint/commit/14ed4492b05535f0f88654359d6186ef4adea727"><code>14ed449</code></a> ci(deps): bump actions/cache from 4.2.4 to 4.3.0 in the all group (<a href="https://redirect.github.com/raven-actions/actionlint/issues/42">#42</a>)</li>
<li><a href="https://github.com/raven-actions/actionlint/commit/0ecc78948d0f2638a712944e00ac7a986405c2cd"><code>0ecc789</code></a> ci(deps): bump actions/github-script from 7.0.1 to 8.0.0 in the all group (<a href="https://redirect.github.com/raven-actions/actionlint/issues/41">#41</a>)</li>
<li><a href="https://github.com/raven-actions/actionlint/commit/193928d7453e51825534c0e4ecf3b351e33421d7"><code>193928d</code></a> ci(deps): bump the all group across 1 directory with 2 updates (<a href="https://redirect.github.com/raven-actions/actionlint/issues/40">#40</a>)</li>
<li><a href="https://github.com/raven-actions/actionlint/commit/a9f1bde40b4c45b3f92816d3834ec9e719895a0e"><code>a9f1bde</code></a> chore: synced file(s) with raven-actions/.workflows (<a href="https://redirect.github.com/raven-actions/actionlint/issues/38">#38</a>)</li>
<li><a href="https://github.com/raven-actions/actionlint/commit/e35ca82cf59bb9307dc5d1b736a7f39fe2c2101e"><code>e35ca82</code></a> chore: synced file(s) with raven-actions/.workflows (<a href="https://redirect.github.com/raven-actions/actionlint/issues/37">#37</a>)</li>
<li><a href="https://github.com/raven-actions/actionlint/commit/81552547fe0ddf263672fe282716e70633b2f155"><code>8155254</code></a> chore: synced file(s) with raven-actions/.workflows (<a href="https://redirect.github.com/raven-actions/actionlint/issues/36">#36</a>)</li>
<li><a href="https://github.com/raven-actions/actionlint/commit/0d9faa245f7f0d94835f7f0b199f7069a27180db"><code>0d9faa2</code></a> chore: synced file(s) with raven-actions/.workflows (<a href="https://redirect.github.com/raven-actions/actionlint/issues/35">#35</a>)</li>
<li>Additional commits viewable in <a href="https://github.com/raven-actions/actionlint/compare/3a24062651993d40fed1019b58ac6fbdfbf276cc...963d4779ef039e217e5d0e6fd73ce9ab7764e493">compare view</a></li>
</ul>
</details>
<br />


[![Dependabot compatibility score](https://dependabot-badges.githubapp.com/badges/compatibility_score?dependency-name=raven-actions/actionlint&package-manager=github_actions&previous-version=2.0.1&new-version=2.1.0)](https://docs.github.com/en/github/managing-security-vulnerabilities/about-dependabot-security-updates#about-compatibility-scores)

Dependabot will resolve any conflicts with this PR as long as you don't alter it yourself. You can also trigger a rebase manually by commenting `@dependabot rebase`.

[//]: # (dependabot-automerge-start)
[//]: # (dependabot-automerge-end)

---

<details>
<summary>Dependabot commands and options</summary>
<br />

You can trigger Dependabot actions by commenting on this PR:
- `@dependabot rebase` will rebase this PR
- `@dependabot recreate` will recreate this PR, overwriting any edits that have been made to it
- `@dependabot merge` will merge this PR after your CI passes on it
- `@dependabot squash and merge` will squash and merge this PR after your CI passes on it
- `@dependabot cancel merge` will cancel a previously requested merge and block automerging
- `@dependabot reopen` will reopen this PR if it is closed
- `@dependabot close` will close this PR and stop Dependabot recreating it. You can achieve the same result by closing it manually
- `@dependabot show <dependency name> ignore conditions` will show all of the ignore conditions of the specified dependency
- `@dependabot ignore this major version` will close this PR and stop Dependabot creating any more for this major version (unless you reopen the PR or upgrade to it yourself)
- `@dependabot ignore this minor version` will close this PR and stop Dependabot creating any more for this minor version (unless you reopen the PR or upgrade to it yourself)
- `@dependabot ignore this dependency` will close this PR and stop Dependabot creating any more for this dependency (unless you reopen the PR or upgrade to it yourself)


</details>
"""

# Sample 2
SAMPLE_2 = r"""Bumps [actions/checkout](https://github.com/actions/checkout) from 6.0.0 to 6.0.1.
<details>
<summary>Release notes</summary>
<p><em>Sourced from <a href="https://github.com/actions/checkout/releases">actions/checkout's releases</a>.</em></p>
<blockquote>
<h2>v6.0.1</h2>
<h2>What's Changed</h2>
<ul>
<li>Update all references from v5 and v4 to v6 by <a href="https://github.com/ericsciple"><code>@​ericsciple</code></a> in <a href="https://redirect.github.com/actions/checkout/pull/2314">actions/checkout#2314</a></li>
<li>Add worktree support for persist-credentials includeIf by <a href="https://github.com/ericsciple"><code>@​ericsciple</code></a> in <a href="https://redirect.github.com/actions/checkout/pull/2327">actions/checkout#2327</a></li>
<li>Clarify v6 README by <a href="https://github.com/ericsciple"><code>@​ericsciple</code></a> in <a href="https://redirect.github.com/actions/checkout/pull/2328">actions/checkout#2328</a></li>
</ul>
<p><strong>Full Changelog</strong>: <a href="https://github.com/actions/checkout/compare/v6...v6.0.1">https://github.com/actions/checkout/compare/v6...v6.0.1</a></p>
</blockquote>
</details>
<details>
<summary>Commits</summary>
<ul>
<li><a href="https://github.com/actions/checkout/commit/8e8c483db84b4bee98b60c0593521ed34d9990e8"><code>8e8c483</code></a> Clarify v6 README (<a href="https://redirect.github.com/actions/checkout/issues/2328">#2328</a>)</li>
<li><a href="https://github.com/actions/checkout/commit/033fa0dc0b82693d8986f1016a0ec2c5e7d9cbb1"><code>033fa0d</code></a> Add worktree support for persist-credentials includeIf (<a href="https://redirect.github.com/actions/checkout/issues/2327">#2327</a>)</li>
<li><a href="https://github.com/actions/checkout/commit/c2d88d3ecc89a9ef08eebf45d9637801dcee7eb5"><code>c2d88d3</code></a> Update all references from v5 and v4 to v6 (<a href="https://redirect.github.com/actions/checkout/issues/2314">#2314</a>)</li>
<li>See full diff in <a href="https://github.com/actions/checkout/compare/1af3b93b6815bc44a9784bd300feb67ff0d1eeb3...8e8c483db84b4bee98b60c0593521ed34d9990e8">compare view</a></li>
</ul>
</details>
<br />


[![Dependabot compatibility score](https://dependabot-badges.githubapp.com/badges/compatibility_score?dependency-name=actions/checkout&package-manager=github_actions&previous-version=6.0.0&new-version=6.0.1)](https://docs.github.com/en/github/managing-security-vulnerabilities/about-dependabot-security-updates#about-compatibility-scores)

Dependabot will resolve any conflicts with this PR as long as you don't alter it yourself. You can also trigger a rebase manually by commenting `@dependabot rebase`.

[//]: # (dependabot-automerge-start)
[//]: # (dependabot-automerge-end)

---

<details>
<summary>Dependabot commands and options</summary>
<br />

You can trigger Dependabot actions by commenting on this PR:
- `@dependabot rebase` will rebase this PR
- `@dependabot recreate` will recreate this PR, overwriting any edits that have been made to it
- `@dependabot merge` will merge this PR after your CI passes on it
- `@dependabot squash and merge` will squash and merge this PR after your CI passes on it
- `@dependabot cancel merge` will cancel a previously requested merge and block automerging
- `@dependabot reopen` will reopen this PR if it is closed
- `@dependabot close` will close this PR and stop Dependabot recreating it. You can achieve the same result by closing it manually
- `@dependabot show <dependency name> ignore conditions` will show all of the ignore conditions of the specified dependency
- `@dependabot ignore this major version` will close this PR and stop Dependabot creating any more for this major version (unless you reopen the PR or upgrade to it yourself)
- `@dependabot ignore this minor version` will close this PR and stop Dependabot creating any more for this minor version (unless you reopen the PR or upgrade to it yourself)
- `@dependabot ignore this dependency` will close this PR and stop Dependabot creating any more for this dependency (unless you reopen the PR or upgrade to it yourself)


</details>
"""

# Sample 3
SAMPLE_3 = r"""Updates the requirements on [cmake](https://github.com/scikit-build/cmake-python-distributions) to permit the latest version.
<details>
<summary>Release notes</summary>
<p><em>Sourced from <a href="https://github.com/scikit-build/cmake-python-distributions/releases">cmake's releases</a>.</em></p>
<blockquote>
<h2>4.2.1</h2>
<!-- raw HTML omitted -->
<h2>What's Changed</h2>
<ul>
<li>fix: add missing f-string prefix for --parallel bootstrap arg by <a href="https://github.com/shanemcd"><code>@​shanemcd</code></a> in <a href="https://redirect.github.com/scikit-build/cmake-python-distributions/pull/665">scikit-build/cmake-python-distributions#665</a></li>
<li>fix: workaround issue in lastversion with OpenSSL by <a href="https://github.com/mayeut"><code>@​mayeut</code></a> in <a href="https://redirect.github.com/scikit-build/cmake-python-distributions/pull/669">scikit-build/cmake-python-distributions#669</a></li>
<li>chore(deps): update clang to 21.1.8.0 by <a href="https://github.com/mayeut"><code>@​mayeut</code></a> in <a href="https://redirect.github.com/scikit-build/cmake-python-distributions/pull/670">scikit-build/cmake-python-distributions#670</a></li>
<li>[Bot] Update to CMake 4.2.1 by <a href="https://github.com/scikit-build-app-bot"><code>@​scikit-build-app-bot</code></a>[bot] in <a href="https://redirect.github.com/scikit-build/cmake-python-distributions/pull/666">scikit-build/cmake-python-distributions#666</a></li>
</ul>
<h2>New Contributors</h2>
<ul>
<li><a href="https://github.com/shanemcd"><code>@​shanemcd</code></a> made their first contribution in <a href="https://redirect.github.com/scikit-build/cmake-python-distributions/pull/665">scikit-build/cmake-python-distributions#665</a></li>
</ul>
<p><strong>Full Changelog</strong>: <a href="https://github.com/scikit-build/cmake-python-distributions/compare/4.2.0...4.2.1">https://github.com/scikit-build/cmake-python-distributions/compare/4.2.0...4.2.1</a></p>
</blockquote>
</details>
<details>
<summary>Commits</summary>
<ul>
<li><a href="https://github.com/scikit-build/cmake-python-distributions/commit/535af4c8b8d2711ceabff507875fa27953a8a9e6"><code>535af4c</code></a> [Bot] Update to CMake 4.2.1 (<a href="https://redirect.github.com/scikit-build/cmake-python-distributions/issues/666">#666</a>)</li>
<li><a href="https://github.com/scikit-build/cmake-python-distributions/commit/b9233cd026c7138ad8a9a4e5e9d293fdc41204f4"><code>b9233cd</code></a> chore(deps): update clang to 21.1.8.0 (<a href="https://redirect.github.com/scikit-build/cmake-python-distributions/issues/670">#670</a>)</li>
<li><a href="https://github.com/scikit-build/cmake-python-distributions/commit/5a1434b51fed97a2a7e743722ec3164571572cae"><code>5a1434b</code></a> fix: workaround issue in lastversion with OpenSSL (<a href="https://redirect.github.com/scikit-build/cmake-python-distributions/issues/669">#669</a>)</li>
<li><a href="https://github.com/scikit-build/cmake-python-distributions/commit/dd81945351bfa635d387d1911110f04d552f9aec"><code>dd81945</code></a> chore(deps): bump the actions group with 3 updates (<a href="https://redirect.github.com/scikit-build/cmake-python-distributions/issues/668">#668</a>)</li>
<li><a href="https://github.com/scikit-build/cmake-python-distributions/commit/40acf75985b28bd112b28ddc4506f5e0283cbd45"><code>40acf75</code></a> chore(deps): update pre-commit hooks (<a href="https://redirect.github.com/scikit-build/cmake-python-distributions/issues/664">#664</a>)</li>
<li><a href="https://github.com/scikit-build/cmake-python-distributions/commit/a28030b84ad12ef814367f125e995bac9ddde61c"><code>a28030b</code></a> fix: add missing f-string prefix for --parallel bootstrap arg (<a href="https://redirect.github.com/scikit-build/cmake-python-distributions/issues/665">#665</a>)</li>
<li><a href="https://github.com/scikit-build/cmake-python-distributions/commit/aa7b6bfddec7549d25b5b0a9dcb8024a64815947"><code>aa7b6bf</code></a> chore(deps): bump the actions group with 3 updates (<a href="https://redirect.github.com/scikit-build/cmake-python-distributions/issues/663">#663</a>)</li>
<li><a href="https://github.com/scikit-build/cmake-python-distributions/commit/a611e250e1e0a39309e3083c9846b4b81f333e37"><code>a611e25</code></a> chore(deps): update clang to 21.1.6.0 (<a href="https://redirect.github.com/scikit-build/cmake-python-distributions/issues/661">#661</a>)</li>
<li><a href="https://github.com/scikit-build/cmake-python-distributions/commit/c89cf393ba25a148cc04c57a1e1ccb965fa1cabf"><code>c89cf39</code></a> chore: monthly updates (<a href="https://redirect.github.com/scikit-build/cmake-python-distributions/issues/660">#660</a>)</li>
<li><a href="https://github.com/scikit-build/cmake-python-distributions/commit/597f1588a98dd58627940cc9f415c15f94b1d175"><code>597f158</code></a> chore: add changelog exclusion for bots (<a href="https://redirect.github.com/scikit-build/cmake-python-distributions/issues/658">#658</a>)</li>
<li>Additional commits viewable in <a href="https://github.com/scikit-build/cmake-python-distributions/compare/3.28.1...4.2.1">compare view</a></li>
</ul>
</details>
<br />


Dependabot will resolve any conflicts with this PR as long as you don't alter it yourself. You can also trigger a rebase manually by commenting `@dependabot rebase`.

[//]: # (dependabot-automerge-start)
[//]: # (dependabot-automerge-end)

---

<details>
<summary>Dependabot commands and options</summary>
<br />

You can trigger Dependabot actions by commenting on this PR:
- `@dependabot rebase` will rebase this PR
- `@dependabot recreate` will recreate this PR, overwriting any edits that have been made to it
- `@dependabot merge` will merge this PR after your CI passes on it
- `@dependabot squash and merge` will squash and merge this PR after your CI passes on it
- `@dependabot cancel merge` will cancel a previously requested merge and block automerging
- `@dependabot reopen` will reopen this PR if it is closed
- `@dependabot close` will close this PR and stop Dependabot recreating it. You can achieve the same result by closing it manually
- `@dependabot show <dependency name> ignore conditions` will show all of the ignore conditions of the specified dependency
- `@dependabot ignore this major version` will close this PR and stop Dependabot creating any more for this major version (unless you reopen the PR or upgrade to it yourself)
- `@dependabot ignore this minor version` will close this PR and stop Dependabot creating any more for this minor version (unless you reopen the PR or upgrade to it yourself)
- `@dependabot ignore this dependency` will close this PR and stop Dependabot creating any more for this dependency (unless you reopen the PR or upgrade to it yourself)


</details>
"""


def test_process_text_basic_conversion():
    html_input = "<p>Hello <b>World</b></p>"
    expected_output = "Hello **World**"
    result = cleaner.process_text(html_input)
    assert result.strip() == expected_output


def test_remove_commits_section():
    html_input = """
    <p>Some text</p>
    <details>
    <summary>Commits</summary>
    <ul><li>Commit 1</li></ul>
    </details>
    <br />
    <p>More text</p>
    """
    # "Commits" section should be removed.
    # html2text introduces newlines.
    result = cleaner.process_text(html_input)

    assert "Commits" not in result
    assert "Commit 1" not in result
    assert "Some text" in result
    assert "More text" in result


def test_remove_dependabot_commands():
    html_input = """
    <p>Content</p>
    <hr />
    <details>
    <summary>Dependabot commands and options</summary>
    <p>commands</p>
    </details>
    """
    result = cleaner.process_text(html_input)

    assert "Content" in result
    assert "Dependabot commands and options" not in result


def test_basic_example():
    # A simplified version of a real Dependabot body
    html_input = """
    Bumps [package](link) from 1.0 to 2.0.
    <details>
    <summary>Release notes</summary>
    <p>Notes</p>
    </details>
    <details>
    <summary>Commits</summary>
    <ul><li>123456</li></ul>
    </details>
    <br />
    ---
    <details>
    <summary>Dependabot commands and options</summary>
    <p>help</p>
    </details>
    """
    result = cleaner.process_text(html_input)

    assert "Bumps [package](link) from 1.0 to 2.0." in result
    assert "Release notes" in result
    assert "Commits" not in result
    assert "123456" not in result
    assert "Dependabot commands and options" not in result


@pytest.mark.parametrize("input_content", [SAMPLE_1, SAMPLE_2, SAMPLE_3])
def test_clean_real_samples(input_content):
    cleaned = cleaner.process_text(input_content)

    assert cleaned is not None
    assert "Dependabot commands and options" not in cleaned
    assert "<details>" not in cleaned
    assert "</details>" not in cleaned
    assert "Release notes" in cleaned
    assert "Commits" not in cleaned


def test_idempotency():
    cleaned = cleaner.process_text(SAMPLE_1)
    cleaned_again = cleaner.process_text(cleaned)
    assert "Dependabot commands and options" not in cleaned_again

    # Check that links are preserved (and not escaped).
    assert (
        "[raven-actions/actionlint](https://github.com/raven-actions/actionlint)"
        in cleaned_again
    )
    assert "\\[raven-actions/actionlint\\]" not in cleaned_again


def test_markdown_preservation():
    # Ensure links are not escaped (basic check)
    cleaned = cleaner.process_text(SAMPLE_1)
    assert (
        "[raven-actions/actionlint](https://github.com/raven-actions/actionlint)"
        in cleaned
    )


def test_custom_separator_removal():
    content = "Some content\n\n--- \nDependabot commands and options\nJunk at the end"
    cleaned = cleaner.process_text(content)
    assert "Some content" in cleaned
    assert "Dependabot commands and options" not in cleaned
    assert "Junk at the end" not in cleaned
    assert "---" not in cleaned


def test_no_separator_removal():
    content = "Some content\nDependabot commands and options\nJunk at the end"
    cleaned = cleaner.process_text(content)
    assert "Some content" in cleaned
    assert "Dependabot commands and options" not in cleaned
    assert "Junk at the end" not in cleaned
