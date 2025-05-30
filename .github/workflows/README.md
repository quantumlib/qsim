# GitHub Actions workflows

Herea are some notes about the workflows in this directory:

  Some workflows are reusable and are meant to called from other workflows
  only, not invoked directly. As of mid-2025, GitHub's user interface for
  Actions does not provide a way to hide, group, or otherwise distinguish these
  kinds of reusable modular workflows from the main workflows. To make these
  workflows more apparent in the user interface, we use the convention of
  naming these workflows with a leading tilde (`~`) character; this makes them
  appear last in the list of workflows on GitHub.
