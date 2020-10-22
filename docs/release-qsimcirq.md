# Release process

The qsimcirq release is maintained by Google contributors through a dedicated
Kokoro build (a Google-internal variant of Jenkins). If you are a Google
contributor and need to cut a release, please see `RELEASE.md` in the
Google-internal fork of qsim.

<!-- TODO(95-martin-orion): redirect to internal docs when available -->

## Version semantics

Version numbering follows the semantic versioning guidelines at
[semver.org](https://semver.org/), whose summary is copied below:

> Given a version number MAJOR.MINOR.PATCH, increment the:
> 
>   1. MAJOR version when you make incompatible API changes,
>   2. MINOR version when you add functionality in a backwards compatible manner, and
>   3. PATCH version when you make backwards compatible bug fixes.
> 
> Additional labels for pre-release and build metadata are available
> as extensions to the MAJOR.MINOR.PATCH format.

Note that this behavior is altered for MAJOR version zero (0.y.z):

> Major version zero (0.y.z) is for initial development. Anything MAY change at
> any time. The public API SHOULD NOT be considered stable.
