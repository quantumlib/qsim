# Before you begin

The following tutorials demonstrate how to configure the Google Cloud Platform
to run quantum simulations with qsim.

You can use Google Cloud to run high-performance CPU-based simulations or
GPU-based simulations, depending on your requirements. For more information
about making a choice between CPU- and GPU-based simulations, see
[Choosing hardware for your qsim simulation]().

This tutorial depends on resources provided by the Google Cloud Platform.

*   **Ensure that you have a Google Cloud Platform project.** You can reuse an
    existing project, or create a new one, from your
    [project dashboard](https://console.cloud.google.com/projectselector2/home/dashboard).
    *   For more information about Google Cloud projects, see
        [Creating and managing projects](https://cloud.google.com/resource-manager/docs/creating-managing-projects)
        in the Google Cloud documentation.
*   **Ensure that billing is enabled for your project.**
    *   For more information about billing, see
        [Enable, disable, or change billing for a project](https://cloud.google.com/billing/docs/how-to/modify-project#enable-billing)
        in the Google Cloud documentation.
*   **Estimate costs for your project** Use the
    [Google Cloud Pricing Calculator](https://cloud.google.com/products/calculator)
    to estimate the scale of the costs you might incur, based on your projected
    usage. The resources that you use to simulate a quantum circuit on the
    Google Cloud platform are billable.
*   **Enable the Compute Engine API for your project.** You can enable APIs from
    the [API Library Console](https://console.cloud.google.com/apis/library). On
    the console, in the search box, enter "compute engine api" to find the API
    and click through to Enable it.
    *   For more information about enabling the Compute Engine API, see
        [Getting Started](https://cloud.google.com/apis/docs/getting-started) in
        the Google Cloud documentation.
