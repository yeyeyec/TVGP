# Upstream source provenance

The necessary Python files under `vendor/JumpGaussianProcess` come from:

```text
Repository: https://github.com/crushonyfg/JumpGaussianProcess
Commit:    2eccbc30ceb4d38521a88137856b65761f4a9030
Author-page link: https://www.chiwoopark.net/code-and-dataset
```

The vendored subset was compared with the GitHub tarball for that commit. Only
the two changes in `jumpgp_compatibility.patch` differ from upstream model
source. `UPSTREAM_README.md` is the upstream `README.md` under a descriptive
name.

The repository README states that the project is MIT licensed, but no separate
`LICENSE` file is present in the fixed snapshot. In addition, the author's code
page permits academic and non-commercial use with attribution while stating
that distribution and modification require prior consent. This directory is
therefore prepared for the present academic analysis. Permission should be
confirmed before redistributing the vendored source in a public manuscript
repository or software release.

The bundled CRAN source `R/lhs_1.2.0.tar.gz` is version 1.2.0 of the `lhs`
package by Rob Carnell, licensed GPL-3. Its SHA-256 digest is:

```text
6b4b773c6b322f1ffb3e2ef49c244ec8a3958a0346a1cc8f42f0d0951e8b0724
```
