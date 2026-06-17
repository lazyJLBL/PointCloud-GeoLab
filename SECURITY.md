# Security Policy

PointCloud-GeoLab is a learning and portfolio project for point-cloud geometry.
It is not a hosted service and does not process untrusted network traffic by
default.

## Supported Versions

The current `main` branch and the latest tagged release receive security
attention.

## Reporting A Concern

If you find a security issue, please open a GitHub issue with a clear
description and reproduction steps. If the report contains sensitive details,
contact the repository owner first and avoid posting exploit payloads publicly.

Useful details include:

- affected command, script, or module;
- input file format and size;
- expected behavior and actual behavior;
- environment details such as OS and Python version.

## Scope

In scope:

- unsafe file handling in CLI, examples, or scripts;
- dependency or packaging issues that affect normal installation;
- crashes triggered by malformed local input files.

Out of scope:

- claims that require production hardening beyond this portfolio project;
- denial-of-service reports based only on intentionally huge local datasets;
- vulnerabilities in optional third-party packages that should be reported
  upstream.
