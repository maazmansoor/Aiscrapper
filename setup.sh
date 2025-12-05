#!/bin/bash

# Create a directory for playwright if it doesn't exist
mkdir -p ~/.cache/ms-playwright

# Install playwright's browser dependencies
playwright install-deps

# Install playwright browsers
playwright install
