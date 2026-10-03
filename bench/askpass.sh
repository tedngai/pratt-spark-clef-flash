#!/bin/sh
# GIT_ASKPASS helper: supplies x-access-token / token from the environment.
case "$1" in
  *Username*) echo "x-access-token" ;;
  *) echo "$GITHUB_TOKEN" ;;
esac
