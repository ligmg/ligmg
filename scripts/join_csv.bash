#!/bin/bash

cat <(head -n 1 $1) <(tail -q -n +2 "$@")
