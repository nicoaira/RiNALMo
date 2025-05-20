#!/usr/bin/env bash
# Usage: ./rand_rna.sh N

if [[ -z "$1" || "$1" -lt 1 ]]; then
  echo "Usage: $0 <positive_integer_length>"
  exit 1
fi

# Read from /dev/urandom, filter to ACGU, take first N characters
tr -dc 'ACGU' < /dev/urandom | head -c "$1"
echo

