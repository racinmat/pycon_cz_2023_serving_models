#!/usr/bin/env bash
set -x
if [[ -n $AIP_STORAGE_URI ]]; then
    export MODEL_PATH="/app/models"
    mkdir -p /app/models
    gsutil -m cp -r $AIP_STORAGE_URI/* /app/models
else
    echo "AIP_STORAGE_URI not set, not downloading the model"
fi
exec "$@"
