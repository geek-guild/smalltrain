#!/bin/bash

echo "Exec setup script in entrypoint(after volumes attached)."

# Setup SmallTrain
/var/work/setup_smalltrain.sh
# Check GPU usage
nvidia-smi
# run SmallTrain
/usr/local/bin/run_smalltrain.sh

tail -f /dev/null
