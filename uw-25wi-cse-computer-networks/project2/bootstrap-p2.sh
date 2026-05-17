#! /usr/bin/env bash

set -e

if [[ -d /home/vagrant/project-2/pox ]]
then
    ln -s /home/vagrant/project-2/pox/* /home/vagrant/pox/pox/misc/ && echo "Success"
else
    echo "ERROR -> could not find /home/vagrant/project-2/pox/*"
    echo "You might have cloned the bootstrap repo to the wrong location, or might be running as root/sudo?"
    echo "If you're stuck please reach out to the TAs on Ed!"
fi

# Note: This script is very simple now, but remains a script for consistency
# between the projects :D
