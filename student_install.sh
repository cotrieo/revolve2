#!/bin/sh

# See https://github.com/pypa/pip3/issues/10216
# for why it is at the time of writing not possible to create a student_requirements.txt

cd "$(dirname "$0")"

pip3 install -e ./ci_group[dev] && \
pip3 install -e ./simulators/mujoco[dev] && \
pip3 install -e ./experimentation[dev] && \
pip3 install -e ./rpi_controller_remote[dev] && \
pip3 install -e ./rpi_controller[dev] && \
pip3 install -e ./modular_robot[dev] && \
pip3 install -e ./simulation[dev] && \
pip3 install -e ./actor_controller[dev] && \
pip3 install -e ./serialization[dev] && \
pip3 install -r ./examples/robot_bodybrain_ea_database/requirements.txt && \
pip3 install -r ./examples/robot_brain_cmaes_database/requirements.txt && \
pip3 install -r ./examples/simple_ea_xor_database/requirements.txt
