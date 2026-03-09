#!/bin/bash
# Verify JAX CPU after CUDA plugin removal, then train
set -e
source /home/op3_rl_venv/bin/activate
export JAX_PLATFORMS=cpu

echo "=== Verifying JAX CPU ==="
python -c "
import jax
print(f'JAX {jax.__version__}, backend: {jax.default_backend()}')
print(f'Devices: {jax.devices()}')
import jax.numpy as jnp
x = jnp.ones(10)
print(f'Sum: {float(jnp.sum(x))}')
print('JAX CPU OK!')
" || { echo "FAIL: JAX still broken"; exit 1; }

echo ""
echo "=== Loading Op3Joystick ==="  
python -c "
import os; os.environ['JAX_PLATFORMS'] = 'cpu'
from mujoco_playground import registry
env = registry.load('Op3Joystick')
print(f'Env loaded! action_size={env.action_size}, obs_size={env.observation_size}')
"

echo ""
echo "=== Starting Training ==="
cd /mnt/c/GitHub/OpenServoSim
JAX_PLATFORMS=cpu python training/train_op3_walk.py 2>&1

echo ""
echo "=== ALL DONE ==="
