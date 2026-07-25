# Shared RViz configuration

`vehicle_simulator.rviz` is mounted into the system container by
`compose_rviz.yml`, so the SysNav instruction target marker is enabled without
manual RViz setup.

Start or recreate the system container with:

```bash
docker compose -f compose_gpu.yml -f compose_rviz.yml up -d system
```

Then launch the simulation inside `iros2026_system` as usual:

```bash
/home/docker/autonomy_stack_mecanum_wheel_platform/system_simulation.sh
```
