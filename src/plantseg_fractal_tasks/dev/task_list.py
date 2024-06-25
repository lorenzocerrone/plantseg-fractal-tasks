from fractal_tasks_core.dev.task_models import NonParallelTask, ParallelTask

TASK_LIST = [
    NonParallelTask(
        name="Import from PlantSeg H5",
        executable="import_from_plantseg_h5.py",
        meta={"cpus_per_task": 1, "mem": 4000},
    ),
    ParallelTask(
        name="Run Plantseg predictions",
        executable="plantseg_workflow.py",
        meta={"cpus_per_task": 1, "mem": 4000},
    ),
]
