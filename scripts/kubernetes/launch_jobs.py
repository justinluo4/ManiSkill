import os
import subprocess
import glob
import time
from string import Template

# --- Configuration ---
LOCAL_MODELS_DIR = os.path.expanduser("~/.maniskill/data/tasks/grasping/mani_skill2_ycb/models")
START_OBJECT_NUM = 0
TOTAL_TIMESTEPS = "20_000_000"
IMAGE_NAME = "gitlab-registry.nrp-nautilus.io/jluo2/ucsd:latest"
MAX_RUNNING_JOBS = 10

PVC_NAME = "maniskill-pvc" 

job_template = Template("""
apiVersion: batch/v1
kind: Job
metadata:
  name: maniskill-ppo-${sanitized_name}
  labels:
    app: maniskill-sweep
    object: ${object_name}
spec:
  backoffLimit: 2
  template:
    metadata:
      labels:
        app: maniskill-sweep
    spec:
      restartPolicy: OnFailure
      containers:
      - name: runner
        image: ${image}
        imagePullPolicy: IfNotPresent
        env:
        - name: WANDB_API_KEY
          valueFrom:
            secretKeyRef:
              name: jluo-wandb
              key: WANDB_API_KEY
        resources:
          limits:
            cpu: "1"
            memory: 8Gi
            nvidia.com/gpu: 1
            ephemeral-storage: 100Gi
          requests:
            cpu: "1"
            memory: 8Gi
            nvidia.com/gpu: 1
            ephemeral-storage: 100Gi
        
        # --- MOUNT CONFIGURATION ---
        volumeMounts:
        # 1. Mount the PVC to /data
        - name: persistent-storage
          mountPath: /data
        
        # 2. (Optional) Keep Shm for PyTorch speed
        - name: dshm
          mountPath: /dev/shm
        
        command: ["/bin/bash", "-lc"]
        args:
        - |
          set -euo pipefail
          echo "Starting ManiSkill PPO runs inside Kubernetes pod"
          
          # Clone ManiSkill repo if not already present
          if [ ! -d "/opt/ManiSkill" ]; then
            echo "Cloning ManiSkill repository..."
            git clone --depth 1 --branch main "https://github.com/justinluo4/ManiSkill.git" /opt/ManiSkill
          fi
          
          cd /opt/ManiSkill
          
          # Install dependencies
          echo "Installing ManiSkill dependencies..."
          conda install python=3.12
          pip install --upgrade pip
          pip install -e .
          pip install torch tensorboard
          
          # Download physx GPU binary via sapien
          echo "Downloading physx GPU binary..."
          python -c "import sapien.physx as physx; physx.enable_gpu()" || true
          mkdir -p /root/.maniskill/data/tasks/grasping/mani_skill2_ycb
          ln -s /data/models /root/.maniskill/data/tasks/grasping/mani_skill2_ycb/

          echo "Starting Job for Object: ${object_name}"
          
          

          cd /opt/ManiSkill
          
          echo "Running PPO (use_decomp)..."
          python examples/baselines/ppo/ppo.py \\
            --env_id="CustomPick-v1" \\
            --num_envs=1024 \\
            --update_epochs=8 \\
            --num_minibatches=32 \\
            --total_timesteps="${timesteps}" \\
            --eval_freq=10 \\
            --num-steps=20 \\
            --track \\
            --pick_object_name "${object_name}" \\
            --wandb_project_name="maniskill-ppo" \\
            --use_decomp

          echo "Running PPO (no decomp)..."
          python examples/baselines/ppo/ppo.py \\
            --env_id="CustomPick-v1" \\
            --num_envs=1024 \\
            --update_epochs=8 \\
            --num_minibatches=32 \\
            --total_timesteps="${timesteps}" \\
            --eval_freq=10 \\
            --num-steps=20 \\
            --track \\
            --pick_object_name "${object_name}" \\
            --wandb_project_name="maniskill-ppo"

      volumes:
      # --- VOLUME DEFINITION ---
      - name: persistent-storage
        persistentVolumeClaim:
          claimName: ${pvc_name}
      - name: dshm
        emptyDir:
          medium: Memory
""")

def get_running_job_count():
    """Get the number of currently running jobs."""
    try:
        # Get all jobs with the app label and check their active status
        cmd = ['kubectl', 'get', 'jobs', '-l', 'app=maniskill-sweep', 
               '-o', 'jsonpath={range .items[*]}{.status.active}{"\\n"}{end}']
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            return 0
        
        # Count jobs with active pods (status.active > 0)
        count = 0
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                try:
                    active = int(line.strip())
                    if active > 0:
                        count += 1
                except ValueError:
                    # If it's not a number, skip it
                    pass
        return count
    except Exception as e:
        print(f"Warning: Error checking running jobs: {e}")
        return 0

def wait_for_job_slot():
    """Wait until there's room for a new job (less than MAX_RUNNING_JOBS running)."""
    while True:
        running_count = get_running_job_count()
        if running_count < MAX_RUNNING_JOBS:
            return running_count
        print(f"  Waiting for job slot... ({running_count}/{MAX_RUNNING_JOBS} jobs running)")
        time.sleep(10)  # Check every 10 seconds

def main():
    if not os.path.exists(LOCAL_MODELS_DIR):
        print(f"Error: Models directory not found at {LOCAL_MODELS_DIR}")
        exit(1)

    subdirs = glob.glob(os.path.join(LOCAL_MODELS_DIR, "*/"))
    subdirs.sort()
    
    for object_dir in subdirs:
        object_name = os.path.basename(os.path.normpath(object_dir))
        
        try:
            prefix_str = object_name[:3]
            object_num = int(prefix_str)
        except ValueError:
            object_num = -1

        if START_OBJECT_NUM > 0 and 0 <= object_num < START_OBJECT_NUM:
            continue

        sanitized_name = object_name.replace("_", "-").lower()
        job_name = f"maniskill-ppo-{sanitized_name}"
        
        # Wait until there's room for a new job
        running_count = wait_for_job_slot()
        print(f"Submitting Job for: {object_name}... ({running_count}/{MAX_RUNNING_JOBS} jobs running)")

        # Delete existing job if it exists (Jobs have immutable spec.template)
        delete_cmd = ['kubectl', 'delete', 'job', job_name, '--ignore-not-found=true']
        delete_process = subprocess.run(delete_cmd, capture_output=True, text=True)
        if delete_process.returncode == 0 and delete_process.stdout:
            print(f"  Deleted existing job: {job_name}")

        k8s_manifest = job_template.substitute(
            object_name=object_name,
            sanitized_name=sanitized_name,
            timesteps=TOTAL_TIMESTEPS,
            image=IMAGE_NAME,
            pvc_name=PVC_NAME
        )

        # Debug: write YAML to file for inspection
        if object_name == "077_rubiks_cube":
            with open(f"/tmp/debug_{sanitized_name}.yaml", "w") as f:
                f.write(k8s_manifest)
            print(f"Debug: Wrote YAML to /tmp/debug_{sanitized_name}.yaml")

        process = subprocess.Popen(['kubectl', 'apply', '-f', '-'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate(input=k8s_manifest)

        if process.returncode != 0:
            print(f"Error submitting {object_name}: {stderr}")

if __name__ == "__main__":
    main()