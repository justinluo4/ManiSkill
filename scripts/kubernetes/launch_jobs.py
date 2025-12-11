import os
import subprocess
import glob
from string import Template

# --- Configuration ---
LOCAL_MODELS_DIR = os.path.expanduser("~/.maniskill/data/tasks/grasping/mani_skill2_ycb/models")
START_OBJECT_NUM = 76
TOTAL_TIMESTEPS = "2_000_000"
IMAGE_NAME = "gitlab-registry.nrp-nautilus.io/jluo2/ucsd:latest"

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
  backoffLimit: 0 
  template:
    metadata:
      labels:
        app: maniskill-sweep
    spec:
      restartPolicy: Never
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
            cpu: "4"
            memory: 16Gi
            nvidia.com/gpu: 1
          requests:
            cpu: "2"
            memory: 8Gi
            nvidia.com/gpu: 1
        
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
            --pick_object_name "${object_name}"

      volumes:
      # --- VOLUME DEFINITION ---
      - name: persistent-storage
        persistentVolumeClaim:
          claimName: ${pvc_name}
      - name: dshm
        emptyDir:
          medium: Memory
""")

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
        print(f"Submitting Job for: {object_name}...")

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