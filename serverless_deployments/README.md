# Serverless Deployment Guide

This guide covers deploying TEAS-Bench using ServerlessLLM on Kubernetes for production-grade, auto-scaling inference.

> **Note:** Currently tested on Kubernetes clusters with NVIDIA GPU support.

## Prerequisites

- Kubernetes cluster with GPU nodes
- `kubectl` configured to access your cluster
- PersistentVolumeClaim named `model-pvc` for model storage
- Docker installed (for building custom benchmark containers)

## Architecture Overview

The serverless deployment consists of:
- **SLLM Head Node** - Manages model routing and scaling decisions
- **Pylet Head Node** - Coordinates worker scheduling
- **GPU Workers** - Run model inference

## Step-by-Step Deployment

### Step 1: Build Your Docker Container

Create a Docker container with your testing scripts and ServerlessLLM installed.

Your container must have ServerlessLLM installed:

```bash
pip install serverlessllm
```

**Supported Backends:**
| Backend | Description |
|---------|-------------|
| `vllm` | Standard vLLM backend |
| `sglang` | Standard SGLang backend |
| `moe-cap-sglang` | MoE-CAP optimized SGLang backend |
| `moe-cap-vllm` | MoE-CAP optimized vLLM backend |

### Step 2: Create Model Deployment Config

Add a deployment configuration JSON file in `serverless_config_examples/`. This defines how your model will be deployed.

**Example:** [`serverless_config_examples/sglang-moe-qwen3-30b-1worker-4gpu.json`](../serverless_config_examples/sglang-moe-qwen3-30b-1worker-4gpu.json)

```json
{
    "model": "Qwen/Qwen3-30B-A3B",
    "backend": "moe-cap-sglang",
    "num_gpus": 4,
    "auto_scaling_config": {
        "metric": "concurrency",
        "target": 10,
        "min_instances": 0,
        "max_instances": 1,
        "keep_alive": 60
    },
    "backend_config": {
        "pretrained_model_name_or_path": "Qwen/Qwen3-30B-A3B",
        "tensor_parallel_size": 4,
        "torch_dtype": "bfloat16"
    }
}
```

**Config Fields:**
| Field | Description |
|-------|-------------|
| `model` | HuggingFace model name |
| `backend` | One of: `vllm`, `sglang`, `moe-cap-sglang`, `moe-cap-vllm` |
| `num_gpus` | Number of GPUs per model instance |
| `tensor_parallel_size` | Tensor parallelism degree |
| `max_instances` | Maximum number of model instances that can run concurrently |
| `keep_alive` | Time in seconds the instance stays alive before being deleted after idle |

**Note on `max_instances`:** This depends on your deployment's total GPU capacity. For example, if you have 4 workers × 2 GPUs each (8 GPUs total) and each model instance requires 2 GPUs, you can set `max_instances = 4`.

### Step 3: Write K8s Benchmark Job YAML

Create a Kubernetes Job YAML that runs your benchmark.

See examples in each benchmark's `serverless-scripts/` folder:
- [`MoE-Benchmark/serverless-scripts/sglang-qwen3-30b-moe-gsm8k.yaml`](../MoE-Benchmark/serverless-scripts/sglang-qwen3-30b-moe-gsm8k.yaml)

### Step 4: Deploy the ServerlessLLM Infrastructure

Choose a deployment configuration based on your cluster resources:

**Available Deployments:**

| Configuration | File |
|---------------|------|
| 1 Worker × 2 A100 40GB | `sllm-deployment-1-worker-2-A100-40GB.yaml` |
| 1 Worker × 2 A100 80GB | `sllm-deployment-1-worker-2-A100-80GB.yaml` |
| 1 Worker × 2 H100 | `sllm-deployment-1-worker-2-H100.yaml` |
| 1 Worker × 2 H200 | `sllm-deployment-1-worker-2-H200.yaml` |
| 1 Worker × 4 H100 | `sllm-deployment-1-worker-4-H100.yaml` |
| 2 Workers × 4 H100 | `sllm-deployment-2-worker-4-H100.yaml` |
| 4 Workers × 2 A100 40GB | `sllm-deployment-4-worker-2-A100-40GB.yaml` |
| 4 Workers × 2 H100 | `sllm-deployment-4-worker-2-H100.yaml` |
| 5 Workers × 2 H200 | `sllm-deployment-5-worker-2-H200.yaml` |

**Deploy Example (1 Worker with 4 H100 GPUs):**

```bash
kubectl apply -f serverless_deployments/sllm-deployment-1-worker-4-H100.yaml
```

Wait for all pods to be ready:

```bash
kubectl get pods -w
```

You should see:
- `pylet-head-xxx` - Running
- `sllm-head-xxx` - Running  
- `pylet-worker-0-xxx` - Running

### Step 5: Run the Benchmark

Once all deployments are ready, run the benchmark job:

```bash
kubectl create -f MoE-Benchmark/serverless-scripts/sglang-qwen3-30b-moe-gsm8k.yaml
```

The benchmark job will:
1. Wait for the head nodes to be ready
2. Deploy the Qwen3-30B-A3B MoE model
3. Run evaluation on the GSM8K dataset
4. Save results to `/models/moecap_results`

### Step 6: Check Results

Monitor the benchmark progress:

```bash
kubectl logs -f job/sllm-benchmark-xxxxx
```

Results are saved to the PVC at `/models/moecap_results`.

## Cleanup

To remove the deployment:

```bash
# Remove the infrastructure
kubectl delete -f serverless_deployments/sllm-deployment-1-worker-4-H100.yaml

# Remove benchmark jobs
kubectl delete job -l app=sllm-benchmark
```

## Troubleshooting

### Pods stuck in Pending state
Check if your cluster has sufficient GPU resources:
```bash
kubectl describe pod <pod-name>
```

### Model loading issues
Ensure the model is available on the PVC:
```bash
kubectl exec -it <worker-pod> -- ls /models
```

### Head node not ready
Check the logs for errors:
```bash
kubectl logs sllm-head-xxx
kubectl logs pylet-head-xxx
```
