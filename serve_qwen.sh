export CUDA_HOME=/usr/local/cuda/

uvx --python 3.10 vllm@0.14.0 serve Qwen/Qwen3-4B-Thinking-2507 --api-key ryan123 --reasoning-parser deepseek_r1 --trust-remote-code --tokenizer Qwen/Qwen3-4B-Thinking-2507

