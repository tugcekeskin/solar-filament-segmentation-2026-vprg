# SAM ViT-B base weights

These files are the unmodified weights and configuration of the Hugging Face model
[`facebook/sam-vit-base`](https://huggingface.co/facebook/sam-vit-base) (Segment Anything, Meta AI),
re-saved as PyTorch state-dict shards under 100 MB so that the pipeline runs without downloads.

They are distributed under the Apache License 2.0 (https://www.apache.org/licenses/LICENSE-2.0).
Kirillov et al., "Segment Anything", ICCV 2023.

Our trained parameters (LoRA adapters, prompt encoder, mask decoder) are in `weights/sam_lora_refiner.pt`.
