# Evaluation Instructions

This folder contains evaluation examples and configurations for different experimental setups. The main evaluation script `example_evaluation.sh` demonstrates the standard evaluation configuration, while various experimental configurations can be implemented by modifying specific parameters and adding additional evaluation arguments.

## Detailed Explanation of the Evaluation Script

In this section, we will provide detailed explanations of the main configuration components in the standard evaluation script.

### Evaluation Settings

The evaluation script is fundamentally similar to the training script, with two key modifications in the `OUTPUT_ARGS` section:

1. **Skip Training**: Add `--skip-train` to bypass the training loop and focus solely on evaluation.
2. **Full Validation Evaluation**: Set `--eval-iters` to 100 to ensure comprehensive coverage of the entire validation set, providing more reliable evaluation metrics.
3. **Evaluation Interval**: Set `--eval-interval` to 1 for direct evaluation.

### Monte Carlo Sampling

For MDM (Masked Diffusion Models), we implement Monte Carlo estimation to compute the expected loss over the validation set. This is achieved by setting `--num-mc > 1` to enable Monte Carlo sampling.

**Implementation Details:**
- **Sampling Strategy**: The codebase employs dataset replication to construct Monte Carlo samples.
- **Maximum Support**: The system supports up to 256 Monte Carlo samples (`num-mc ≤ 256`).
- **Research Configuration**: In our paper experiments, we utilized `num-mc = 32` for optimal balance between computational efficiency and statistical reliability.

**Technical Note**: The Monte Carlo sampling approach provides more robust loss estimates by averaging over multiple stochastic forward passes, which is important for diffusion models where the generation process involves inherent randomness. We didn't apply this during training because it would significantly increase the training time.