# Wave_Field_Improved
This repo contains experimentation and improvisation of the novel architecture wave field llm created by Avinash Badaramoni 

## Goal
To engineer a new attention paradigm that matches the efficacy of the standard Transformer while maintaining sub-quadratic time complexity.

## Our Approach: Latent Wave Integration
Our architecture optimizes this process by focusing on Latent Projection and Global Context. We utilize a more efficient 512-dimensional field and introduce the following procedural shifts: <br>
**Latent Projection:** Transformed Keys (TK) are projected into a compact 66-dimension latent space.<br>
**Global Broadcasting:** Instead of splitting the vector into heads, the entire 66-dimension latent vector is copied across 6 heads [B, 6, 512, 66]. <br>
**Global Wave Application:** We apply 6 unique wave configurations to these identical copies. This ensures that every wave kernel processes the complete context of the latent vector rather than a fragment. <br>
**BMM & Synthesis:** Following BMM on each field, the heads are bilinearly gathered and projected back. We take the mean of these 6 heads and project the result back to the original embedding dimension [B, 100, 256]. <br>
**Gated Output:** The final output is integrated with a gated embedding vector, maintaining the necessary non-linearity for complex sequences.

## Experimental Results & Analysis

| Metric | Std Transformer | Wave_Field_Original | Wave_Field_Improved |
| :--- | :--- | :--- | :--- |
| **Test PPL** | 91.4 | 175.1 | 132.1 |
| **Test Accuracy** | 26.1% | 18.3% | 22.2% |
| **Parameters** | 6,852,864 | 7,775,378 | 6,604,394 |
| **Train Time/epoch** | 40s | 143s | 118s |
| **Complexity** | $O(n^2)$ | $O(n \log n)$ | $O(n \log n)$ |
| **Tokenizer** | BPE 8000 | BPE 8000 | BPE 8000 |
| **Seq Length** | 256 | 256 | 256 |

## Novelty & Competitive Advantage
**Sub-quadratic Efficiency:** By utilizing wave kernels for convolution over a field, we maintain O(n log n) time complexity, bypassing the limitations of traditional O(n^2) attention. <br>
**Contextual Preservation:** Applying convolution over the complete latent dimension prevents the information loss inherent in standard head-splitting techniques. Every "head" sees the whole picture. <br>
**Parameter Optimization:** Projecting the embedding vector into a latent dimension allows us to reduce the model size by approximately 1 million parameters with no degradation in performance—and, as shown in results, a significant gain in accuracy.


**Parameter Efficiency:** Reduced from 7.7M to 6.6M (lower than the Standard Transformer). <br>
**Operational Latency:** Training time per epoch improved from 143s to 118s. <br>
**Model Performance**: Perplexity (PPL) dropped from 175.1 to 132.3, while Test Accuracy rose from 18.3% to 22.4%. 

## Limitations
The primary limitation is that attention is currently static. While the model processes the full sequence, the wave kernels are fixed after training and do not adapt to specific input data in real-time.

## Future Works
**Dynamic Kernels**: Transition to input-dependent wave parameters so the data itself dictates the attention pattern. <br>
**Causal Global Context:** Implement a novel mechanism that ensures tokens maintain causal awareness of the full sequence, similar to masked cross-attention. <br>
**Optimization:** Conduct a grid search to finalize the optimal ratio of latent dimensions to field size for maximum efficiency.
