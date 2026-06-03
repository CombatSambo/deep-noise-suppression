# Project Report

## 1. Subject Description

This project implements a neural-network-based speech enhancement system in the style of the Microsoft DNS Challenge. The main task is deep noise suppression: given a noisy speech waveform, the model estimates a spectral mask that suppresses background noise while preserving the speech signal.

The input data are audio waveforms, primarily `.wav` files and optionally `.flac` files. During training, the data pipeline loads clean speech and noise recordings, synthesizes noisy mixtures, and returns PyTorch tensors shaped as `[B, 1, T]`. Before the model receives them, waveforms are converted into STFT magnitude features. The model input has shape `[T_frames, B, 322]`, composed of 161 microphone magnitude features and 161 far-end placeholder features. In the current setup, no true far-end signal is used, so the far-end branch is filled with a neutral constant.

The expected model output is a mask with shape `[T_frames, B, 161]`. During training, this mask is compared against an ideal target mask. During inference, the predicted mask is applied to the noisy STFT magnitude, combined with the noisy phase, reconstructed with inverse FFT and overlap-add, and saved as an enhanced `.wav` file.

This task is useful because noise suppression improves speech quality and intelligibility in communication, conferencing, hearing-assistance, and audio preprocessing systems. The implementation assumes a target sample rate of 16 kHz, single-channel speech enhancement, and a mask-based frequency-domain enhancement strategy.

## 2. Dataset

The dataset is partially synthetic. The project uses real clean speech and real noise files as source material, but training pairs are synthesized on the fly. The clean speech sources are configured in `synthesizer_config.yaml` under `dataset/clean_speech/emotional_speech`, `dataset/clean_speech/read_speech`, and `dataset/clean_speech/VocalSet_48kHz_mono`. The noise source is configured under `dataset/noise`.

The main dataset class is `SpeechNoiseDataset` in `data_pipeline.py`. For each training example, it randomly selects one clean speech file and one noise file from weighted audio pools. It then loads the files, converts multi-channel audio to mono, optionally resamples to the configured sample rate, crops or pads to a fixed duration, applies random gain, scales the noise to a sampled SNR, and mixes clean speech with scaled noise.

The training sample returned by the dataset contains:

- `clean`: clean speech tensor, shape `[1, T]`
- `noisy`: synthesized noisy speech tensor, shape `[1, T]`
- `noise`: scaled noise tensor, shape `[1, T]`
- `snr_db`: sampled SNR value
- `sample_rate`, file paths, and dataset metadata

The target label is not a waveform. It is an ideal spectral mask computed in `train_baseline_pytorch.py` as `clamp(clean_mag / noisy_mag, 0, 1)`. The model learns to predict this mask from noisy STFT features.

Preprocessing includes mono conversion, resampling, random crop/pad, random gain in the range `[-6, 6]` dB, SNR mixing in the interval `[-5, 20]` dB, optional normalization to `-27` dB, and clipping protection. The default training segment length in `synthesizer_config.yaml` is 1.0 second, while `train3.slurm` uses 10.0-second segments for the encoder-decoder experiment.

Audio examples are sampled randomly during training, but the train/validation/test separation follows a fixed file-level split when `--enable-file-split` is passed to `train_baseline_pytorch.py`. In that mode, each source audio file is assigned deterministically to exactly one split using a path hash and split seed. Random clean/noise selection then happens only inside the selected split, so training files are not reused in validation or testing. The configured/default split ratios are 80% training, 10% validation, and 10% testing. Without `--enable-file-split`, train and validation datasets are sampled from the same source pools, so file-level separation is not guaranteed.

For evaluation data generation, `generate_eval_set.py` uses `Synthesizer` to create synthetic evaluation samples and writes noisy, target, and nearend waveforms into separate directories. A default of 50 evaluation samples is specified in that script.

## 3. Model Architecture

The project currently defines two PyTorch model variants in `models/baseline_pytorch.py`: `BaselineGRUMaskNet` and `EncoderDecoderMaskNet`.

`BaselineGRUMaskNet` is the recurrent baseline model based on the topology of the provided ONNX baseline. It uses two one-layer GRU modules and one linear projection:

- GRU 1: input size 322, hidden size 322
- Residual connection: input features are added to the first GRU output
- GRU 2: input size 322, hidden size 322
- Linear projection: 322 to 161
- Sigmoid activation and clamp to produce a non-negative mask

The model input is `[seq_len, batch_size, 322]`, and the output mask is `[seq_len, batch_size, 161]`. It also returns the hidden states of both GRU layers. This model is closest to the ONNX baseline structure. Exact ONNX weight mapping and exact GRU operator parity are marked as TODO items in the code.

`EncoderDecoderMaskNet` is a recurrent encoder-decoder model added as an architectural improvement attempt. It keeps the same input/output interface but changes the recurrent structure:

- Encoder GRU: input size 322, hidden size 256
- Decoder GRU: input size 256, hidden size 322
- Linear projection: 322 to 161
- Sigmoid activation and clamp to produce the predicted mask

The encoder-decoder variant was introduced to address the teacher recommendation to use an encoder-decoder style architecture while still preserving temporal modeling through GRU layers. Compared with the baseline GRU model, the encoder-decoder explicitly compresses features into a lower-dimensional recurrent representation and then decodes them back to the mask-estimation space. Both models remain mask-based enhancement models and are compatible with the same STFT feature pipeline.

No convolutional layers, attention layers, embeddings, dropout, or batch/layer normalization are explicitly used in the current model definitions.

## 4. Training Setup of Different Attempts

All training attempts use the same general training script, `train_baseline_pytorch.py`. The loss function is mean squared error between the predicted mask and the target ideal mask. The optimizer is Adam. The model is trained on STFT-derived features, and validation uses the same loss computation without gradient updates.

The training script tracks:

- `train_loss`
- `val_loss`
- epoch wall-clock time
- estimated time remaining
- optional final `test_loss` when strict file splitting is enabled

Gradient clipping is used through `torch.nn.utils.clip_grad_norm_`, with a default and Slurm value of `5.0`. No learning-rate scheduler, early stopping, dropout regularization, or explicit weight decay is specified in the project files.

The local/default training configuration in `train_baseline_pytorch.py` is:

- model type: `gru` by default
- epochs: 5
- batch size: 8
- epoch size: 2000 synthetic examples
- validation size: `epoch_size // 10` if not explicitly specified
- segment length: 1.0 second
- learning rate: `1e-3`
- device: CPU unless another device is passed
- gradient clipping: `5.0`

The first HPC training attempt is specified in `train.slurm`. It trains the GRU baseline on a Tesla GPU with:

- model type: GRU baseline by default
- epochs: 10
- batch size: 32
- epoch size: 50000
- validation epoch size: 5000
- segment length: 1.0 second
- learning rate: `5e-4`
- gradient clipping: `5.0`
- device: CUDA
- allocation time: 8 hours
- checkpoint directory: `checkpoints_v100_10ep`

The encoder-decoder HPC attempt is specified in `train3.slurm`. It trains the recurrent encoder-decoder model with strict file-level splitting:

- model type: `encoder_decoder`
- train/validation/test split: 80% / 10% / 10%
- split seed: 42
- epochs: 20
- batch size: 32
- epoch size: 80000
- validation epoch size: 8000
- test epoch size: 8000
- segment length: 10.0 seconds
- learning rate: `5e-4`
- gradient clipping: `5.0`
- device: CUDA
- allocation time: 12 hours
- checkpoint directory: `checkpoints_encdec_20ep_10s`

The available local evaluation files show a one-file comparison between a raw `nearend.wav` sample and an enhanced `nearend_enh.wav` sample. For this example, DNSMOS background score improved from `1.1042` to `2.4129`, DNSMOS overall improved from `1.0974` to `2.0221`, and SigMOS overall improved from `1.2434` to `1.6319`. These numbers are from local output CSV files and should be treated as a small smoke-test result, not a final benchmark.

Formal test-set results for the new recurrent encoder-decoder training attempt are not explicitly specified in the project files.
