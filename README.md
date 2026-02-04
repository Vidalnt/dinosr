
# [DinoSR: Self-Distillation and Online Clustering for Self-supervised Speech Representation Learning](https://arxiv.org/pdf/2305.10005.pdf)

We have integrated the **Adversarial Style Augmentation (ASA)** module from the **[SAVC](https://arxiv.org/abs/2405.00603)** framework into DinoSR. 

**Why?** 
Similar to **ContentVec**, the goal is to disentangle speaker identity from linguistic content. However, instead of using an external Voice Conversion system to preprocess the dataset, we use ASA to apply dynamic statistical perturbations to the feature space during training. This forces the model to ignore speaker style (timbre/channel) and focus purely on phonetic content, achieving robust speaker disentanglement in an end-to-end manner.

### Setup

- Codebase preparation (based on [`fairseq`](https://github.com/facebookresearch/fairseq))
```
# we use fairseq to build the model
git clone https://github.com/One-sixth/fairseq
cd fairseq
pip install --editable ./

# plug in DinoSR
cd examples
git clone https://github.com/Vidalnt/dinosr.git
```

- Data preparation:
please follow [`instruction provided by wav2vec2`](https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec) for pre-training/fine-tuning data preprocessing


### Usage

- Training

    For the list of hyper-parameters, see [`config file`](config/base.yaml) and also [`model attributes`](models/dinosr.py) where default settings used in the paper are provided. 

```
# minimal example to reproduce model
python fairseq_cli/hydra_train.py -m \
    --config-dir examples/dinosr/config/ \
    --config-name base \
    task.data=/path/to/prepared/librispeech/ \
    common.user_dir=examples/dinosr &
```

- Loading pre-trained model as python object

```
import fairseq
import argparse
code_path = "examples/dinosr"
fairseq.utils.import_user_module(argparse.Namespace(user_dir=code_path))
ckpt_path = "/path/to/the/checkpoint.pt"
models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
model = models[0]
```

- Fine-tuning pre-trained checkpoint as ASR

```
# minimal example for fine-tuning with 100hr data
python fairseq_cli/hydra_train.py -m \
        --config-dir examples/wav2vec/config/finetuning \
        --config-name base_100h \
        common.user_dir=examples/dinosr \
        task.data=/path/to/labeled/librispeech/ \
        model.w2v_path=/path/to/dinosr.ckpt \
        task.normalize=True
```

- Fine-tuning with Adversarial Style Augmentation (ASA)

```
python fairseq_cli/hydra_train.py -m \
  --config-dir examples/dinosr/config/ \
  --config-name finetune_asa \
  +task.data=/path/to/dataset/manifests \
  +common.user_dir=examples/dinosr \
  ++checkpoint.restore_file=/path/to/dinosr_base.ckpt \
  ++checkpoint.reset_optimizer=true \
  ++checkpoint.reset_lr_scheduler=true
```
### Pre-trained checkpoint

Pre-trained checkpoint without fine-tuning can be downloaded [here](https://data.csail.mit.edu/placesaudio/dinosr/dinosr.ckpt).

A model fine-tuned using the ASA module for enhanced speaker disentanglement is available [here](https://huggingface.co/vidalnt/DinoSR-Savc/tree/main).
