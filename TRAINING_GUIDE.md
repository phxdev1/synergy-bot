# Training Guide - Corporate Synergy Bot 7B

Due to issues with AutoTrain UI, here are the working alternatives:

## Option 1: Google Colab (Recommended) 🚀

1. Open [Google Colab](https://colab.research.google.com/)
2. Create a new notebook
3. Change runtime to GPU (Runtime → Change runtime type → T4 GPU)
4. Copy the cells from `training/train_colab_notebook.py` into your notebook
5. Run each cell in order
6. Training takes ~2-3 hours on T4 GPU

## Option 2: AutoTrain CLI (Local/Cloud)

```bash
cd training
./train_simple_autotrain.sh
```

Or manually:
```bash
pip install autotrain-advanced
huggingface-cli login
autotrain llm --train --model mistralai/Mistral-7B-Instruct-v0.2 --data-path phxdev/corporate-speak-dataset-autotrain --text-column text --batch-size 4 --epochs 3 --lr 2e-4 --use-peft --lora-r 16 --push-to-hub --hub-model-id phxdev/corporate-synergy-bot-7b
```

## Option 3: Direct Python Script

```bash
cd training
pip install transformers datasets peft accelerate bitsandbytes tensorboard
python train_direct.py
```

## Option 4: Upload CSV to AutoTrain

1. Go to [AutoTrain UI](https://ui.autotrain.huggingface.co/)
2. Create new project
3. Upload `training/corporate_speak_train.csv`
4. Select column: `text`
5. Configure LoRA settings (r=16, alpha=32)

## Datasets Available

- **Original**: `phxdev/corporate-speak-dataset`
- **AutoTrain Fixed**: `phxdev/corporate-speak-dataset-autotrain`
- **Local CSV**: `training/corporate_speak_train.csv`

## After Training

Your model will be available at: https://huggingface.co/phxdev/corporate-synergy-bot-7b

To create a demo space:
1. Go to https://huggingface.co/new-space
2. Upload `demo/app.py` and `demo/requirements_space.txt`
3. Your bot will be live!

## Troubleshooting

- **AutoTrain UI 400 Error**: Use CLI or Colab instead
- **Out of Memory**: Reduce batch size or use gradient accumulation
- **CUDA Error**: Ensure you have GPU runtime selected in Colab