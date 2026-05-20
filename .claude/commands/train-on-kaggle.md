# /train-on-kaggle — Generate Kaggle Training Notebook

When the user runs `/train-on-kaggle`, do the following:

1. Ask: policy name and env name (default: act + CubeReachV1)
2. Verify `configs/training/kaggle.yaml` exists and has checkpoint resume config
3. Generate a Kaggle-ready version of `notebooks/02_train_kaggle.ipynb` with:
   - Hardware check cell
   - Install cell (uv + deps)
   - Secret loading cell (HF_TOKEN, WANDB_API_KEY)
   - Checkpoint detection cell
   - Training cell: `!python train.py training=kaggle policy=<name> env=<env> run_name=<run>`
   - Hub push cell (at end, private=False for benchmark models)
4. Print instructions for attaching the repo as Kaggle dataset
