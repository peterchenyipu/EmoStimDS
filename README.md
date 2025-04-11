# EmoStim DS for NVILA Training

## Install

1. Download the EmoStim dataset from [(1): Mohammadi, G.; Vuilleumier, P.; Somarathna, R. EmoStim Dataset, 2023](https://doi.org/10.26037/yareta:usqhnbgauvgx3ggf3bqbrwtjsa).

2. Unzip and put it in the `EmoStimFiles` dir.

3. Download the videos and put it in the `videos` dir

4. Run `convert_to_vila_ds.py` to create the dataset.

For inference on custom videos, put the videos in some directory and specify with variable `INFERENCE_VIDEOS_DIR` in `prepare_for_inference.py` file. Then run `prepare_for_inference.py` to create the dataset.
