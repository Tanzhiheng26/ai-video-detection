## ResNet 3D
Code adapted from https://www.tensorflow.org/tutorials/video/video_classification

## VideoMAE
Code adapted from https://huggingface.co/docs/transformers/en/tasks/video_classification

## Dataset 
The dataset should be organized like so:
```
data/
    train/
        0_real/
            video_1.mp4
            video_2.mp4
            ...
        1_fake/
            video_1.mp4
            video_2.mp4
            ...
    val/
        0_real/
            video_1.mp4
            video_2.mp4
            ...
        1_fake/
            video_1.mp4
            video_2.mp4
            ...
    test/
        0_real/
            video_1.mp4
            video_2.mp4
            ...
        1_fake/
            video_1.mp4
            video_2.mp4
            ...
```