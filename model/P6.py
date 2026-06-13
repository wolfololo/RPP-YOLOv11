from ultralytics.nn.modules import Detect

def get_p6_head_config():
    backbone = [
        [-1, 1, "Conv", [64, 3, 2]],
        [-1, 1, "Conv", [128, 3, 2]],
        [-1, 2, "C3k2", [256, False, 0.25]],
        [-1, 1, "Conv", [256, 3, 2]],
        [-1, 2, "C3k2", [512, False, 0.25]],
        [-1, 1, "Conv", [512, 3, 2]],
        [-1, 2, "C3k2", [512, True]],
        [-1, 1, "Conv", [768, 3, 2]],
        [-1, 2, "C3k2", [768, True]],
        [-1, 1, "Conv", [1024, 3, 2]],
        [-1, 2, "C3k2", [1024, True]],
        [-1, 1, "SPPF", [1024, 5]],
        [-1, 2, "C2PSA", [1024]],
    ]

    head = [
        [-1, 1, "nn.Upsample", [None, 2, "nearest"]],
        [[-1, 8], 1, "Concat", [1]],
        [-1, 2, "C3k2", [768, False]],

        [-1, 1, "nn.Upsample", [None, 2, "nearest"]],
        [[-1, 6], 1, "Concat", [1]],
        [-1, 2, "C3k2", [512, False]],

        [-1, 1, "nn.Upsample", [None, 2, "nearest"]],
        [[-1, 4], 1, "Concat", [1]],
        [-1, 2, "C3k2", [256, False]],

        [-1, 1, "Conv", [256, 3, 2]],
        [[-1, 18], 1, "Concat", [1]],
        [-1, 2, "C3k2", [512, False]],

        [-1, 1, "Conv", [512, 3, 2]],
        [[-1, 15], 1, "Concat", [1]],
        [-1, 2, "C3k2", [768, True]],

        [-1, 1, "Conv", [512, 3, 2]],
        [[-1, 12], 1, "Concat", [1]],
        [-1, 2, "C3k2", [1024, True]],

        [[21, 24, 27, 30], 1, "Detect", ["nc"]],
    ]

    return backbone, head

def get_p6_psconv_head_config():
    backbone = [
        [-1, 1, "PSConv", [64, 3, 2]],
        [-1, 1, "PSConv", [128, 3, 2]],
        [-1, 2, "C3k2", [256, False, 0.25]],
        [-1, 1, "PSConv", [256, 3, 2]],
        [-1, 2, "C3k2", [512, False, 0.25]],
        [-1, 1, "PSConv", [512, 3, 2]],
        [-1, 2, "C3k2", [512, True]],
        [-1, 1, "PSConv", [1024, 3, 2]],
        [-1, 2, "C3k2", [1024, True]],
        [-1, 1, "PSConv", [1024, 3, 2]],
        [-1, 2, "C3k2", [1024, True]],
        [-1, 1, "SPPF", [1024, 5]],
        [-1, 2, "C2PSA", [1024]],
    ]

    head = [
        [-1, 1, "nn.Upsample", [None, 2, "nearest"]],
        [[-1, 6], 1, "Concat", [1]],
        [-1, 2, "C3k2", [512, False]],

        [-1, 1, "nn.Upsample", [None, 2, "nearest"]],
        [[-1, 4], 1, "Concat", [1]],
        [-1, 2, "C3k2", [256, False]],

        [-1, 1, "PSConv", [256, 3, 2]],
        [[-1, 15], 1, "Concat", [1]],
        [-1, 2, "C3k2", [512, False]],

        [-1, 1, "PSConv", [512, 3, 2]],
        [[-1, 10], 1, "Concat", [1]],
        [-1, 2, "C3k2", [1024, True]],

        [-1, 1, "PSConv", [512, 3, 2]],
        [[-1, 12], 1, "Concat", [1]],
        [-1, 2, "C3k2", [1024, True]],

        [[18, 21, 24, 27], 1, "Detect", ["nc"]],
    ]

    return backbone, head
