import logging
import math
import os
import random
import re
import sys
import time
from argparse import Namespace
from collections import namedtuple

import ast
import numpy as np
import sentencepiece as spm
import torch
from fairseq_cli.generate import get_symbols_to_strip_from_output
from PIL import Image
from torchvision import transforms

from fairseq import checkpoint_utils, distributed_utils, options, tasks, utils
from fairseq.dataclass.configs import FairseqConfig
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.token_generation_constraints import pack_constraints, unpack_constraints
from omegaconf import OmegaConf

from kosmos_2.unilm.tasks import *
from kosmos_2.unilm.models.unigpt import *

import matplotlib.pyplot as plt

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.interactive")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Batch = namedtuple("Batch", "ids src_tokens src_lengths constraints img_src_tokens img_gpt_input_mask img_path_batch")
Translation = namedtuple("Translation", "src_str hypos pos_scores alignments")

def square_transform(size=224):
    inception_normalize = transforms.Compose(
        [transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])]
    )
    return transforms.Compose(
        [
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            inception_normalize,
        ]
    )

def split_string(string, separators):
    """
    Function to split a given string based on a list of separators.

    Args:
    string (str): The input string to be split.
    separators (list): A list of separators to be used for splitting the string.

    Returns:
    A list containing the split string with separators included.
    """
    pattern = "|".join(re.escape(separator) for separator in separators) 
    result = re.split(f'({pattern})', string)  
    return [elem for elem in result if elem] 

def get_interactive_tokens_and_lengths(self, lines, encode_fn, tokenizer=None):
    """
    line format: [image]path<tab>text<tab>[image]path
    model input: `<s> <image> image hidden </image> My cat looking very dignified.</s>`
    """
    image_feature_length = self.args.image_feature_length
    bos_id = self.dictionary.bos()
    eos_id = self.dictionary.eos()
    boi_id = self.dictionary.index("<image>")
    eoi_id = self.dictionary.index("</image>")
    
    def convert_one_line(input_str):
        # TODO: input interleave image and text
        token = []
        img_src_token = []
        img_gpt_input_mask = []
        segments = input_str.split('<tab>')
        token.append(bos_id)
        img_gpt_input_mask.append(0)
        for i, segment in enumerate(segments):
            if segment.startswith('[image]'):
                image_path = segment[7:]
                # read image and transform to tensor
                image = Image.open(image_path).convert("RGB")
                # update the global_path
                # global global_image_path
                # global_image_path = image_path
                image_tensor = square_transform(self.args.input_resolution)(image)
                img_src_token.append(image_tensor)
                # global global_image_tensor
                # global_image_tensor = image_tensor
                token.extend([boi_id] + list(range(4, image_feature_length+4)) + [eoi_id])
                
                img_gpt_input_mask.extend([0] + [1] * image_feature_length + [0])
            else:
                special_tokens = [self.source_dictionary[idx] for idx in range(tokenizer.vocab_size(), 
                                                                               len(self.source_dictionary))]
                split_special_token_words = []
                split_resutls = split_string(segment, special_tokens)
                for string in split_resutls:
                    if string in special_tokens:
                        # print(f"dict-length({len(self.source_dictionary)}), substring {string} is a special token")
                        split_special_token_words.append(string)
                    else:
                        encode_tokens = tokenizer.encode(string, out_type=str)
                        # print(f"dict-length({len(self.source_dictionary)}), substring {string} is not a special token, tokenized into {encode_tokens}")
                        split_special_token_words.extend(encode_tokens)
                segment = ' '.join(split_special_token_words)
                
                text_tokens = self.source_dictionary.encode_line(
                    encode_fn(segment), add_if_not_exist=False
                ).tolist()
                
                text_tokens = text_tokens[:-1] # </s> in token
                token.extend(text_tokens)
                img_gpt_input_mask.extend([0] * (len(text_tokens))) # </s> in token
        token.append(eos_id)
        # img_gpt_input_mask = img_gpt_input_mask[:-1]
        assert len(token) == len(img_gpt_input_mask) + 1 
        token = torch.LongTensor(token)
        img_gpt_input_mask = torch.LongTensor(img_gpt_input_mask)
        img_src_token = torch.stack(img_src_token, dim=0)
        return token, img_src_token, img_gpt_input_mask
    
    tokens = []
    img_src_tokens = []
    img_gpt_input_masks = []
    for src_str in lines:
        token, img_src_token, img_gpt_input_mask = convert_one_line(src_str)
        tokens.append(token)
        img_src_tokens.append(img_src_token)
        img_gpt_input_masks.append(img_gpt_input_mask)
    lengths = [t.numel() for t in tokens]
    
    return tokens, lengths, img_src_tokens, img_gpt_input_masks


def make_batches(lines, cfg, task, max_positions, encode_fn):
    def encode_fn_target(x):
        return encode_fn(x)

    if cfg.generation.constraints:
        # Strip (tab-delimited) contraints, if present, from input lines,
        # store them in batch_constraints
        batch_constraints = [list() for _ in lines]
        for i, line in enumerate(lines):
            if "\t" in line:
                lines[i], *batch_constraints[i] = line.split("\t")

        # Convert each List[str] to List[Tensor]
        for i, constraint_list in enumerate(batch_constraints):
            batch_constraints[i] = [
                task.target_dictionary.encode_line(
                    encode_fn_target(constraint),
                    append_eos=False,
                    add_if_not_exist=False,
                )
                for constraint in constraint_list
            ]

    if cfg.generation.constraints:
        constraints_tensor = pack_constraints(batch_constraints)
    else:
        constraints_tensor = None

    tokenizer = spm.SentencePieceProcessor()
    if os.path.exists('data/sentencepiece.bpe.model'):
        tokenizer.Load('data/sentencepiece.bpe.model')
    else:
        tokenizer = None
    tokens, lengths, img_src_tokens, img_gpt_input_mask = get_interactive_tokens_and_lengths(task, lines, encode_fn, tokenizer)

    itr = task.get_batch_iterator(
        dataset=task.build_dataset_for_caption_inference(
            tokens, lengths, img_src_tokens, img_gpt_input_mask, constraints=constraints_tensor
        ),
        max_tokens=cfg.dataset.max_tokens,
        max_sentences=cfg.dataset.batch_size,
        max_positions=max_positions,
        ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
    ).next_epoch_itr(shuffle=False)
    for batch in itr:
        ids = batch["id"]
        src_tokens = batch["net_input"]["src_tokens"]
        src_lengths = batch["net_input"]["src_lengths"]
        img_src_tokens = batch["net_input"]["img_src_tokens"]
        img_gpt_input_mask = batch["net_input"]["img_gpt_input_mask"]
        constraints = batch.get("constraints", None)

        yield Batch(
            ids=ids,
            src_tokens=src_tokens,
            src_lengths=src_lengths,
            img_src_tokens=img_src_tokens,
            img_gpt_input_mask=img_gpt_input_mask,
            constraints=constraints,
        )

def main(cfg: FairseqConfig):
    if isinstance(cfg, Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)
    logger.info(cfg)

    utils.import_user_module(cfg.common)

    cfg.interactive.buffer_size = 1
    if cfg.dataset.max_tokens is None and cfg.dataset.batch_size is None:
        cfg.dataset.batch_size = 1

    assert (
        not cfg.generation.sampling or cfg.generation.nbest == cfg.generation.beam
    ), "--sampling requires --nbest to be equal to --beam"
    assert (
        not cfg.dataset.batch_size
        or cfg.dataset.batch_size <= cfg.interactive.buffer_size
    ), "--batch-size cannot be larger than --buffer-size"

    # Fix seed for stochastic decoding
    if cfg.common.seed is not None and not cfg.generation.no_seed_provided:
        np.random.seed(cfg.common.seed)
        utils.set_torch_seed(cfg.common.seed)

    use_cuda = torch.cuda.is_available() and not cfg.common.cpu

    # Setup task, e.g., translation
    logger.info("Task: {}".format(cfg.task))
    task = tasks.setup_task(cfg.task)

    # Load ensemble
    overrides = ast.literal_eval(cfg.common_eval.model_overrides)
    logger.info("loading model(s) from {}".format(cfg.common_eval.path))
    models, _model_args = checkpoint_utils.load_model_ensemble(
        utils.split_paths(cfg.common_eval.path),
        arg_overrides=overrides,
        task=task,
        suffix=cfg.checkpoint.checkpoint_suffix,
        strict=(cfg.checkpoint.checkpoint_shard_count == 1),
        num_shards=cfg.checkpoint.checkpoint_shard_count,
    )

    # Set dictionaries
    src_dict = task.source_dictionary
    tgt_dict = task.target_dictionary

    logger.info(cfg.distributed_training)

    # Optimize ensemble for generation
    for model in models:
        if model is None:
            continue
        if cfg.common.fp16:
            model.half()
        if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
            model.cuda()
        model.prepare_for_inference_(cfg)

    # Initialize generator
    generator = task.build_generator(models, cfg.generation)

    # Handle tokenization and BPE
    tokenizer = task.build_tokenizer(cfg.tokenizer)
    bpe = task.build_bpe(cfg.bpe)


def cli_main():
    parser = options.get_interactive_generation_parser()
    args = options.parse_args_and_arch(parser)
    main(convert_namespace_to_omegaconf(args))
    # distributed_utils.call_main(convert_namespace_to_omegaconf(args), main)

if __name__ == "__main__":
    cli_main()
