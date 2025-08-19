# This code is originally from https://github.com/bigscience-workshop/Megatron-DeepSpeed
# under the license https://huggingface.co/spaces/bigscience/license

from functools import reduce
from logging import logMultiprocessing
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir,os.path.pardir)))

from lm_eval.models.gpt2 import GPT2LM
from lm_eval import evaluator, tasks, utils
from lm_eval.base import CacheHook
from tqdm import tqdm
import torch.nn.functional as F

from lm_eval.tasks import ALL_TASKS
from pretrain_diff_gpt import model_provider
import numpy as np
import time

import torch
from megatron import get_args
from megatron import print_rank_0
from megatron import get_tokenizer
from megatron.core.enums import ModelType
from megatron.core import mpu
from megatron.training import setup_model_and_optimizer, get_model, _create_ds_config_dict
from megatron.core.tensor_parallel.mappings import gather_from_tensor_model_parallel_region

from megatron.utils import get_ltor_masks_and_position_ids, unwrap_model
from megatron.p2p_communication import recv_forward, send_forward
import pickle
import json

from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
from megatron.model.distributed import DistributedDataParallel as LocalDDP
from megatron.model.module import Float16Module
from deepspeed.runtime.pipe import schedule
from deepspeed.accelerator import get_accelerator

class EvalHarnessDiffAdaptor(GPT2LM):
    def __init__(self, model, tokenizer):
        args = get_args()
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.VOCAB_SIZE = tokenizer.vocab_size
        self.EOT_TOKEN_ID = tokenizer.eod
        self.NOISY_MASK_TOKEN_ID = args.padded_vocab_size if args.untie_embeddings_and_output_weights and args.untie_with_additional_mask else tokenizer.vocab_size

        self._max_length = args.seq_length
        self._num_mc = args.num_mc
        self._max_chunk_size = args.max_chunk_size
        self.sampling_eps = args.sampling_eps
        self.only_generate = args.only_generate
        self.only_mc_nll = args.only_mc_nll
        

        # For ds we split into mini batches and then micro batches to keep pipelining api happy.
        # With Megatron we just go to micro_batches directly
        self._batch_size = args.micro_batch_size

        self.is_main = args.rank == 0
        self.is_local_main = args.local_rank == 0
        self._device = get_accelerator().current_device_name()
        self.is_model_parallel = mpu.get_tensor_model_parallel_world_size() > 1
        self.is_pipe_parallel = mpu.get_pipeline_model_parallel_world_size() > 1
        self.is_data_parallel = mpu.get_data_parallel_world_size() > 1
        self.adaptive_seq_len = args.adaptive_seq_len
        if self.is_data_parallel and args.moe_expert_parallel_size == 1: # For MoE model, allow a "fake data parallel" in order to partition model into multiple gpus
            raise NotImplementedError("Data parallelism is currently not supported for evaluation")

        self.is_last_stage = True if not self.is_pipe_parallel else mpu.is_pipeline_last_stage()  # only the last stage of the pipeline model will receive the logits

        self.eval_method = args.eval_method
        self._setup_eval_method()

        self.model.eval()

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return self._device
    
    def _setup_eval_method(self):
        if self.eval_method == 'ar':
            self.eval_target = self._eval_target_nll_ar
        elif self.eval_method == 'mc':
            self.eval_target = self._eval_target_nll_mc

    def loglikelihood(self, requests):
        new_reqs = []
        for context, continuation in requests:
            if context == "":
                # end of text as context
                context_enc = [self.EOT_TOKEN_ID]
            else:
                context_enc = self.tokenizer_encode(context)

            continuation_enc = self.tokenizer_encode(continuation)

            new_reqs.append(((context, continuation), context_enc, continuation_enc))

        return self._loglikelihood_tokens(new_reqs)

    def loglikelihood_rolling(self, requests):
        # TODO: Implement caching once we've confirmed the perplexity implementation
        # TODO: automatic batch size detection for vectorization

        loglikelihoods = []
        with torch.no_grad():
            for string, in tqdm(requests):
                rolling_token_windows = list(map(utils.make_disjoint_window, utils.get_rolling_token_windows(
                    token_list=self.tokenizer_encode(string),
                    prefix_token=self.EOT_TOKEN_ID,
                    max_seq_len=self.max_length,
                    context_len=1,
                )))

                rolling_token_windows = [(None,) + x for x in rolling_token_windows]

                # TODO: extract out this call so it only gets called once and also somehow figure out partial caching for that
                string_nll = self._loglikelihood_tokens(rolling_token_windows, disable_tqdm=True)

                # discard is_greedy
                string_nll = [x[0] for x in string_nll]

                string_nll = sum(string_nll)
                loglikelihoods.append(string_nll)

        return loglikelihoods

    def _loglikelihood_tokens(self, requests, disable_tqdm=False):
        disable_tqdm = disable_tqdm if self.is_main else True
        res = []
        res_len = 0  # storing the result length for later
        self.model.eval()
        with torch.no_grad():
            def _collate(x):
                toks = x[1] + x[2]
                return (-len(toks), tuple(toks))

            reord = utils.Reorderer(requests, _collate)
            for chunk in utils.chunks(tqdm(reord.get_reordered(), disable=disable_tqdm), self.batch_size):
                inps, contlens, inplens, mask_positions, padding_length = [], [], [], [], None
                for cache_key, context_enc, continuation_enc in chunk: # text, context_enc(question), continuation_enc(answer)
                    # when too long to fit in context, truncate from the left
                    inp = torch.tensor(
                        (context_enc + continuation_enc)[-self.max_length:] # current token prediction
                        , dtype=torch.long).to(self.device)
                    inplen, = inp.shape

                    cont = continuation_enc

                    # since in _collate we make sure length is descending, the longest is always the first one.
                    padding_length = padding_length if padding_length is not None else inplen

                    if not self.adaptive_seq_len:
                        padding_length = self.max_length

                    if padding_length > inplen:
                        # pad to length
                        inp = torch.cat([
                            inp,  # [seq]
                            torch.full((padding_length - inplen,), self.NOISY_MASK_TOKEN_ID, dtype=torch.long).to(inp.device)  # [padding_length - seq]
                        ], dim=0)
                    else:
                        inp = inp[-padding_length:]
                        inplen = padding_length

                    if self.only_generate:
                        # add mask to inp
                        inp[inplen - len(cont): inplen + 1] = self.NOISY_MASK_TOKEN_ID
                        mask_pos = torch.arange(inplen - len(cont), inplen)
                        inps.append(inp.unsqueeze(0))
                        contlens.append(cont)
                        inplens.append(inplen)
                        mask_positions.append(mask_pos)
                    else:
                        # answer is a tuple of (logits target, bool if greedy prediction is correct)
                        answer = self.eval_target(inp, cont, inplen)
                        res.append(answer)
                        res_len += 1

                if self.only_generate:
                    input_ids = torch.cat(inps, dim=0)
                    batch_size = input_ids.shape[0]
                    max_step = max(len(cont) for cont in contlens)
                    for i in range(max_step):
                        logits = self._model_call(input_ids)
                        for j in range(batch_size):
                            mask_index = (input_ids[j] == self.NOISY_MASK_TOKEN_ID) & (torch.arange(input_ids.shape[1], device=self.device) < inplens[j])
                            if mask_index.sum() == 0:
                                continue
                            logits_cur = logits[j, mask_index]
                            x0 = torch.argmax(logits_cur, dim=-1)
                            p = torch.softmax(logits_cur.to(torch.float32), dim=-1)
                            confidence = torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)).squeeze(dim=-1)
                            _, index = torch.sort(confidence, descending=True)
                            x0[index[1:]] = self.NOISY_MASK_TOKEN_ID
                            input_ids[j, mask_index] = x0.clone()
                        del logits
                        torch.cuda.empty_cache()

                    for i in range(batch_size):
                        cont_toks = torch.tensor(contlens[i], dtype=torch.long, device=self.device)
                        pred_ans = input_ids[i, mask_positions[i]]
                        max_equal = (pred_ans == cont_toks).all().cpu()
                        res.append((float(0.0), bool(max_equal))) # ignore nll
                    res_len += len(chunk)

        if not mpu.is_pipeline_last_stage():
            # @HACK: To make the eval harness happy on threads that don't have access to the results.
            #        We just randomly generate some data.
            res = [(np.random.rand(), np.random.rand()>0.5) for _ in requests]

        return reord.get_original(res)

    def _eval_target_nll_ar(self, inp, cont_toks, inplen):
        self.model.eval()
        target_len = len(cont_toks)
        all_logits = []

        # Create batch of inputs by repeating and masking different positions
        # Use chunking to avoid OOM when target_len is too large
        max_batch_size = self.args.max_chunk_size  # Adjust based on available memory
        all_logits_list = []

        for chunk_start in range(0, target_len, max_batch_size):
            chunk_end = min(chunk_start + max_batch_size, target_len)
            chunk_size = chunk_end - chunk_start
            
            batch_inp = inp.unsqueeze(0).repeat(chunk_size, 1)
            mask_positions = torch.arange(chunk_start, chunk_end)
            mask_positions = inplen - len(cont_toks) + mask_positions

            for i in range(chunk_size):
                batch_inp[i, mask_positions[i]: mask_positions[-1] + 1] = self.NOISY_MASK_TOKEN_ID
                    
            with torch.no_grad():
                logits = self._model_call(batch_inp)
                
            if logits is not None:
                # Get logits for masked positions
                chunk_logits = logits[torch.arange(chunk_size), mask_positions, :]
                all_logits_list.append(chunk_logits)

        # Concatenate all chunks
        all_logits = torch.cat(all_logits_list, dim=0)
        all_logits = all_logits.unsqueeze(0)  # Add batch dimension to match original shape
        all_logits = F.log_softmax(all_logits, dim=-1)
        greedy_tokens = all_logits.argmax(dim=-1)
        cont_toks = torch.tensor(cont_toks, dtype=torch.long, device=self.device).unsqueeze(0)
        max_equal = (greedy_tokens == cont_toks).all().cpu()
        logits = torch.gather(all_logits, 2, cont_toks.unsqueeze(-1)).squeeze(-1)
        answer = (float(logits.sum()), bool(max_equal))
        return answer

    def _forward_process(self, batch_inp, chunk_start=0, u0=None):
        b = batch_inp.shape[0]
        l = batch_inp.shape[1]
        # sample from U[0, 1] following https://arxiv.org/pdf/2107.00630 I.1
        if u0 is None:
            u0 = torch.rand(1, device=self.device, dtype=torch.float32)
        indices = torch.arange(b, device=self.device).float() + chunk_start
        t = (u0 + indices / self._num_mc) % 1

        p_mask = (1 - self.sampling_eps) * t + self.sampling_eps
        p_mask = p_mask[:, None].repeat(1, l)

        mask_indices = torch.rand((b, l), device=self.device) < p_mask
        noisy_batch = torch.where(mask_indices, self.NOISY_MASK_TOKEN_ID, batch_inp)

        return noisy_batch, p_mask, u0

    def _eval_target_nll_mc(self, inp, cont_toks, inplen):
        self.model.eval()
        target_len = len(cont_toks)

        # Use chunking to avoid OOM when num_mc is too large
        max_batch_size = self._max_chunk_size  # Adjust based on available memory
        total_loss = 0.0
        u0 = None
        
        # Define mask_positions outside the loop so it can be used in greedy prediction
        mask_positions = torch.arange(target_len, device=self.device)
        mask_positions = inplen - len(cont_toks) + mask_positions
        
        for chunk_start in range(0, self._num_mc, max_batch_size):
            chunk_end = min(chunk_start + max_batch_size, self._num_mc)
            chunk_size = chunk_end - chunk_start
            
            # Create batch of inputs by repeating and masking different positions
            batch_inp = inp.unsqueeze(0).repeat(chunk_size, 1)

            # generate random masks
            noisy_batch_inp = batch_inp.clone()
            noisy_batch_inp_, p_mask, u0 = self._forward_process(batch_inp, chunk_start=chunk_start, u0=u0)
            cont_toks = torch.tensor(cont_toks, dtype=torch.long, device=self.device)

            noisy_batch_inp[:, mask_positions] = noisy_batch_inp_[:, mask_positions]
            mask_indices = (noisy_batch_inp == self.NOISY_MASK_TOKEN_ID) & (torch.arange(noisy_batch_inp.shape[1], device=self.device) < inplen)

            with torch.no_grad():
                logits = self._model_call(noisy_batch_inp)

            if logits is not None:
                chunk_loss = F.cross_entropy(logits[mask_indices], batch_inp[mask_indices], reduction='none') / p_mask[mask_indices]
                total_loss += chunk_loss.sum().cpu().item()
        loss = total_loss / self._num_mc

        # greedy prediction
        if not self.only_mc_nll:
            pred_inp = inp.unsqueeze(0)
            pred_inp[: , mask_positions] = self.NOISY_MASK_TOKEN_ID
            with torch.no_grad():
                for i in range(len(cont_toks)):
                    mask_index = (pred_inp == self.NOISY_MASK_TOKEN_ID)
                    pred_logits = self._model_call(pred_inp)[mask_index]
                    # select the argmax of the logits
                    pred_tok = pred_logits.argmax(dim=-1)
                    
                    p = torch.softmax(pred_logits.to(torch.float32), dim=-1)
                    confidence = torch.gather(p, dim=-1, index=torch.unsqueeze(pred_tok, -1)).squeeze(dim=-1)
                    _, index = torch.sort(confidence, descending=True)
                    pred_tok[index[1:]] = self.NOISY_MASK_TOKEN_ID
                    pred_inp[mask_index] = pred_tok.clone()
                pred_ans = pred_inp[: , mask_positions]
                max_equal = (pred_ans == cont_toks).all().cpu()
            answer = (float(-loss), bool(max_equal))
        else:
            answer = (float(-loss), None)
        return answer


    def create_model_inputs(self, tokens):
        args = get_args()
        # create attention mask (no need so all ones)
        attention_mask = torch.ones(1, 1, tokens.shape[1], tokens.shape[1], device=tokens.device)
        attention_mask = (attention_mask < 0.5)
        
        # Get the masks and postition ids.
        _, loss_mask, position_ids = get_ltor_masks_and_position_ids(
            tokens,
            self.EOT_TOKEN_ID,
            args.reset_position_ids,
            args.reset_attention_mask,
            args.eod_mask_loss,
            True)
        
        masked_indices = tokens == self.NOISY_MASK_TOKEN_ID
        # Calculate p_mask as ratio of masked tokens to total tokens
        num_masked = masked_indices.sum()
        total_tokens = tokens.shape[0] * tokens.shape[1]
        p_mask = (num_masked / total_tokens).item()
        p_mask = torch.full_like(masked_indices, p_mask, dtype=torch.float)
        return (tokens, position_ids, attention_mask), (tokens, loss_mask, masked_indices, p_mask)

    def _model_call(self, inps):
        args = get_args()
        if args.deepspeed:
            if args.no_pipeline_parallel:
                # self.model.set_batch_fn(self.create_model_inputs)
                # round up to multiple of micro_batch_size
                new_size = ((len(inps) + args.micro_batch_size-1)  // args.micro_batch_size) * args.micro_batch_size
                padded = F.pad(inps, (0, 0, 0, new_size-len(inps)), value = 0)
                # dummy data iterator for pipelining.
                data_iterator = list((torch.stack(inp) for inp in utils.chunks(padded, args.micro_batch_size)))
                self.model.micro_batches = len(data_iterator)
                # output = self.model.eval_batch(iter(data_iterator), compute_loss = False, reduce_output = None)
                output = []
                for tokens in data_iterator:
                    attention_mask = torch.ones(1, 1, tokens.shape[1], tokens.shape[1], device=tokens.device)
                    attention_mask = (attention_mask < 0.5)
                    _, loss_mask, position_ids = get_ltor_masks_and_position_ids(
                                                                tokens,
                                                                self.EOT_TOKEN_ID,
                                                                args.reset_position_ids,
                                                                args.reset_attention_mask,
                                                                args.eod_mask_loss,
                                                                True)
                    a_output, *other_losses = self.model(tokens,
                        position_ids,
                        attention_mask,
                        tokentype_ids=None)
                    output.append(a_output)

                if output is not None:
                    output = torch.cat(output, 0)
                    output = output[:len(inps)]
                else:
                    output = None

                # hack #2 for adaptive_seq_len to work as total_loss gets appended to and shapes aren't the same
                if args.adaptive_seq_len:
                    self.model.total_loss = None
            else:
                self.model.set_batch_fn(self.create_model_inputs)

                # round up to multiple of micro_batch_size
                new_size = ((len(inps) + self._batch_size-1)  // self._batch_size) * self._batch_size
                padded = F.pad(inps, (0, 0, 0, new_size-len(inps)), value = 0)
                # dummy data iterator for pipelining.
                data_iterator = list((torch.stack(inp) for inp in utils.chunks(padded, self._batch_size)))
                self.model.micro_batches = len(data_iterator)
                output = self.model.eval_batch(iter(data_iterator), compute_loss = False, reduce_output = None)
                
                if output is not None:
                    output = torch.cat(output, 1).permute(1, 0, 2)
                    output = output[:len(inps)]
                else:
                    output = None

                # hack #2 for adaptive_seq_len to work as total_loss gets appended to and shapes aren't the same
                if args.adaptive_seq_len:
                    self.model.total_loss = None
        else:
            # Since the shape of the micro-batch will change
            # We need set the correct shapes here
            # So that latter pipeline stages knows which shapes to expect.
            # Otherwise we will deadlock.

            args.micro_batch_size = len(inps)
            args.seq_length = len(inps[0])
            args.max_position_embeddings = args.seq_length

            input_tensor = recv_forward()

            # Forward pass through the model.
            unwrapped_model = unwrap_model(self.model, (torchDDP, LocalDDP, Float16Module))
            unwrapped_model.set_input_tensor(input_tensor)
            output = self.model(*self.create_model_inputs(inps)[0])
            send_forward(output)

        if mpu.is_pipeline_last_stage():
            return gather_from_tensor_model_parallel_region(output.contiguous())[..., :self.tokenizer.vocab_size]
        else:
            return None

    def tokenizer_encode(self, text):
        """Tokenize text *without* adding special tokens."""
        # Splitting this into its own method in case we need to handle special cases for different tokenizers
        from megatron.tokenizer.gpt2_tokenization import GPT2Tokenizer
        if isinstance(self.tokenizer.tokenizer, GPT2Tokenizer):
            return self.tokenizer.tokenizer.encode(text)
        else:
            return self.tokenizer.tokenizer.encode(text, add_special_tokens=False)


from megatron.initialize import initialize_megatron
import megatron

from tools.convert_checkpoint.deepspeed_checkpoint import DeepSpeedCheckpoint
from tools.convert_checkpoint.deepspeed_to_megatron import _create_rank_checkpoint

def override_args(args, override_args, skip_keys, skip_if_specified_keys):
    for k, v in vars(override_args).items():
        if k in skip_keys:
            continue
        if k in skip_if_specified_keys and getattr(args, k) is not None:
            continue
        setattr(args, k, v)


# Note(Hesslow):
# The model loading is a bit convoluted.
# We want to parse out the model arguments from the checkpoint and use those to initialize megatron-ds.
#
# However megatron-ds expects its arguments on the command line.
# And at that point we don't know them.
#
# Instead we use Jasons way: we load the arguments form the checkpoint and then override _parse_args to return whatever args we want.
#
# If the checkpoint is old, some new arguments may have been introduced and the code will expect these arguments to exist.
# In order to support this we _first_ parse the arguments normally, and then override them with the arguments from the checkpoint.
# Keeping the default-value of newer arguments.
#
# We then use the megatron deepspeed converter to load the deepspeed checkpoints as if they we're megatron checkpoints.
def load_ds_checkpoint_and_setup_megatron(extra_args_provider):
    # parse the megatorn args. But wait with initalizing megatron.
    # avoid printing the arguments, since they will later be overridden.
    _print_args = megatron.arguments._print_args
    megatron.arguments._print_args = lambda *_args, **kwarg: None
    args = parse_args(extra_args_provider=extra_args_provider)

    ds_checkpoint = DeepSpeedCheckpoint(args.load,
                                        tp_degree=args.tensor_model_parallel_size,
                                        pp_degree=args.pipeline_model_parallel_size,
                                        no_pp=args.no_pipeline_parallel)


    cp_args = ds_checkpoint.get_args()
    # Merge the current args with the checkpoint args.
    skip_keys = ['world_size', 'rank', 'local_rank','device_count', 'micro_batch_size','global_batch_size', 'batch_size', 'tensorboard_dir', 'deepspeed', 'deepspeed_config',
                     'data_parallel_size', 'pipeline_model_parallel_size', 'tensor_model_parallel_size', 'moe_expert_parallel_size', 'moe_token_dropping', 'load', 'load_tag', 'rampup_batch_size', 'iteration', 'inference', 'random_ltd'
                     ,'num_mc', 'make_vocab_size_divisible_by', 'padded_vocab_size', 'untie_embeddings_and_output_weights', 'untie_with_additional_mask']

    skip_if_specified = ['merge_file', 'vocab_file']

    if args.eval_fp32:
        cp_args.fp16 = False
        cp_args.bf16 = False
        cp_args.params_dtype = torch.float32

    cp_args.tokenizer_type = 'GPT2BPETokenizer'

    override_args(args, cp_args, skip_keys, skip_if_specified)

    # stop megatron from reparsing the arguments.
    megatron.arguments.parse_args = lambda *_args, **kwarg: args
    megatron.global_vars._ensure_var_is_not_initialized = lambda *_args, **kwarg: None
    megatron.global_vars._GLOBAL_ARGS = args

    initialize_megatron(extra_args_provider=extra_args_provider, args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})
    megatron.global_vars._GLOBAL_ARGS = args
    torch.distributed.barrier()

    # Initializing megatron will update eg. tokenizer size. Override again.
    override_args(args, cp_args, skip_keys, skip_if_specified)

    # print final arguments.
    _print_args("eval_harness arguments", args)
    if args.deepspeed:

        # Hack #3:
        # Loading pipelined models in deepspeed with different TP than it was originally trained on fails
        # due to a sanity check, that makes sure that all state_dicts that we merge contains attention layers.
        # This, however, is not true for pipelining when we will merge the state_dict for the embeddings which
        # which does not contain these attention-specific keys.
        #
        # Deepspeed does however manage to load the model if we just turn off this sanity check.
        import deepspeed
        deepspeed.runtime.state_dict_factory.MegatronSDLoader.sanity_check = lambda self, ckpt_file_name: None


        cp_path = args.load
        args.load = None
        args.deepspeed_config_dict = _create_ds_config_dict()
        model, _, _ = setup_model_and_optimizer(model_provider, ModelType.encoder_or_decoder)
        model = model[0]
        zero_enabled = model._config.zero_enabled
        model._config.zero_enabled = False
        _, _ = model.load_checkpoint(cp_path, tag = args.load_tag, load_optimizer_states=False, load_lr_scheduler_states=False, load_module_only=True)
        model._config.zero_enabled = zero_enabled
    else:
        model = get_model(model_provider)[0]
        # Initialize megatron model using the parsed state dict.
        sd = _create_rank_checkpoint(ds_checkpoint, None, mpu.get_tensor_model_parallel_rank(), mpu.get_pipeline_model_parallel_rank(), True)

        model.load_state_dict(sd['model'], strict=True)

    if args.eval_fp32:
        model = model.float()

    torch.distributed.barrier()
    return model

def tasks_args(parser):
    """Provide extra arguments required for tasks."""
    group = parser.add_argument_group(title='Evaluation options')
    group.add_argument('--task_list', type=str, default = "all", help='Either "all" or comma separated list of tasks.')
    group.add_argument('--results_path', type=str, default = "./results.json", help='Path to where the results will be stored.')
    group.add_argument('--adaptive_seq_len',  default = False, action='store_true',
                       help='Should the sequence length be adapted to the batch during evaluation, if in fp16 the results will be slightly different due to numerical errors but greatly speed up evaluation.')
    group.add_argument('--num_fewshot', type=int, default = 0, help='Number of few-shot prompts.')
    group.add_argument('--eval_fp32',  default = False, action='store_true', help='Should the evaluation run in fp32')
    group.add_argument('--eval_method', type=str, choices=['ar', 'mc'], default='ar', help='Evaluation method of diffusion model')
    group.add_argument('--sampling_eps', type=float, default=0.0, help='Sampling epsilon for diffusion model')
    group.add_argument('--only_generate', default=False, action='store_true', help='Only evaluate the accuracy, not the nll')
    group.add_argument('--only_mc_nll', default=False, action='store_true', help='Only evaluate the nll, not the accuracy')
    group.add_argument('--max_chunk_size', type=int, default=32, help='Maximum chunk size for mc evaluation')
    return parser

from megatron.arguments import parse_args

def main():
    start = time.time()
    model = load_ds_checkpoint_and_setup_megatron(extra_args_provider=tasks_args)

    args = get_args()
    if args.deepspeed and args.adaptive_seq_len:
        # adaptive_seq_len hack #1:
        # CL automatically enables reset_activation_shape() which allows us to change input shapes
        # and it also reshapes the attenion scores in attention_mask_func
        args.curriculum_learning_legacy = 1

    task_list = ALL_TASKS if args.task_list == 'all' else args.task_list.split(',')
    task_dict = tasks.get_task_dict(task_list)

    model.module.activation_checkpoint_interval = 0
    model._compute_loss = False
    model.fwd_outputs = []

    tokenizer = get_tokenizer()
    adaptor = EvalHarnessDiffAdaptor(model, tokenizer)
    results = evaluator.evaluate(adaptor, task_dict, False, args.num_fewshot, None)

    if mpu.is_pipeline_last_stage() and mpu.get_tensor_model_parallel_rank() == 0:
        print(json.dumps(results, indent=2))
        with open(args.results_path, 'w') as outfile:
            json.dump(results, outfile, indent = 4)
    end = time.time()
    print("evaluation of {} ends in {:.2f} sec, or {:.2f} min, or {:.2f} hr".format(args.task_list, end-start, (end-start)/60.0, (end-start)/3600.0))

if __name__ == '__main__':
    main()