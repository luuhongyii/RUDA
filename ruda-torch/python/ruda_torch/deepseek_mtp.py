from __future__ import annotations

import copy
import inspect
import json
import math
from dataclasses import dataclass
from pathlib import Path
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .causal_finetuning import chunked_lm_cross_entropy, CausalLMFinetuner, CausalLMTrainer
from .weight_loading import SafetensorWeights, TensorSource


@dataclass
class DeepseekMtpTrainingOutput:
    loss: torch.Tensor | None
    logits: torch.Tensor | None
    main_loss_sum: torch.Tensor | None
    main_count: torch.Tensor | None
    mtp_loss_sums: tuple
    mtp_counts: tuple


class _Norm(nn.Module):
    def __init__(self, width, epsilon):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(width))
        self.epsilon = epsilon

    def forward(self, value):
        work = value.float()
        return (work * torch.rsqrt(work.square().mean(-1, keepdim=True) + self.epsilon)).to(value.dtype) * self.weight


class DeepseekMtpPredictor(nn.Module):
    def __init__(self, decoder, rotary, config):
        super().__init__()
        self.decoder, self.rotary_emb = decoder, rotary
        self.enorm = _Norm(config.hidden_size, config.rms_norm_eps)
        self.hnorm = _Norm(config.hidden_size, config.rms_norm_eps)
        self.eh_proj = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)
        self.shared_head = nn.Module()
        self.shared_head.norm = _Norm(config.hidden_size, config.rms_norm_eps)
        self.config = copy.deepcopy(config)
        self.config.num_hidden_layers = 1
        self.decoder.self_attn.layer_idx = 0
        from transformers.masking_utils import create_causal_mask
        parameters = inspect.signature(create_causal_mask).parameters
        if not {'inputs_embeds', 'input_embeds'} & set(parameters):
            raise ValueError('unsupported Transformers causal-mask interface')
        self._mask_input = 'inputs_embeds' if 'inputs_embeds' in parameters else 'input_embeds'
        self._mask_cache_position = 'cache_position' in parameters

    def forward(self, embedding, previous, attention_mask, position_ids, *, cache=None, document_ids=None):
        from transformers.masking_utils import create_causal_mask
        if embedding.shape != previous.shape or embedding.ndim != 3:
            raise ValueError('MTP embedding and preceding hidden states must have matching [B,T,H] shapes')
        hidden = self.eh_proj(torch.cat((self.enorm(embedding), self.hnorm(previous)), dim=-1))
        if hidden.shape[1] == 0:
            return self.shared_head.norm(hidden + self.decoder.mlp(self.decoder.post_attention_layernorm(hidden)))
        start = 0 if cache is None else cache.get_seq_length()
        cache_position = torch.arange(start, start + hidden.shape[1], device=hidden.device)
        if document_ids is not None:
            if cache is not None or tuple(document_ids.shape) != tuple(hidden.shape[:2]):
                raise ValueError('document-isolated MTP requires uncached aligned document IDs')
            length = hidden.shape[1]
            allowed = document_ids.unsqueeze(-1) == document_ids.unsqueeze(-2)
            allowed = allowed & (torch.arange(length, device=hidden.device)[:, None] >= torch.arange(length, device=hidden.device)[None, :])
            allowed = allowed & attention_mask[:, None, :].bool() & (document_ids[:, None, :] >= 0)
            attention_mask = torch.where(allowed[:, None], torch.zeros((), device=hidden.device, dtype=hidden.dtype),
                                         torch.full((), torch.finfo(hidden.dtype).min, device=hidden.device, dtype=hidden.dtype))
        kwargs = dict(config=self.config, attention_mask=attention_mask, past_key_values=cache, position_ids=position_ids)
        kwargs[self._mask_input] = hidden
        if self._mask_cache_position:
            kwargs['cache_position'] = cache_position
        mask = create_causal_mask(**kwargs)
        output = self.decoder(hidden, attention_mask=mask, position_ids=position_ids,
                              position_embeddings=self.rotary_emb(hidden, position_ids),
                              past_key_values=cache, cache_position=cache_position, use_cache=cache is not None)
        output = output[0] if isinstance(output, tuple) else output
        return self.shared_head.norm(output)


def _targets(labels, attention_mask, document_ids, depths):
    labels = labels.masked_fill(~attention_mask.bool(), -100)
    if document_ids is not None:
        boundary = torch.cat((torch.zeros_like(labels[:, :1], dtype=torch.bool), document_ids[:, 1:] != document_ids[:, :-1]), dim=1)
        labels = labels.masked_fill(boundary | (document_ids < 0), -100)
    valid, result = attention_mask.bool(), []
    for depth in range(depths):
        offset = depth + 1
        valid = valid[:, :-1] & attention_mask[:, offset:].bool()
        if document_ids is not None:
            documents = document_ids[:, offset:]
            valid = valid & (document_ids[:, :documents.shape[1]] == documents) & (documents >= 0)
        targets = labels[:, offset + 1:].masked_fill(~valid[:, :-1], -100)
        if document_ids is not None:
            targets = targets.masked_fill(document_ids[:, offset:-1] != document_ids[:, offset + 1:], -100)
        result.append((valid, targets))
    return labels, result


def _logits(head, hidden):
    logits = head(hidden)
    if getattr(head, 'gather_output', True) is False:
        from .parallel_training import gather_from_tensor_parallel
        logits = gather_from_tensor_parallel(logits, head.group)
    return logits


class DeepseekMtpForCausalLM(CausalLMFinetuner):
    def __init__(self, target, predictors, *, mtp_loss_weight, token_chunk_size=32, activation_checkpointing=True):
        nn.Module.__init__(self)
        if not predictors or not math.isfinite(mtp_loss_weight) or mtp_loss_weight < 0:
            raise ValueError('MTP requires loaded predictors and an explicit finite nonnegative objective weight')
        if type(token_chunk_size) is not int or token_chunk_size < 1:
            raise ValueError('MTP token chunk size must be positive')
        self.target, self.predictors = target, nn.ModuleList(predictors)
        self.config = target.config
        self.mtp_loss_weight = float(mtp_loss_weight)
        self.token_chunk_size = token_chunk_size
        self.activation_checkpointing = bool(activation_checkpointing)
        self.label_smoothing = 0.

    @property
    def backbone(self):
        return self.target.model

    @property
    def head(self):
        return self.target.get_output_embeddings()

    def get_extra_state(self):
        return dict(version=1, predictors=len(self.predictors), mtp_loss_weight=self.mtp_loss_weight,
                    token_chunk_size=self.token_chunk_size, activation_checkpointing=self.activation_checkpointing,
                    label_smoothing=self.label_smoothing)

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError('MTP training objective or checkpoint configuration differs')

    def new_mtp_cache(self):
        from transformers.cache_utils import DynamicCache
        return [DynamicCache(config=predictor.config) for predictor in self.predictors]

    def forward_mtp(self, depth, input_ids, previous_hidden_states, *, position_ids, attention_mask=None, cache=None):
        predictor = self.predictors[depth]
        embedding = self.target.get_input_embeddings()(input_ids)
        hidden = predictor(embedding, previous_hidden_states, attention_mask, position_ids, cache=cache)
        return _logits(self.target.get_output_embeddings(), hidden), hidden

    def forward(self, input_ids, attention_mask=None, labels=None, *, position_ids=None,
                document_ids=None, output_logits=None, label_smoothing=None, reduction=None,
                mtp_loss_scales=None, include_mtp=True):
        if reduction not in (None, 'mean', 'sum') or (reduction is not None and labels is None):
            raise ValueError('MTP loss reduction requires labels and must be mean or sum')
        if reduction == 'sum' and include_mtp and mtp_loss_scales is None:
            raise ValueError('MTP summed accumulation requires globally normalized depth scales; use DeepseekMtpTrainer')
        label_smoothing = self.label_smoothing if label_smoothing is None else label_smoothing
        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError('MTP requires nonempty [B,T] input IDs')
        batch, length = input_ids.shape
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        if attention_mask.shape != input_ids.shape:
            raise ValueError('MTP requires a two-dimensional token-validity mask')
        if position_ids is None:
            position_ids = torch.arange(length, device=input_ids.device).unsqueeze(0).expand(batch, -1)
        else:
            position_ids = position_ids.expand(batch, length)
        if document_ids is not None and document_ids.shape != input_ids.shape:
            raise ValueError('MTP document IDs must cover the input tokens')
        model_mask = attention_mask
        if document_ids is not None:
            allowed = document_ids.unsqueeze(-1) == document_ids.unsqueeze(-2)
            allowed = allowed & (torch.arange(length, device=input_ids.device)[:, None] >= torch.arange(length, device=input_ids.device)[None, :])
            allowed = allowed & attention_mask[:, None, :].bool() & (document_ids[:, None, :] >= 0)
            dtype = self.target.get_input_embeddings().weight.dtype
            model_mask = torch.where(allowed[:, None], torch.zeros((), dtype=dtype, device=input_ids.device),
                                     torch.full((), torch.finfo(dtype).min, dtype=dtype, device=input_ids.device))
        def backbone(ids):
            return self.target.model(input_ids=ids, attention_mask=model_mask, position_ids=position_ids,
                                     use_cache=False, return_dict=True).last_hidden_state
        hidden = checkpoint(backbone, input_ids, use_reentrant=False, preserve_rng_state=True) if (
            self.activation_checkpointing and self.training and torch.is_grad_enabled()) else backbone(input_ids)
        head = self.target.get_output_embeddings()
        logits = _logits(head, hidden) if (labels is None if output_logits is None else output_logits) else None
        if labels is None:
            return DeepseekMtpTrainingOutput(None, logits, None, None, (), ())
        if labels.shape != input_ids.shape or labels.device != input_ids.device:
            raise ValueError('MTP labels must match the input geometry and device')
        labels, supervision = _targets(labels, attention_mask, document_ids, len(self.predictors))
        options = dict(token_chunk_size=self.token_chunk_size, reduction='sum', recompute=self.activation_checkpointing,
                       label_smoothing=label_smoothing)
        main_sum = chunked_lm_cross_entropy(hidden, head, labels, **options)
        main_count = (labels[:, 1:] != -100).sum()
        if not include_mtp:
            return main_sum if reduction == 'sum' else main_sum / main_count.clamp_min(1)
        upstream = hidden
        sums, counts = [], []
        for depth, predictor in enumerate(self.predictors):
            offset = depth + 1
            count = max(0, length - offset)
            previous = upstream[:, :count]
            valid, targets = supervision[depth]
            documents = None if document_ids is None else document_ids[:, offset:]
            positions = position_ids[:, offset:]
            def predict(embedded, prior, mask, positions, module=predictor, documents=documents):
                return module(embedded, prior, mask, positions, document_ids=documents)
            embedding = self.target.get_input_embeddings()(input_ids[:, offset:])
            upstream = checkpoint(predict, embedding, previous, valid, positions, use_reentrant=False,
                                  preserve_rng_state=True) if (self.activation_checkpointing and self.training and torch.is_grad_enabled()) else predict(embedding, previous, valid, positions)
            sums.append(chunked_lm_cross_entropy(upstream[:, :-1], head, targets, shift=False, **options))
            counts.append((targets != -100).sum())
        loss = main_sum / main_count.clamp_min(1)
        loss = loss + self.mtp_loss_weight * sum(value / count.clamp_min(1) for value, count in zip(sums, counts)) / len(self.predictors)
        if reduction == 'mean':
            return loss
        if reduction == 'sum':
            if mtp_loss_scales.shape != (len(self.predictors),) or mtp_loss_scales.device != input_ids.device:
                raise ValueError('MTP accumulation scales must cover every depth on the execution device')
            return main_sum + self.mtp_loss_weight * sum(value * mtp_loss_scales[depth] for depth, value in enumerate(sums)) / len(self.predictors)
        return DeepseekMtpTrainingOutput(loss, logits, main_sum, main_count, tuple(sums), tuple(counts))


class DeepseekMtpTrainer(CausalLMTrainer):
    def _template(self):
        template = getattr(self.model, '_template', self.model)
        if not isinstance(template, DeepseekMtpForCausalLM):
            raise TypeError('DeepseekMtpTrainer requires a loaded MTP target/predictor model')
        return template

    def _batch_counts(self, batches):
        counts = []
        for batch in batches:
            allowed = {'input_ids', 'attention_mask', 'labels', 'position_ids', 'document_ids', 'mtp_loss_scales', 'include_mtp'}
            if not {'input_ids', 'attention_mask', 'labels'} <= set(batch) or not set(batch) <= allowed:
                raise ValueError('MTP batches require CPU token IDs, validity masks and labels')
            ids, mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']
            if any(not isinstance(batch[k], torch.Tensor) or batch[k].device.type != 'cpu' for k in ('input_ids', 'attention_mask', 'labels')):
                raise ValueError('MTP trainer consumes collated CPU batches')
            if ids.ndim != 2 or ids.shape[1] == 0 or mask.shape != ids.shape or labels.shape != ids.shape or mask.dtype != torch.bool:
                raise ValueError('invalid MTP token batch geometry')
            if ids.dtype not in (torch.int32, torch.int64) or labels.dtype not in (torch.int32, torch.int64):
                raise ValueError('MTP token IDs and labels must be integer tensors')
            for name in ('position_ids', 'document_ids'):
                value = batch.get(name)
                if value is not None and (value.shape != ids.shape or value.device.type != 'cpu' or value.dtype != torch.int64):
                    raise ValueError('MTP positions/document IDs must be CPU int64 token matrices')
            normalized, _ = _targets(labels, mask, batch.get('document_ids'), 0)
            counts.append(int((normalized[:, 1:] != -100).sum()))
        return counts

    def _loss_options(self):
        template = self._template()
        return {**super()._loss_options(), 'mtp_loss_weight': template.mtp_loss_weight, 'mtp_depths': len(template.predictors)}

    def train_step(self, microbatches):
        batches = list(microbatches)
        depths = len(self._template().predictors)
        error = None
        try:
            main_counts = self._batch_counts(batches)
            totals = [sum(main_counts)] + [0] * depths
            for batch in batches:
                _, targets = _targets(batch['labels'], batch['attention_mask'], batch.get('document_ids'), depths)
                for depth, (_, target) in enumerate(targets):
                    totals[depth + 1] += int((target != -100).sum())
        except (ValueError, TypeError, AttributeError) as failure:
            if self.replica_group is None:
                raise
            error = str(failure)
        if self.replica_group is not None:
            for failure in self.replica_group.gather_metadata(error):
                if failure:
                    raise ValueError(failure)
            global_totals = []
            for total in totals:
                peers = self.replica_group.gather_metadata(total)
                global_totals.append(self.replica_group.total_weight(total) if any(peers) else 0)
            totals = global_totals
        scales = torch.tensor([totals[0] / count if count else 0. for count in totals[1:]], dtype=torch.float32)
        return super().train_step([{**batch, 'mtp_loss_scales': scales, 'include_mtp': True} for batch in batches])

    def evaluate(self, microbatches):
        return super().evaluate([{**batch, 'include_mtp': False} for batch in microbatches])


def _copy_source(reader, name, indices, destination, block_shape, used, rows=None, columns=None):
    shape = reader.shape(name)[len(indices):]
    used.add(name)
    row_start, row_end = (0, shape[0]) if rows is None else rows
    if not 0 <= row_start < row_end <= shape[0]:
        raise ValueError(f'{name}: invalid MTP row partition')
    if len(shape) == 1:
        if destination.shape != (row_end - row_start,) or columns is not None:
            raise ValueError(f'{name}: MTP vector partition shape differs')
        value = reader.read(TensorSource(name, indices + (slice(row_start, row_end),)))
        if str(value.dtype).startswith('torch.float8'):
            raise ValueError('FP8 vector storage requires an explicit checkpoint vector scaling format')
        destination.copy_(value.to(device=destination.device, dtype=destination.dtype))
        return
    if len(shape) != 2:
        raise ValueError('MTP source must be a vector or a matrix after expert selection')
    total_rows, total_columns = shape
    col_start, col_end = (0, total_columns) if columns is None else columns
    if not 0 <= col_start < col_end <= total_columns or destination.shape != (row_end - row_start, col_end - col_start):
        raise ValueError(f'{name}: MTP matrix partition shape differs')
    step = block_shape[0] if block_shape else 128
    scale_name = name[:-len('.weight')] + '.weight_scale_inv' if name.endswith('.weight') else name + '_scale_inv'
    scales = None
    for start in range(row_start, row_end, step):
        end = min(row_end, start + step)
        value = reader.read(TensorSource(name, indices + (slice(start, end), slice(col_start, col_end))))
        if str(value.dtype).startswith('torch.float8'):
            if value.dtype != torch.float8_e4m3fn or block_shape is None or scale_name not in reader:
                raise ValueError(f'{name}: standard DeepSeek E4M3 block scale metadata is required')
            if scales is None:
                scales = reader.read(TensorSource(scale_name, indices)).float()
                if tuple(scales.shape) != (math.ceil(total_rows / block_shape[0]), math.ceil(total_columns / block_shape[1])):
                    raise ValueError(f'{name}: FP8 block scale geometry differs')
                used.add(scale_name)
            row_ids = torch.arange(start, end) // block_shape[0]
            col_ids = torch.arange(col_start, col_end) // block_shape[1]
            value = value.float() * scales[row_ids[:, None], col_ids[None, :]]
        destination[start-row_start:end-row_start].copy_(value.to(device=destination.device, dtype=destination.dtype))


def _load_predictor(predictor, reader, prefix, device, dtype, block_shape):
    from .expert_parallel import ExpertParallelMLP
    from .parallel_training import ColumnParallelLinear, RowParallelLinear
    used = set()
    with torch.no_grad():
        for name, source in list(predictor.named_parameters()) + list(predictor.named_buffers()):
            if name.startswith('rotary_emb.'):
                continue
            local = name.removeprefix('decoder.')
            source_name = prefix + '.' + local
            parent_name, _, leaf = name.rpartition('.')
            parent = predictor.get_submodule(parent_name) if parent_name else predictor
            target_dtype = torch.float32 if name.endswith(('e_score_correction_bias', 'mlp.gate.weight')) else dtype
            value = torch.empty(tuple(source.shape), device=device, dtype=target_dtype)
            if source.ndim == 3 and local in ('mlp.experts.gate_up_proj', 'mlp.experts.down_proj'):
                direct = source_name if source_name in reader else source_name + '.weight'
                bank = parent if isinstance(parent, ExpertParallelMLP) else None
                offset = 0 if bank is None else bank.partition.start
                width = source.shape[1] // 2 if local.endswith('gate_up_proj') else source.shape[2]
                first = 0 if bank is None or bank.tensor_group is None else width * bank.tensor_group.rank
                intermediate = width if bank is None else bank.intermediate_size
                for expert in range(source.shape[0]):
                    global_id = offset + expert
                    if direct in reader:
                        if local.endswith('gate_up_proj'):
                            for half in range(2):
                                _copy_source(reader, direct, (global_id,), value[expert, half*width:(half+1)*width], block_shape, used,
                                             rows=(half*intermediate+first, half*intermediate+first+width))
                        else:
                            _copy_source(reader, direct, (global_id,), value[expert], block_shape, used, columns=(first, first+width))
                    elif local.endswith('gate_up_proj'):
                        half = source.shape[1] // 2
                        for projection, target in [('gate_proj', value[expert, :half]), ('up_proj', value[expert, half:])]:
                            _copy_source(reader, f'{prefix}.mlp.experts.{global_id}.{projection}.weight', (), target, block_shape, used, rows=(first, first+width))
                    else:
                        _copy_source(reader, f'{prefix}.mlp.experts.{global_id}.down_proj.weight', (), value[expert], block_shape, used, columns=(first, first+width))
            else:
                rows = columns = None
                if isinstance(parent, ColumnParallelLinear):
                    begin = (parent.group.rank // getattr(parent, 'replicas', 1)) * value.shape[0]
                    rows = (begin, begin + value.shape[0])
                elif isinstance(parent, RowParallelLinear) and leaf == 'weight':
                    begin = parent.group.rank * value.shape[1]
                    columns = (begin, begin + value.shape[1])
                _copy_source(reader, source_name, (), value, block_shape, used, rows=rows, columns=columns)
            if leaf in parent._parameters:
                parameter = nn.Parameter(value, requires_grad=source.requires_grad)
                for key, metadata in vars(source).items():
                    if key.startswith('_ruda_'):
                        setattr(parameter, key, metadata)
                parent._parameters[leaf] = parameter
            else:
                parent._buffers[leaf] = value
    return used


def load_deepseek_mtp(target, directory, *, mtp_loss_weight, device=None, dtype=None,
                      token_chunk_size=32, activation_checkpointing=True,
                      expert_group=None, tensor_group=None, expert_group_key='deepseek-mtp-expert'):
    root = Path(directory)
    source = json.loads((root / 'config.json').read_text(encoding='utf-8'))
    count = source.get('num_nextn_predict_layers')
    if type(count) is not int or count < 1 or getattr(target.config, 'model_type', None) != 'deepseek_v3':
        raise ValueError('MTP loading requires a HF DeepSeek V3 target and real checkpoint predictor modules')
    for key in ('hidden_size', 'num_hidden_layers', 'num_attention_heads', 'kv_lora_rank', 'qk_rope_head_dim',
                'qk_nope_head_dim', 'v_head_dim', 'vocab_size', 'n_routed_experts', 'moe_intermediate_size'):
        if source.get(key) != getattr(target.config, key, None):
            raise ValueError(f'MTP checkpoint and target configuration differ: {key}')
    embedding = target.get_input_embeddings().weight
    device = embedding.device if device is None else torch.device(device)
    dtype = embedding.dtype if dtype is None else dtype
    if device.type == 'meta' or device != embedding.device or dtype != embedding.dtype or dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError('MTP parameters must use the materialized target embedding device and floating dtype')
    decoder_class = type(target.model.layers[-1])
    if decoder_class.__name__ != 'DeepseekV3DecoderLayer' or not decoder_class.__module__.startswith('transformers.models.deepseek_v3.'):
        raise ValueError('load MTP before wrapping or parallelizing the HF target decoder')
    quantization = source.get('quantization_config') or {}
    block_shape = quantization.get('weight_block_size')
    if block_shape is not None and (len(block_shape) != 2 or any(type(n) is not int or n < 1 for n in block_shape)):
        raise ValueError('invalid checkpoint FP8 weight block shape')
    reader = SafetensorWeights(root)
    predictors, used = [], set()
    for depth in range(count):
        index = target.config.num_hidden_layers + depth
        prefix = f'model.layers.{index}'
        with torch.device('meta'):
            decoder = decoder_class(copy.deepcopy(target.config), layer_idx=index)
            predictor = DeepseekMtpPredictor(decoder, nn.Identity(), target.config)
        predictor.rotary_emb = copy.deepcopy(target.model.rotary_emb)
        if expert_group is not None:
            from .expert_parallel import expert_parallelize_deepseek
            expert_parallelize_deepseek(predictor, expert_group, group_key=expert_group_key)
        if tensor_group is not None:
            from .parallel_training import tensor_parallelize
            plan = {name: {'kind': 'column', 'gather_output': True} for name, module in predictor.named_modules()
                    if isinstance(module, nn.Linear) and not name.endswith('.gate')}
            tensor_parallelize(predictor, tensor_group, plan)
            if expert_group is not None:
                from .expert_parallel import tensor_parallelize_experts
                tensor_parallelize_experts(predictor, tensor_group)
        used.update(_load_predictor(predictor, reader, prefix, device, dtype, block_shape))
        if expert_group is not None:
            from .expert_parallel import ExpertParallelMLP
            for name, bank in predictor.named_modules():
                if not isinstance(bank, ExpertParallelMLP):
                    continue
                bank.source_prefix = prefix + '.' + name.removeprefix('decoder.')
                for expert in range(bank.num_experts):
                    if not bank.partition.start <= expert < bank.partition.end:
                        unused_prefix = f'{bank.source_prefix}.{expert}.'
                        used.update(key for key in reader if key.startswith(unused_prefix))
        for alias, original in [('embed_tokens.weight', 'model.embed_tokens.weight'), ('shared_head.head.weight', 'lm_head.weight'),
                                ('shared_head.head.weight_scale_inv', 'lm_head.weight_scale_inv')]:
            alias = prefix + '.' + alias
            if alias not in reader:
                continue
            if original not in reader or reader.shape(alias) != reader.shape(original):
                raise ValueError('MTP shared tensor does not match the target checkpoint shape')
            rows = reader.shape(alias)[0]
            for start in range(0, rows, 128):
                selection = (slice(start, min(rows, start + 128)),)
                a, b = reader.read(TensorSource(alias, selection)), reader.read(TensorSource(original, selection))
                if a.dtype != b.dtype or not torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)):
                    raise ValueError('MTP embedding/head alias differs from the target checkpoint tensor')
            used.add(alias)
        unused = [name for name in reader if name.startswith(prefix + '.') and name not in used]
        if unused:
            raise ValueError(f'unrepresented MTP checkpoint tensors: {unused}')
        predictor.train(target.training)
        predictors.append(predictor)
    result = DeepseekMtpForCausalLM(target, predictors, mtp_loss_weight=mtp_loss_weight,
                                  token_chunk_size=token_chunk_size, activation_checkpointing=activation_checkpointing)
    result.train(target.training)
    return result


def parallelize_deepseek_mtp(model, *, expert_group=None, tensor_group=None, expert_group_key='deepseek-mtp-expert', tensor_plan=None):
    if not isinstance(model, DeepseekMtpForCausalLM):
        raise TypeError('MTP parallel conversion requires the target and loaded predictors together')
    if expert_group is not None:
        from .expert_parallel import expert_parallelize_deepseek
        expert_parallelize_deepseek(model, expert_group, group_key=expert_group_key)
    if tensor_group is not None:
        from .parallel_training import tensor_parallelize
        from .expert_parallel import ExpertParallelMLP, tensor_parallelize_experts
        if tensor_plan is None:
            tensor_plan = {}
            for name, module in model.named_modules():
                if isinstance(module, nn.Embedding):
                    tensor_plan[name] = 'embedding'
                elif isinstance(module, nn.Linear) and not name.endswith('.gate'):
                    tensor_plan[name] = {'kind': 'column', 'gather_output': name != 'target.lm_head'}
        tensor_parallelize(model, tensor_group, tensor_plan)
        if any(isinstance(module, ExpertParallelMLP) for module in model.modules()):
            tensor_parallelize_experts(model, tensor_group)
    return model
