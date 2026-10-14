"""Bounded invocation-metadata specialization of symbolic AOT graphs."""
from collections import OrderedDict
from dataclasses import dataclass, replace
from threading import RLock

import torch

from ._compile_native import NativeCoverageError, _stream_id, partition
from ._compile_shapes import ShapeSpecializationError, attribute, specialize, value_key


class _SpecializationState:
    """Own regions locally: eviction must not leave them in the backend registry."""
    def __init__(self, parent):
        self.parent=parent
        self.regions=[]
        self.graphs=[]
        self.closed=False

    def check_open(self):
        self.parent.check_open()
        if self.closed:raise RuntimeError('dynamic native specialization is closed')

    def add(self, name, amount=1):self.parent.add(name,amount)
    def reason(self, phase, reason):self.parent.reason(phase,reason)

    def record(self, graph):
        self.check_open()
        self.graphs.append(graph)

    def register_region(self, region):
        self.check_open()
        self.regions.append(region)

    def close(self):
        self.closed=True
        remaining=[];first_error=None
        for region in self.regions:
            try:region.close()
            except Exception as exc:
                remaining.append(region)
                first_error=first_error or exc
        self.regions=remaining
        if first_error is not None:raise first_error


@dataclass
class _Entry:
    identifier: int
    graph: object
    state: _SpecializationState | None
    folded: tuple
    reason: str | None = None


class DynamicGraph(torch.nn.Module):
    def __init__(self, reference, state, phase, options):
        super().__init__()
        self.reference=reference
        self.state=state
        self.phase=phase
        self.options=options
        # The outer LRU already separates shape, scalar arguments and stream.
        self._region_options=replace(options,cache_size=1)
        self._attributes=tuple(n.target for n in reference.graph.nodes if n.op=='get_attr')
        self._cache=OrderedDict()
        self._lock=RLock()
        self._closed=False
        self._pending=None
        self._next_identifier=0
        calls=[n for n in reference.graph.nodes if n.op in ('call_function','call_method','call_module')]
        self._template={
            'phase':phase,'operator_count':len(calls),'native_candidate_ops':0,
            'planned_native_ops':0,'native_metadata_ops':0,'host_metadata_ops':0,
            'native_regions':0,'device_dispatch_ops':[],
            'operators':[{'node':n.name,'target':str(n.target),'execution':'deferred-metadata-specialization'} for n in calls],
        }
        self._record=dict(self._template)
        self._record.update({
            'specialization':'invocation-metadata',
            'template_operator_count':len(calls),
            'specialization_cache_size':options.cache_size,
            'specialization_region_cache_size':1,
            'active_specializations':[],
            'last_specialization':None,
            'specialization_error':None,
        })
        with state.lock:
            state.register_region(self)
            state.record(self._record)

    def _publish(self, entry=None, error=None):
        active=[{'id':item.identifier,'folded_shape_nodes':item.folded,
                 'reference_reason':item.reason,'graphs':item.state.graphs if item.state is not None else []}
                for item in self._cache.values()]
        with self.state.lock:
            self._record['active_specializations']=active
            self._record['specialization_error']=error
            if entry is not None:
                self._record['last_specialization']=entry.identifier
                if entry.state is not None and entry.state.graphs:
                    self._record.update(entry.state.graphs[-1])
                else:
                    self._record.update(self._template)
                    self._record['operators']=[dict(op,execution='device-dispatch') for op in self._template['operators']]
                    self._record['device_dispatch_ops']=[op['target'] for op in self._template['operators']]

    def _release_pending(self):
        if self._pending is not None:
            self._pending.close()
            self._pending=None

    def _evict(self, old_key=None):
        if old_key is None:old_key=next(iter(self._cache))
        old=self._cache[old_key]
        # Delete only after close succeeds; a failed release remains owned and
        # prevents allocating another specialization beyond the capacity bound.
        if old.state is not None:old.state.close()
        del self._cache[old_key]
        self.state.add('dynamic_cache_evictions')
        self._publish()

    def _failure(self, reason):
        self.state.add('dynamic_specialization_failures')
        self._publish(error=reason)
        if self.options.native=='required':
            raise NativeCoverageError(f'{self.phase}: dynamic native specialization unavailable: {reason}')

    def _reference(self, args, reason):
        self.state.add('dynamic_reference_runs')
        self.state.reason(self.phase,f'dynamic metadata specialization: {reason}')
        return self.reference(*args)

    def forward(self, *args):
        with self._lock:
            self.state.check_open()
            if self._closed:raise RuntimeError('dynamic native graph is closed')
            self._release_pending()
            try:
                metadata=(value_key(args),tuple((str(name),value_key(attribute(self.reference,name))) for name in self._attributes))
            except ShapeSpecializationError as exc:
                reason=str(exc);self._failure(reason)
                return self._reference(args,reason)
            key=(_stream_id(),torch.is_inference_mode_enabled(),metadata)
            entry=self._cache.get(key)
            if entry is not None and entry.state is not None and entry.state.closed:
                # A previous failed eviction can have released only part of
                # this graph. Finish its release before considering dispatch.
                self._evict(key)
                entry=None
            if entry is None:
                if len(self._cache)>=self.options.cache_size:self._evict()
                child=_SpecializationState(self.state)
                self._pending=child
                try:
                    concrete,folded=specialize(self.reference,args)
                    lowered=partition(concrete,child,self.phase,self._region_options)
                except (ShapeSpecializationError,NativeCoverageError) as exc:
                    reason=str(exc)
                    self._release_pending()
                    self._failure(reason)
                    entry=_Entry(self._next_identifier,self.reference,None,(),reason)
                except BaseException:
                    self._release_pending()
                    raise
                else:
                    entry=_Entry(self._next_identifier,lowered,child,folded)
                    self.state.add('dynamic_specializations')
                self._cache[key]=entry
                self._pending=None
                self._next_identifier+=1
            else:
                self._cache.move_to_end(key)
                self.state.add('dynamic_cache_hits')
            self._publish(entry,error=entry.reason)
            # Only planning above can select the original graph. In particular,
            # a failure after any kernel/RNG/mutation dispatch never retries it.
            if entry.reason is not None:return self._reference(args,entry.reason)
            return entry.graph(*args)

    def close(self):
        with self._lock:
            self._closed=True
            errors=[]
            try:self._release_pending()
            except Exception as exc:errors.append(exc)
            for key,entry in list(self._cache.items()):
                try:
                    if entry.state is not None:entry.state.close()
                except Exception as exc:errors.append(exc)
                else:del self._cache[key]
            self._publish()
            if errors:raise RuntimeError('failed to release dynamic native specializations') from errors[0]
