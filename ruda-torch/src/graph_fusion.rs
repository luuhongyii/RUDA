use super::*;
use super::graph_contract::NodeSpec;
use std::collections::BTreeMap;

#[ruda]
pub(super) fn rounded(value: f32, #[comptime] dtype: u32) -> f32 {
    let mut result = value;
    if comptime!(dtype == 1) { result = f32::cast_from(f16::cast_from(value)); }
    if comptime!(dtype == 2) { result = f32::cast_from(bf16::cast_from(value)); }
    if comptime!(dtype == 8) { result = f32::cast_from(u8::cast_from(value)); }
    if comptime!(dtype == 12 || dtype == 13 || dtype == 15) {
        result = crate::foreach_copy::decode_fp8(u32::cast_from(crate::foreach_copy::encode_fp8(value,dtype)),dtype);
    }
    result
}

#[ruda]
pub(super) fn evaluate(x: f32, y: f32, scalar: f32, #[comptime] op: u32, #[comptime] dtype: u32) -> f32 {
    let mut value = x;
    if comptime!(op == 1 || op == 116) { value = x + scalar * y; }
    else if comptime!(op == 2 || op == 117) { value = x * y; }
    else if comptime!(op == 3) { if x < 0.0 { value = 0.0; } }
    else if comptime!(op == 4) { if y <= rounded(scalar, dtype) { value = 0.0; } }
    else if comptime!(op == 8 || op == 118) { value = x / y; }
    else if comptime!(op == 9) { value = x.exp(); }
    else if comptime!(op == 10) { value = x.ln(); }
    else if comptime!(op == 11) { value = x.sqrt(); }
    else if comptime!(op == 12) { value = 1.0 / x.sqrt(); }
    else if comptime!(op == 13) { value = 1.0 / (1.0 + (-x).exp()); }
    else if comptime!(op == 14) { value = x / (1.0 + (-x).exp()); }
    else if comptime!(op == 15) {
        let sigmoid = 1.0f32 / (1.0f32 + (-y).exp());
        value = x * (sigmoid * (1.0f32 + y * (1.0f32 - sigmoid)));
    }
    else if comptime!(op == 16) { value = x * y * (1.0f32 - y); }
    else if comptime!(op == 17) { value = x.tanh(); }
    else if comptime!(op == 18) { value = x * (1.0f32 - y * y); }
    else if comptime!(op == 19) { value = x.sin(); }
    else if comptime!(op == 20) { value = x.cos(); }
    else if comptime!(op == 21) { value = x.abs(); }
    else if comptime!(op == 22) {
        value = 0.0;
        if x > 0.0 { value = 1.0; } else if x < 0.0 { value = -1.0; }
    }
    else if comptime!(op == 23) { value = x.floor(); }
    else if comptime!(op == 24) { value = x.ceil(); }
    else if comptime!(op == 25) { value = x.trunc(); }
    else if comptime!(op == 27) { value = x.recip(); }
    else if comptime!(op == 35) { value = x.log1p(); }
    else if comptime!(op == 36) { value = x.sinh(); }
    else if comptime!(op == 37) { value = x.cosh(); }
    else if comptime!(op == 38) { value = x.asinh(); }
    else if comptime!(op == 39) { value = x.acosh(); }
    else if comptime!(op == 40) { value = x.atanh(); }
    else if comptime!(op == 43 || op == 45) {
        let mut clipped = x + 3.0f32;
        if clipped < 0.0 { clipped = 0.0; }
        if clipped > 6.0 { clipped = 6.0; }
        if comptime!(op == 43) { value = clipped / 6.0f32; }
        else { value = x * clipped / 6.0f32; }
    }
    else if comptime!(op == 101) { value = rounded(x / (1.0f32 + (-x).exp()), dtype) * y; }
    else if comptime!(op == 108) { value = -x; }
    else if comptime!(op == 109) { value = x + scalar; }
    else if comptime!(op == 110) { value = x * scalar; }
    else if comptime!(op == 111) { value = x / scalar; }
    else if comptime!(op == 114 && dtype == 3) {value=f32::cast_from(x!=0.0);}
    else if comptime!(op == 121) { value = x.powf(y); }
    else if comptime!(op == 129) {
        if scalar == 0.0 { value = 1.0; }
        else if scalar == 1.0 { value = x; }
        else if scalar == 2.0 { value = x * x; }
        else if scalar == 3.0 { value = x * x * x; }
        else if scalar == 0.5 { value = x.sqrt(); }
        else if scalar == -0.5 { value = 1.0 / x.sqrt(); }
        else if scalar == -1.0 { value = x.recip(); }
        else { value = x.powf(rounded(scalar, dtype)); }
    }
    else if comptime!(op == 146) {
        if y!=0.0 {
            value=scalar;
            if comptime!(dtype==3) {value=f32::cast_from(scalar!=0.0);}
        }
    }
    else if comptime!(op >= 148 && op <= 159) {
        let mut other=y;if comptime!(op>=154) {other=scalar;}
        let comparison=comptime!((op-148)%6);
        if comptime!(comparison==0) {value=f32::cast_from(x==other);}
        else if comptime!(comparison==1) {value=f32::cast_from(x!=other);}
        else if comptime!(comparison==2) {value=f32::cast_from(x<other);}
        else if comptime!(comparison==3) {value=f32::cast_from(x<=other);}
        else if comptime!(comparison==4) {value=f32::cast_from(x>other);}
        else {value=f32::cast_from(x>=other);}
    }
    else if comptime!(op>=164 && op<=171) {
        if comptime!(op<=167 || dtype==3) {
            let mode=comptime!((op-164)%4);
            if comptime!(mode==0) {value=f32::cast_from(x==0.0);}
            else if comptime!(mode==1) {value=f32::cast_from(x!=0.0 && y!=0.0);}
            else if comptime!(mode==2) {value=f32::cast_from(x!=0.0 || y!=0.0);}
            else {value=f32::cast_from((x!=0.0)!=(y!=0.0));}
        } else {
            let a=u32::cast_from(x);let b=u32::cast_from(y);
            if comptime!(op==168) {value=f32::cast_from(a^255u32);}
            else if comptime!(op==169) {value=f32::cast_from(a&b);}
            else if comptime!(op==170) {value=f32::cast_from(a|b);}
            else {value=f32::cast_from(a^b);}
        }
    }
    else if comptime!(op >= 187 && op <= 188) {
        value = crate::operation_pointwise::activation_value(x, y, scalar, 20.0, comptime!(op-187));
    }
    else if comptime!(op >= 174 && op <= 186) {
        let activation = comptime!(if op <= 181 { op - 172 } else { op - 171 });
        value = crate::operation_pointwise::activation_value(x, y, scalar, 0.0, activation);
    }
    rounded(value, dtype)
}

#[ruda]
pub(super) fn address<F: Numeric>(input: &Tensor<F>, position: usize, #[comptime] shape: Vec<usize>) -> usize {
    let mut remaining = position;
    let mut result = 0usize;
    #[unroll]
    for reverse in 0usize..shape.len() {
        let dim = comptime!(shape.len() - 1 - reverse);
        let coordinate = remaining % comptime!(shape[dim]);
        remaining /= comptime!(shape[dim]);
        if dim + input.rank() >= shape.len() {
            let axis = dim + input.rank() - shape.len();
            if input.shape(axis) != 1 { result += coordinate * input.stride(axis); }
        }
    }
    result
}

#[ruda(launch)]
fn pointwise_chain(
    inputs32: &Sequence<Tensor<f32>>, inputs16: &Sequence<Tensor<f16>>, inputsbf: &Sequence<Tensor<bf16>>,
    inputs8: &Sequence<Tensor<u8>>,
    outputs32: &mut Sequence<Tensor<f32>>, outputs16: &mut Sequence<Tensor<f16>>, outputsbf: &mut Sequence<Tensor<bf16>>,
    outputs8: &mut Sequence<Tensor<u8>>,
    length: usize, #[comptime] shape: Vec<usize>,
    #[comptime] input_types: Vec<u32>, #[comptime] input_slots: Vec<usize>,
    #[comptime] codes: Vec<u32>, #[comptime] left: Vec<usize>, #[comptime] right: Vec<usize>, #[comptime] third: Vec<usize>,
    #[comptime] scalar_bits: Vec<u32>, #[comptime] output_types: Vec<u32>, #[comptime] output_slots: Vec<usize>,
    #[comptime] stores: Vec<bool>,
) {
    let position = ABSOLUTE_POS as usize;
    if position < length {
        let mut values = Array::<f32>::new(input_types.len() + codes.len());
        #[unroll]
        for i in 0usize..input_types.len() {
            let slot = comptime!(input_slots[i]);
            if comptime!(input_types[i] == 0) {
                let input = inputs32.index(slot);
                values[i] = input[address(input, position, shape.clone())];
            } else if comptime!(input_types[i] == 1) {
                let input = inputs16.index(slot);
                values[i] = f32::cast_from(input[address(input, position, shape.clone())]);
            } else if comptime!(input_types[i] == 2) {
                let input = inputsbf.index(slot);
                values[i] = f32::cast_from(input[address(input, position, shape.clone())]);
            } else {
                let input = inputs8.index(slot);
                let raw=input[address(input, position, shape.clone())];
                if comptime!(input_types[i]==3 || input_types[i]==8) {values[i]=f32::cast_from(raw);}
                else {values[i] = crate::foreach_copy::decode_fp8(u32::cast_from(raw),comptime!(input_types[i]));}
            }
        }
        #[unroll]
        for i in 0usize..codes.len() {
            let x = values[comptime!(left[i])];
            let y = values[comptime!(right[i])];
            let mut scalar=f32::from_bits(comptime!(scalar_bits[i]));
            if comptime!(codes[i]>=154 && codes[i]<=159) {
                let dtype=comptime!(if left[i]<input_types.len() {input_types[left[i]]} else {output_types[left[i]-input_types.len()]});
                scalar=rounded(scalar,dtype);
            }
            let mut value = 0.0f32;
            if comptime!(codes[i]==147) {
                value=values[comptime!(third[i])];if x!=0.0 {value=y;}
            } else {value=evaluate(x, y, scalar, comptime!(codes[i]), comptime!(output_types[i]));}
            values[input_types.len() + i] = value;
            if comptime!(stores[i]) {
                let slot = comptime!(output_slots[i]);
                if comptime!(output_types[i] == 0) { outputs32.index_mut(slot)[position] = value; }
                else if comptime!(output_types[i] == 1) { outputs16.index_mut(slot)[position] = f16::cast_from(value); }
                else if comptime!(output_types[i] == 2) { outputsbf.index_mut(slot)[position] = bf16::cast_from(value); }
                else if comptime!(output_types[i]==3 || output_types[i]==8) {outputs8.index_mut(slot)[position]=u8::cast_from(value);}
                else { outputs8.index_mut(slot)[position] = crate::foreach_copy::encode_fp8(value,comptime!(output_types[i])); }
            }
        }
    }
}

#[ruda(launch)]
fn pointwise_chain_vector<V: Size>(
    inputs32: &Sequence<Tensor<f32>>, inputs16: &Sequence<Tensor<f16>>, inputsbf: &Sequence<Tensor<bf16>>,
    inputs8: &Sequence<Tensor<u8>>,
    outputs32: &mut Sequence<Tensor<f32>>, outputs16: &mut Sequence<Tensor<f16>>, outputsbf: &mut Sequence<Tensor<bf16>>,
    outputs8: &mut Sequence<Tensor<u8>>,
    length: usize, #[comptime] periods: Vec<usize>,
    #[comptime] input_types: Vec<u32>, #[comptime] input_slots: Vec<usize>,
    #[comptime] codes: Vec<u32>, #[comptime] left: Vec<usize>, #[comptime] right: Vec<usize>, #[comptime] third: Vec<usize>,
    #[comptime] scalar_bits: Vec<u32>, #[comptime] output_types: Vec<u32>, #[comptime] output_slots: Vec<usize>,
    #[comptime] stores: Vec<bool>,
) {
    let position = ABSOLUTE_POS as usize;
    if position < length / V::value() {
        let mut values = Array::<Vector<f32, V>>::new(input_types.len() + codes.len());
        #[unroll]
        for i in 0usize..input_types.len() {
            let slot = comptime!(input_slots[i]);
            let period = comptime!(periods[i]);
            if comptime!(input_types[i] == 0) {
                let input = inputs32.index(slot);
                if comptime!(period == 1) { values[i] = Vector::new(input[0usize]); }
                else {
                    let packed = input.slice(0usize, input.len()).into_vectorized().with_vector_size::<V>();
                    values[i] = packed[position % (period / V::value())];
                }
            } else if comptime!(input_types[i] == 1) {
                let input = inputs16.index(slot);
                if comptime!(period == 1) { values[i] = Vector::new(f32::cast_from(input[0usize])); }
                else {
                    let packed = input.slice(0usize, input.len()).into_vectorized().with_vector_size::<V>();
                    values[i] = Vector::cast_from(packed[position % (period / V::value())]);
                }
            } else if comptime!(input_types[i] == 2) {
                let input = inputsbf.index(slot);
                if comptime!(period == 1) { values[i] = Vector::new(f32::cast_from(input[0usize])); }
                else {
                    let packed = input.slice(0usize, input.len()).into_vectorized().with_vector_size::<V>();
                    values[i] = Vector::cast_from(packed[position % (period / V::value())]);
                }
            } else {
                let input = inputs8.index(slot);
                let mut decoded = Vector::<f32,V>::empty();
                if comptime!(period == 1) {
                    if comptime!(input_types[i]==3 || input_types[i]==8) {decoded=Vector::new(f32::cast_from(input[0usize]));}
                    else {decoded = Vector::new(crate::foreach_copy::decode_fp8(u32::cast_from(input[0usize]),comptime!(input_types[i])));}
                } else {
                    let packed = input.slice(0usize,input.len()).into_vectorized().with_vector_size::<V>();
                    let raw = packed[position % (period / V::value())];
                    #[unroll] for lane in 0usize..V::value() {
                        if comptime!(input_types[i]==3 || input_types[i]==8) {decoded[lane]=f32::cast_from(raw[lane]);}
                        else {decoded[lane] = crate::foreach_copy::decode_fp8(u32::cast_from(raw[lane]),comptime!(input_types[i]));}
                    }
                }
                values[i] = decoded;
            }
        }
        #[unroll]
        for i in 0usize..codes.len() {
            let x = values[comptime!(left[i])];
            let y = values[comptime!(right[i])];
            let mut value = Vector::<f32, V>::empty();
            let mut scalar=f32::from_bits(comptime!(scalar_bits[i]));
            if comptime!(codes[i]>=154 && codes[i]<=159) {
                let dtype=comptime!(if left[i]<input_types.len() {input_types[left[i]]} else {output_types[left[i]-input_types.len()]});
                scalar=rounded(scalar,dtype);
            }
            #[unroll] for lane in 0usize..V::value() {
                if comptime!(codes[i]==147) {
                    let other=values[comptime!(third[i])];
                    value[lane]=other[lane];if x[lane]!=0.0 {value[lane]=y[lane];}
                } else {
                    value[lane] = evaluate(x[lane], y[lane], scalar,
                        comptime!(codes[i]), comptime!(output_types[i]));
                }
            }
            values[input_types.len() + i] = value;
            if comptime!(stores[i]) {
            let slot = comptime!(output_slots[i]);
            if comptime!(output_types[i] == 0) {
                let output = outputs32.index_mut(slot);
                let mut packed = output.slice_mut(0usize, output.len()).into_vectorized().with_vector_size::<V>();
                packed[position] = value;
            } else if comptime!(output_types[i] == 1) {
                let output = outputs16.index_mut(slot);
                let mut packed = output.slice_mut(0usize, output.len()).into_vectorized().with_vector_size::<V>();
                packed[position] = Vector::cast_from(value);
            } else if comptime!(output_types[i] == 2) {
                let output = outputsbf.index_mut(slot);
                let mut packed = output.slice_mut(0usize, output.len()).into_vectorized().with_vector_size::<V>();
                packed[position] = Vector::cast_from(value);
            } else {
                let mut encoded = Vector::<u8,V>::empty();
                #[unroll] for lane in 0usize..V::value() {
                    if comptime!(output_types[i]==3 || output_types[i]==8) {encoded[lane]=u8::cast_from(value[lane]);}
                    else {encoded[lane] = crate::foreach_copy::encode_fp8(value[lane],comptime!(output_types[i]));}
                }
                let output = outputs8.index_mut(slot);
                let mut packed = output.slice_mut(0usize,output.len()).into_vectorized().with_vector_size::<V>();
                packed[position] = encoded;
            }
            }
        }
    }
}

pub(super) fn vector_period(input: &View, shape: &[usize]) -> Option<usize> {
    if !input.contiguous() || input.shape.len() > shape.len() || input.len == 0 { return None; }
    if input.len == 1 { return Some(1); }
    if input.len % 4 != 0 || input.handle.offset_start.unwrap_or(0) % (4 * element_bytes(input.dtype)) as u64 != 0 {
        return None;
    }
    let padding = shape.len() - input.shape.len();
    let mut repeated = false;
    for d in (0..shape.len()).rev() {
        let size = if d < padding { 1 } else { input.shape[d-padding] };
        if size == 1 && shape[d] != 1 { repeated = true; }
        else if size != shape[d] || repeated && size > 1 { return None; }
    }
    Some(input.len)
}

pub(super) fn overlaps(a: &View, b: &View) -> bool {
    if a.handle.memory.clone().binding().id() != b.handle.memory.clone().binding().id() { return false; }
    let span = |v: &View| v.shape.iter().zip(&v.strides).try_fold(1u64, |n, (&size, &stride)|
        n.checked_add((size.saturating_sub(1) as u64).checked_mul(stride as u64)?))
        .and_then(|n| n.checked_mul(element_bytes(v.dtype) as u64));
    let left = a.handle.offset_start.unwrap_or(0);
    let right = b.handle.offset_start.unwrap_or(0);
    match (span(a).and_then(|n|left.checked_add(n)),span(b).and_then(|n|right.checked_add(n))) {
        (Some(a_end),Some(b_end)) => left < b_end && right < a_end,
        _ => true,
    }
}

pub(super) fn supported(op: u32) -> bool {
    matches!(op, 0..=4 | 8..=25 | 27 | 35..=40 | 43 | 45 | 101 | 108..=111 | 114..=118 | 121 | 129 | 146..=159 | 164..=171 | 174..=188)
}

fn operands(node:&NodeSpec)->Vec<usize> {
    let mut ids=vec![node.a as usize,node.b as usize];
    if node.op==147 {ids.push(node.scalar as usize);}
    ids
}

pub(super) fn prepare(c: &ComputeClient<CudaRuntime>, tensors: &[View], nodes: &[NodeSpec],
    inputs: usize, start: usize) -> Option<(usize, PreparedKernel<CudaRuntime>)>
{
    let first=&tensors[inputs+start];let mut layouts=Vec::new();
    for (index,node) in nodes.iter().enumerate().skip(start).take(16) {
        let out=&tensors[inputs+index];
        if out.shape!=first.shape || out.len==0 || out.dtype>2 {break;}
        if matches!(node.op,140|212) && (node.a as usize)<inputs+start {
            let source=&tensors[node.a as usize];let code=node.scalar as usize;let axis=code&7;
            let begin=if node.op==212 {(code>>3)&2047}else {code>>3};
            let step=if node.op==212 {code>>14}else {1};
            let mut strides=source.strides.clone();strides[axis]*=step;
            let offset=begin*source.strides[axis]*element_bytes(source.dtype);
            layouts.push((index,View{handle:source.handle.clone().offset_start(offset as u64),
                shape:out.shape.clone(),strides,len:out.len,dtype:source.dtype}));
        } else if !supported(node.op) {break;}
    }
    if layouts.is_empty() {return prepare_chain(c,tensors,nodes,inputs,start,tensors.len());}
    let mut views=tensors.iter().map(|v|View{handle:v.handle.clone(),shape:v.shape.clone(),
        strides:v.strides.clone(),len:v.len,dtype:v.dtype}).collect::<Vec<_>>();
    let mut rewritten=nodes.to_vec();
    for (index,view) in layouts {
        let input=views.len() as u32;views.push(view);
        rewritten[index]=NodeSpec{op:0,a:input,b:input,scalar:0.0};
    }
    prepare_chain(c,&views,&rewritten,inputs,start,tensors.len())
}

fn prepare_chain(c: &ComputeClient<CudaRuntime>, tensors: &[View], nodes: &[NodeSpec],
    inputs: usize, start: usize, original_tensors:usize) -> Option<(usize, PreparedKernel<CudaRuntime>)>
{
    let first = &tensors[inputs + start];
    let floating = |dtype| matches!(dtype,0..=3|8|12|13|15);
    if first.len == 0 || !floating(first.dtype) || !first.contiguous() { return None; }
    let mut end = start;
    while end < nodes.len() && end - start < 16 {
        let node = &nodes[end];
        let out = &tensors[inputs + end];
        if !supported(node.op) || !floating(out.dtype) || out.shape != first.shape || !out.contiguous()
            || operands(node).iter().any(|&id| id >= tensors.len() || !floating(tensors[id].dtype))
        { break; }
        // Same-format copies preserve every FP8 payload bit without decoding it.
        if matches!(out.dtype,12|13|15) && (node.op != 114 || tensors[node.a as usize].dtype == out.dtype) {break;}
        if out.dtype==8 && node.op==114 && tensors[node.a as usize].dtype!=8 {break;}
        end += 1;
    }
    if end == start || end - start < 2 && !(174..=188).contains(&nodes[start].op) { return None; }
    let mut external = Vec::new();
    let mut indices = BTreeMap::new();
    for node in &nodes[start..end] {
        for id in operands(node) {
            if (id < inputs + start || id>=original_tensors) && !indices.contains_key(&id) {
                indices.insert(id, external.len()); external.push(id);
            }
        }
    }
    for index in start..end { indices.insert(inputs + index, external.len() + index - start); }
    // Internal whole-buffer reuse writes the same element in node order. External aliases can race with broadcast reads.
    for i in inputs+start..inputs+end {
        if external.iter().any(|&j|overlaps(&tensors[i],&tensors[j])) {
            return if i>inputs+start {prepare_chain(c,tensors,&nodes[..i-inputs],inputs,start,original_tensors)} else {None};
        }
        if (inputs+start..i).any(|j|overlaps(&tensors[i],&tensors[j])
            && (tensors[i].handle.offset_start!=tensors[j].handle.offset_start
                || tensors[i].shape!=tensors[j].shape || tensors[i].dtype!=tensors[j].dtype)) {return None;}
    }
    let periods = external.iter().map(|&i|vector_period(&tensors[i],&first.shape)).collect::<Option<Vec<_>>>();
    let vectorized = periods.is_some() && first.len % 4 == 0 && external.len() + end - start <= 24
        && (inputs+start..inputs+end).all(|i|tensors[i].handle.offset_start.unwrap_or(0)
            % (4 * element_bytes(tensors[i].dtype)) as u64 == 0);
    let mut vector=4usize;
    if vectorized && end-start<=8 && external.len()+end-start<=16 {
        let views=external.iter().copied().filter(|&i|tensors[i].len!=1)
            .chain(inputs+start..inputs+end).map(|i|&tensors[i]).collect::<Vec<_>>();
        vector=super::row_cached::memory_vector_size(c,first.len,&views,8).max(4);
        if periods.as_ref()?.iter().any(|&period|period!=1 && period%vector!=0) {vector=4;}
    }
    let mut in_counts = [0usize; 4]; let mut out_counts = [0usize; 4];
    let input_types: Vec<_> = external.iter().map(|&i| tensors[i].dtype).collect();
    let input_slots: Vec<_> = input_types.iter().map(|&t| { let bucket=t.min(3) as usize;let slot = in_counts[bucket]; in_counts[bucket] += 1; slot }).collect();
    let output_types: Vec<_> = (start..end).map(|i| tensors[inputs+i].dtype).collect();
    let mut unique_outputs=Vec::<usize>::new();let mut unique_slots=Vec::new();let mut output_slots=Vec::new();
    for i in inputs+start..inputs+end {
        let previous=unique_outputs.iter().position(|&j|tensors[i].handle.memory.clone().binding().id()==tensors[j].handle.memory.clone().binding().id()
            && tensors[i].handle.offset_start==tensors[j].handle.offset_start && tensors[i].dtype==tensors[j].dtype && tensors[i].shape==tensors[j].shape);
        if let Some(index)=previous {output_slots.push(unique_slots[index]);}
        else {
            let dtype=tensors[i].dtype.min(3) as usize;let slot=out_counts[dtype];out_counts[dtype]+=1;
            unique_outputs.push(i);unique_slots.push(slot);output_slots.push(slot);
        }
    }
    let codes = nodes[start..end].iter().map(|n| n.op).collect();
    let left = nodes[start..end].iter().map(|n| indices[&(n.a as usize)]).collect();
    let right = nodes[start..end].iter().map(|n| indices[&(n.b as usize)]).collect();
    let third = nodes[start..end].iter().map(|n|if n.op==147 {indices[&(n.scalar as usize)]}else {0}).collect();
    let scalars = nodes[start..end].iter().map(|n| n.scalar.to_bits()).collect();
    let stores=(0..end-start).map(|i|!(i+1..end-start).any(|j|output_types[i]==output_types[j] && output_slots[i]==output_slots[j])).collect();
    macro_rules! arguments { ($ids:expr, $dtype:expr, $ty:ty) => {
        ($ids).filter(|&i| tensors[i].dtype.min(3) == $dtype).map(|i| tensors[i].arg()).collect::<SequenceArg<CudaRuntime, Tensor<$ty>>>()
    }; }
    macro_rules! vector_chain {($v:ty)=>{unsafe {pointwise_chain_vector::prepare::<$v,CudaRuntime>(c,
        RudaCount::Static(u32::try_from((first.len/vector).div_ceil(128)).ok()?,1,1),RudaDim::new_1d(128),
        arguments!(external.iter().copied(),0,f32),arguments!(external.iter().copied(),1,f16),arguments!(external.iter().copied(),2,bf16),
        arguments!(external.iter().copied(),3,u8),
        arguments!(unique_outputs.iter().copied(),0,f32),arguments!(unique_outputs.iter().copied(),1,f16),arguments!(unique_outputs.iter().copied(),2,bf16),
        arguments!(unique_outputs.iter().copied(),3,u8),
        first.len,periods.unwrap(),input_types,input_slots,codes,left,right,third,scalars,output_types,output_slots,stores)}};}
    let kernel = if vectorized {if vector==8 {vector_chain!(Const<8>)}else {vector_chain!(Const<4>)}} else { unsafe { pointwise_chain::prepare::<CudaRuntime>(c,
        RudaCount::Static(u32::try_from(first.len.div_ceil(128)).ok()?, 1, 1), RudaDim::new_1d(128),
        arguments!(external.iter().copied(), 0, f32), arguments!(external.iter().copied(), 1, f16), arguments!(external.iter().copied(), 2, bf16),
        arguments!(external.iter().copied(),3,u8),
        arguments!(unique_outputs.iter().copied(), 0, f32), arguments!(unique_outputs.iter().copied(), 1, f16), arguments!(unique_outputs.iter().copied(), 2, bf16),
        arguments!(unique_outputs.iter().copied(),3,u8),
        first.len, first.shape.clone(), input_types, input_slots, codes, left, right, third, scalars, output_types, output_slots,stores) }};
    Some((end-start, kernel))
}
