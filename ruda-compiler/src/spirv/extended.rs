use rspirv::{dr::{Builder, Instruction, Operand}, spirv::{Op, Word}};

pub(super) fn import(builder: &mut Builder, name: &str) -> Word {
    if let Some(id) = builder.module_ref().ext_inst_imports.iter().find_map(|i| {
        (i.operands.first() == Some(&Operand::LiteralString(name.into()))).then_some(i.result_id).flatten()
    }) { return id; }
    builder.ext_inst_import(name)
}

pub(super) fn string(builder: &mut Builder, text: &str) -> Word {
    if let Some(id) = builder.module_ref().debug_string_source.iter().find_map(|i| {
        (i.class.opcode == Op::String && i.operands.first() == Some(&Operand::LiteralString(text.into())))
            .then_some(i.result_id).flatten()
    }) { return id; }
    builder.string(text)
}

pub(super) fn global(builder: &mut Builder, opcode: u32, args: impl IntoIterator<Item = Word>, id: Option<Word>) -> Word {
    let set = import(builder, "NonSemantic.Shader.DebugInfo.100");
    let void = builder.type_void();
    let mut operands = vec![Operand::IdRef(set), Operand::LiteralExtInstInteger(opcode)];
    operands.extend(args.into_iter().map(Operand::IdRef));
    if id.is_none() {
        if let Some(id) = builder.module_ref().types_global_values.iter().find_map(|i| {
            (i.class.opcode == Op::ExtInst && i.result_type == Some(void) && i.operands == operands)
                .then_some(i.result_id).flatten()
        }) { return id; }
    }
    let id = id.unwrap_or_else(|| builder.id());
    builder.module_mut().types_global_values.push(Instruction::new(Op::ExtInst, Some(void), Some(id), operands));
    id
}

pub(super) fn local(builder: &mut Builder, set: &str, opcode: u32, args: impl IntoIterator<Item = Word>) -> Result<(), rspirv::dr::Error> {
    let set = import(builder, set);
    let void = builder.type_void();
    builder.ext_inst(void, None, set, opcode, args.into_iter().map(Operand::IdRef)).map(|_| ())
}
