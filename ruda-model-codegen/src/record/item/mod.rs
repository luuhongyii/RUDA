pub(crate) mod codegen;
pub(crate) mod codegen_enum;
pub(crate) mod codegen_enum_full;
pub(crate) mod codegen_struct;
pub(crate) mod codegen_struct_full;
mod enum_deserialize;

pub(crate) fn container_type(ast: &syn::DeriveInput, kind: &str) -> syn::Result<String> {
    let mut name = ast.ident.to_string();
    for attr in &ast.attrs {
        if !attr.path().is_ident("ruda_record") { continue; }
        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("module") {
                name = meta.value()?.parse::<syn::LitStr>()?.value();
                Ok(())
            } else {
                Err(meta.error("Expected module"))
            }
        })?;
    }
    Ok(format!("{kind}:{name}"))
}

pub(crate) fn map_field(value: proc_macro2::TokenStream, name: &str, container: &str,
    paths: &crate::DerivePaths) -> proc_macro2::TokenStream {
    let model = &paths.model;
    quote::quote!({
        mapper.enter_module(#name, #container);
        let value = #model::record::Record::<B>::map_parameters(#value, mapper);
        mapper.exit_module(#name, #container);
        value
    })
}
