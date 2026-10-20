use proc_macro2::{Span, TokenStream};
use quote::{quote, quote_spanned};
use syn::{Token, spanned::Spanned};

use crate::ir::{
    expression::Expression,
    generate::expression::{bind_runtime_value, snapshot_value},
    paths::{frontend_type, prelude_type},
    scope::Context,
    statement::{DefineKind, Statement},
};

impl Statement {
    pub fn to_tokens(&self, context: &mut Context) -> TokenStream {
        match self {
            Statement::Local { variable, init } => {
                let ruda_type = frontend_type("RudaType");
                let name = &variable.name;
                let is_mut = variable.is_mut || init.as_deref().is_some_and(is_mut_owned);
                let mutable = variable.is_mut.then(|| quote![mut]);
                let is_const = init.as_ref().is_some_and(|it| it.is_const());
                let init = if is_mut {
                    if let Some(as_const) =
                        init.as_ref().and_then(|it| it.as_const_primitive(context))
                    {
                        let expand = frontend_type("NativeExpand");
                        Some(quote_spanned![as_const.span()=> #expand::from_lit(scope, #as_const)])
                    } else if let Some(as_const) = init.as_ref().and_then(|it| it.as_const(context))
                    {
                        Some(quote_spanned![as_const.span()=> #as_const.clone()])
                    } else {
                        init.as_ref().map(|it| it.to_tokens(context))
                    }
                } else {
                    init.as_ref().map(|init| {
                        init.as_const(context)
                            .unwrap_or_else(|| init.to_tokens(context))
                    })
                };
                let ty = variable.ty.as_ref().map(|ty| {
                    let mut ty = ty;
                    while let syn::Type::Reference(reference) = ty {
                        ty = &reference.elem;
                    }
                    quote_spanned! {
                        ty.span()=> :<#ty as #ruda_type>::ExpandType
                    }
                });

                let init = if let Some(syn::Type::Reference(reference)) = &variable.ty {
                    if !is_const {
                        init.map(|init| {
                            let into_mut = frontend_type("IntoMut");
                            let writable = reference.mutability.is_some();
                            quote! {{
                                let _init = #init;
                                #into_mut::expand_reborrow(_init.clone(), scope, #writable)
                            }}
                        })
                    } else { init }
                } else { init };
                let init = match (is_mut, init) {
                    (true, Some(init)) => {
                        let into_mut = frontend_type("IntoMut");
                        let init_ty =
                            quote_spanned![init.span()=> #into_mut::into_mut(_init, scope)];
                        Some(quote! {
                            {
                                let _init = #init;
                                #init_ty
                            }
                        })
                    }
                    (_, Some(init)) if !is_const => {
                        let snapshot = snapshot_value(quote![_init.clone()]);
                        Some(quote! {{
                            let _init = #init;
                            #snapshot
                        }})
                    }
                    (_, init) => init,
                };

                if let Some(mut init) = init {
                    if is_mut || !is_const {
                        let name_str = name.to_string();
                        let init_var = if context.debug_symbols {
                            let debug_var = frontend_type("debug_var_expand");

                            quote![
                                #debug_var(scope, #name_str, __init)
                            ]
                        } else {
                            quote![__init]
                        };
                        init = quote! {{
                            let __init = #init;
                            #init_var
                        }};
                    }

                    quote![let #mutable #name #ty = #init;]
                } else {
                    quote![let #mutable #name #ty;]
                }
            }
            Statement::Destructure { pattern, init, is_const } => {
                if *is_const {
                    let init = init.as_const(context).unwrap_or_else(|| init.to_tokens(context));
                    quote![let #pattern = #init;]
                } else {
                    let init = destructure_initializer(init, context);
                    bind_runtime_value(pattern, init)
                }
            }
            Statement::Define { name, kind, init } => {
                let value = init
                    .as_const(context)
                    .unwrap_or_else(|| init.to_tokens(context));
                let define_func = match kind {
                    DefineKind::Type => prelude_type("define_scalar"),
                    DefineKind::Size => prelude_type("define_size"),
                };
                let register = match kind {
                    DefineKind::Size => quote![register_size],
                    DefineKind::Type => quote![register_type],
                };
                quote! {
                    #define_func!(#name);
                    {
                        let __init = #value;
                        scope.#register::<#name>(__init);
                    }
                }
            }
            Statement::Expression {
                expression,
                terminated,
            } => {
                let terminator = terminated.then(|| Token![;](Span::call_site()));
                if let Some(as_const) = expression.as_const(context) {
                    quote![#as_const #terminator]
                } else {
                    let expression = expression.to_tokens(context);
                    quote![#expression #terminator]
                }
            }
            Statement::Verbatim { tokens } => tokens.clone(),
        }
    }
}

fn destructure_initializer(init: &Expression, context: &mut Context) -> TokenStream {
    if let Expression::Tuple { elements, .. } = init {
        let values = elements.iter().map(|element| destructure_initializer(element, context));
        quote![(#(#values,)*)]
    } else if let Expression::Reference { inner, writable } = init
        && init.is_const()
    {
        let value = destructure_initializer(inner, context);
        let into_mut = frontend_type("IntoMut");
        quote! {{
            let __ruda_pattern_value = #value;
            #into_mut::expand_reference(__ruda_pattern_value, scope, #writable)
        }}
    } else if let Some(value) = init.as_const(context) {
        let into_runtime = frontend_type("IntoRuntime");
        quote![#into_runtime::__expand_runtime_method(#value, scope)]
    } else {
        init.to_tokens(context)
    }
}

fn is_mut_owned(init: &Expression) -> bool {
    match init {
        Expression::Variable(var) => var.is_mut && !var.is_ref,
        Expression::FieldAccess { base, .. } => is_mut_owned(base),
        _ => false,
    }
}
