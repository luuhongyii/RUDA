use proc_macro2::{Span, TokenStream};
use quote::{format_ident, quote, quote_spanned};
use syn::{
    Ident, Member, Pat, PatIdent, PatPath, PatStruct, PatTupleStruct, Path,
    PathArguments, parse_quote, spanned::Spanned,
};

use crate::ir::{
    expression::{Block, Expression, MatchArm},
    operator::Operator,
    paths::{frontend_path, frontend_type, prelude_type},
    scope::Context,
};

macro_rules! error {
    ($span:expr, $fmt:literal $(,$args:expr)*) => {
        syn::Error::new($span, format!($fmt $(,$args)*)).into_compile_error()
    };
}

impl Expression {
    pub fn to_tokens(&self, context: &mut Context) -> TokenStream {
        match self {
            Expression::Binary { left, operator, right, .. }
                if operator.is_assign() && matches!(**left, Expression::Unary { operator: Operator::Deref, .. }) =>
            {
                let Expression::Unary { input, .. } = &**left else { unreachable!() };
                let reference = input.to_tokens(context);
                let right = right.to_tokens(context);
                let into_mut = frontend_type("IntoMut");
                let frontend = frontend_path();
                let name = operator.op_name();
                let op = format_ident!("{}", name.strip_suffix("_assign_op").unwrap());
                quote! {{
                    let _reference = #reference;
                    let _right = #right;
                    let _left = #into_mut::expand_deref(_reference.clone(), scope);
                    let _value = #frontend::#op::expand(scope, _left.into(), _right.into());
                    #into_mut::expand_deref_assign(_reference.clone(), scope, _value.into());
                }}
            }
            Expression::Binary {
                left,
                operator,
                right,
                span,
                ..
            } if operator.is_assign() && matches!(**left, Expression::Index { .. }) => {
                let elem = frontend_type("NativeExpand");
                let frontend_path = frontend_path();
                let (array, index) = left.as_index().unwrap();
                let array = array.to_tokens(context);
                let index = index
                    .as_const(context)
                    .map(|as_const| quote![#elem::from_lit(scope, #as_const)])
                    .unwrap_or_else(|| index.to_tokens(context));
                let right = right
                    .as_const(context)
                    .map(|as_const| quote![#elem::from_lit(scope, #as_const)])
                    .unwrap_or_else(|| right.to_tokens(context));
                let op = format_ident!("{}", operator.array_op_name());
                let expand = with_span(
                    context,
                    *span,
                    quote![#frontend_path::#op::expand(scope, _array, _index.into(), _value.into())],
                );
                quote! {
                    {
                        let _array = #array;
                        let _index = #index;
                        let _value = #right;
                        #expand
                    }
                }
            }
            Expression::Binary {
                left,
                operator,
                right,
                span,
                ..
            } => {
                let frontend_path = frontend_path();
                let op = format_ident!("{}", operator.op_name());
                let left = left.to_tokens(context);
                let right = right.to_tokens(context);
                let expand = with_span(
                    context,
                    *span,
                    quote![#frontend_path::#op::expand(scope, _lhs.into(), _rhs.into())],
                );
                quote! {
                    {
                        let _lhs = #left;
                        let _rhs = #right;
                        #expand
                    }
                }
            }
            Expression::Unary {
                input,
                operator: Operator::Deref,
                ..
            } => {
                let input = input.to_tokens(context);
                let into_mut = frontend_type("IntoMut");
                quote! {{ let _reference = #input; #into_mut::expand_deref(_reference.clone(), scope) }}
            }
            Expression::Unary {
                input,
                operator,
                span,
                ..
            } => {
                let frontend_path = frontend_path();
                let input = input.to_tokens(context);
                let op = format_ident!("{}", operator.op_name());
                let expand = with_span(
                    context,
                    *span,
                    quote![#frontend_path::#op::expand(scope, _inner.into())],
                );
                quote! {
                    {
                        let _inner = #input;
                        #expand
                    }
                }
            }
            Expression::Keyword { name } => {
                quote![#name::expand(scope)]
            }
            Expression::Variable(var) => {
                if var.is_const {
                    let name = &var.name;
                    let expand_elem = frontend_type("NativeExpand");
                    quote![#expand_elem::from_lit(scope, #name)]
                } else {
                    let name = &var.name;
                    if var.try_consume(context) {
                        quote![#name]
                    } else {
                        quote![#name.clone()]
                    }
                }
            }
            Expression::FieldAccess { base, field, .. } => {
                let base = base
                    .as_const(context)
                    .unwrap_or_else(|| base.to_tokens(context));
                quote![#base.#field.clone()]
            }
            Expression::Literal { value, .. } => {
                let expand_elem = frontend_type("NativeExpand");
                quote![#expand_elem::from_lit(scope, #value)]
            }

            Expression::Assignment { left, right, .. }
                if matches!(**left, Expression::Unary { operator: Operator::Deref, .. }) =>
            {
                let Expression::Unary { input, .. } = &**left else { unreachable!() };
                let reference = input.to_tokens(context);
                let right = right.to_tokens(context);
                let into_mut = frontend_type("IntoMut");
                quote! {{
                    let _value = #right;
                    let _reference = #reference;
                    #into_mut::expand_deref_assign(_reference.clone(), scope, _value.into());
                }}
            }
            Expression::Assignment { left, right, .. }
                if matches!(**left, Expression::Index { .. }) =>
            {
                let (array, index) = left.as_index().unwrap();
                let array = array.to_tokens(context);
                let index = index.to_tokens(context);
                let right = right.to_tokens(context);
                quote! {
                    {
                        let _array = #array;
                        let _index = #index;
                        let _value = #right;
                        _array.expand_index_mut(scope, _index.into(), _value.into())
                    }
                }
            }
            Expression::Assignment { left, right, .. } => {
                let right = right.to_tokens(context);
                let left = left.to_tokens(context);
                quote! {
                    {
                        let _value = #right;
                        #left.expand_assign(scope, _value.into())
                    }
                }
            }
            Expression::Index { expr, index, span } => {
                let expr = expr.to_tokens(context);
                let index = index.to_tokens(context);
                let expand = with_span(
                    context,
                    *span,
                    quote![_array.expand_index(scope, _index.into())],
                );
                quote! {
                    {
                        let _array = #expr;
                        let _index = #index;
                        #expand
                    }
                }
            }
            Expression::FunctionCall {
                func,
                args,
                associated_type: None,
                span,
                ..
            } => {
                let (args, arg_names) = map_args(args, context);
                let (generics, path) = split_generics(func, context);
                let constructor = matches!(&**func, Expression::Path { path, .. }
                    if path.segments.last().is_some_and(|segment|
                        segment.ident.to_string().chars().next().is_some_and(char::is_uppercase)));
                let invocation = if constructor {
                    quote![#path #generics ::expand(scope, #(#arg_names),*)]
                } else {
                    quote![#path::expand #generics(scope, #(#arg_names),*)]
                };
                let call = with_debug_call(
                    context,
                    *span,
                    invocation,
                );

                quote_spanned! {*span=>
                    {
                        #(#args)*
                        #call
                    }
                }
            }
            Expression::CompilerIntrinsic { func, args } => {
                let (args, arg_names) = map_args(args, context);
                let mut path = func.clone();
                let generics = core::mem::replace(
                    &mut path.segments.last_mut().unwrap().arguments,
                    PathArguments::None,
                );
                quote! {
                    {
                        #(#args)*
                        #path::expand #generics(scope, #(#arg_names),*)
                    }
                }
            }
            Expression::FunctionCall {
                args,
                associated_type: Some((ty_path, qself, func)),
                span,
                ..
            } => {
                let ty_path = if let Some(qself) = qself {
                    let ty = &qself.ty;
                    quote![<#ty as #ty_path>]
                } else {
                    quote![#ty_path]
                };

                let (args, arg_names) = map_args(args, context);
                let mut name = func.clone();
                name.ident = format_ident!("__expand_{}", name.ident);
                let call = with_debug_call(
                    context,
                    *span,
                    quote![#ty_path::#name(scope, #(#arg_names),*)],
                );
                quote_spanned! {*span=>
                    {
                        #(#args)*
                        #call
                    }
                }
            }
            Expression::MethodCall { receiver, method, args, span, .. }
                if method == "clone" && args.is_empty() => {
                let receiver = receiver.as_const(context)
                    .unwrap_or_else(|| receiver.to_tokens(context));
                let snapshot = snapshot_value(quote![(#receiver).clone()]);
                with_debug_call(context, *span, snapshot)
            }
            Expression::MethodCall {
                receiver,
                method,
                generics,
                args,
                span,
                ..
            } => {
                let method = format_ident!("__expand_{method}_method");
                let (args, arg_names) = map_args(args, context);
                let receiver = receiver
                    .as_const(context)
                    .unwrap_or_else(|| receiver.to_tokens(context));
                let call = with_debug_call(
                    context,
                    *span,
                    quote![#receiver.#method #generics(scope, #(#arg_names),*)],
                );
                quote_spanned! {*span=>
                    {
                        #(#args)*
                        #call
                    }
                }
            }
            Expression::Break => {
                let path = frontend_path();
                quote![#path::branch::break_expand(scope);]
            }
            Expression::Continue(span) => error!(*span, "Continue not supported yet"),
            Expression::Return(span) => error!(
                *span,
                "Return not supported yet. Consider using the terminate!() macro instead."
            ),
            Expression::Cast { from, to } => {
                let cast = prelude_type("Cast");
                let from = from.to_tokens(context);
                let to = quote_spanned![to.span()=> <#to as #cast>];
                quote! {{
                    let __from = #from;
                    #to::__expand_cast_from(scope, __from.into())
                }}
            }
            Expression::ForLoop {
                range,
                unroll,
                var_name,
                var_ty,
                block,
                scope,
            } => {
                let for_ty = frontend_type("branch");

                let range = range.to_tokens(context);
                let unroll = unroll
                    .as_ref()
                    .and_then(|it| it.as_const(context))
                    .unwrap_or(quote![false]);
                let block = context.in_fn_mut(scope, |ctx| block.to_tokens(ctx));
                let var_ty = var_ty.as_ref().map(|it| quote![: #it]);

                quote! {
                    {
                        let _range = #range;
                        let _unroll = #unroll;
                        #for_ty::for_expand(scope, _range, _unroll, |scope, #var_name #var_ty| #block);
                    }
                }
            }
            Expression::Loop { block, scope } => {
                let loop_ty = frontend_type("branch");
                let block = context.in_fn_mut(scope, |ctx| block.to_tokens(ctx));

                quote![#loop_ty::loop_expand(scope, |scope| #block);]
            }
            Expression::If {
                condition,
                then_block,
                else_branch,
            } if condition.is_const() => {
                let as_const = condition.as_const(context).unwrap();
                let then_block = then_block.to_tokens(context);
                let else_branch = else_branch
                    .as_ref()
                    .map(|it| it.to_tokens(context))
                    .map(|it| quote![else #it]);
                quote![if #as_const #then_block #else_branch]
            }
            Expression::If {
                condition,
                then_block,
                else_branch: Some(else_branch),
            } if then_block.ret.is_some() && else_branch.needs_terminator() => {
                let path = frontend_path();
                let condition = condition.to_tokens(context);
                let then_block = then_block.to_tokens(context);
                let else_branch = else_branch.to_tokens(context);
                quote! {
                    {
                        let _cond = #condition;
                        #path::branch::if_else_expr_expand(scope, _cond.into(), |scope| #then_block).or_else(scope, |scope| #else_branch)
                    }
                }
            }
            Expression::If {
                condition,
                then_block,
                else_branch: Some(else_branch),
            } => {
                let path = frontend_path();
                let condition = condition.to_tokens(context);
                let then_block = then_block.to_tokens(context);
                let else_branch = else_branch.to_tokens(context);
                quote! {
                    {
                        let _cond = #condition;
                        #path::branch::if_else_expand(scope, _cond.into(), |scope| #then_block).or_else(scope, |scope| #else_branch);
                    }
                }
            }
            Expression::If {
                condition,
                then_block,
                ..
            } => {
                let path = frontend_path();
                let condition = condition.to_tokens(context);
                let then_block = then_block.to_tokens(context);
                quote! {
                    {
                        let _cond = #condition;
                        #path::branch::if_expand(scope, _cond.into(), |scope| #then_block);
                    }
                }
            }
            Expression::Switch {
                value,
                cases,
                default,
            } => {
                let branch = frontend_type("branch");
                let switch = match default.ret.is_some() {
                    true => quote![switch_expand_expr],
                    false => quote![switch_expand],
                };
                let value = value.to_tokens(context);
                let default = default.to_tokens(context);
                let blocks = cases
                    .iter()
                    .map(|(val, block)| {
                        let val_tokens = val
                            .as_const(context)
                            .unwrap_or_else(|| val.to_tokens(context));
                        let block = block.to_tokens(context);
                        quote![.case(scope, #val_tokens, |scope| #block)]
                    })
                    .collect::<Vec<_>>();
                quote! {
                    {
                        let _val = #value;
                        #branch::#switch(scope, _val.into(), |scope| #default)
                            #(#blocks)*
                            .finish(scope)
                    }
                }
            }
            Expression::Path { path, qself } => {
                if let Some(qself) = qself {
                    let ty = &qself.ty;
                    quote![<#ty as #path>]
                } else {
                    quote![#path]
                }
            }
            Expression::Range {
                start,
                end,
                inclusive,
                span,
            } => {
                let start = start
                    .as_const(context)
                    .unwrap_or_else(|| start.to_tokens(context));
                if let Some(end) = end {
                    let range = frontend_type("RangeExpand");
                    let end = end
                        .as_const(context)
                        .unwrap_or_else(|| end.to_tokens(context));
                    quote! {
                        {
                            let _start = #start;
                            let _end = #end;
                            #range::new(_start.into(), _end.into(), #inclusive)
                        }
                    }
                } else {
                    error!(*span, "Slice range not yet supported")
                }
            }

            Expression::Array { span, .. } => {
                if let Some(constant) = self.as_const(context) {
                    constant
                } else {
                    error!(*span, "Array expressions can't be used at runtime")
                }
            }
            Expression::Tuple { elements, .. } => {
                if let Some(constant) = self.as_const(context) {
                    constant
                } else {
                    let elements = elements.iter().map(|it| {
                        let value = it.to_tokens(context);
                        if it.is_const() {
                            value
                        } else {
                            let snapshot = snapshot_value(quote![_value.clone()]);
                            quote! {{
                                let _value = #value;
                                #snapshot
                            }}
                        }
                    });
                    quote![(#(#elements,)*)]
                }
            }
            Expression::Slice { span, .. } => {
                error!(*span, "Slice expressions not yet implemented")
            }
            Expression::ArrayInit { init, len } => {
                let init_ty = frontend_type("ArrayInit");
                let init = init.to_tokens(context);
                let len = len.to_tokens(context);

                quote![#init_ty::new(#len, #init)]
            }
            Expression::VerbatimTerminated { tokens } => tokens.clone(),
            Expression::Reference { inner, writable } => {
                if let Expression::Index { expr, index, .. } = &**inner {
                    let expr = expr.to_tokens(context);
                    let index = index.to_tokens(context);
                    quote! {{
                        let _array = #expr;
                        let _index = #index;
                        _array.clone().expand_index_reference(scope, _index.into(), #writable)
                    }}
                } else if let Expression::Unary { input, operator: Operator::Deref, .. } = &**inner {
                    let input = input.to_tokens(context);
                    let into_mut = frontend_type("IntoMut");
                    quote! {{ let _reference = #input; #into_mut::expand_reborrow(_reference.clone(), scope, #writable) }}
                } else {
                    let (inner, method) = if let Some(value) = inner.as_const(context) {
                        (value, format_ident!("constant_reference"))
                    } else {
                        (inner.to_tokens(context), format_ident!("reference"))
                    };
                    let snapshot = frontend_type("ValueSnapshot");
                    let fallback = frontend_type("ValueSnapshotFallback");
                    quote! {{
                        use #fallback as _;
                        let _value = (#inner).clone();
                        #snapshot(_value).#method(scope, #writable)
                    }}
                }
            }
            Expression::StructInit { path, fields } => {
                let ruda_type = prelude_type("RudaType");
                let fields = init_fields(fields, context);
                quote! {
                    {
                        type _Ty<T> = <T as #ruda_type>::ExpandType;
                        _Ty::<#path> { #(#fields),* }
                    }
                }
            }
            Expression::Closure {
                params,
                body,
                scope,
            } => {
                // Without knowing the closure type, we need to assume it's `FnMut`
                let body = context.in_fn_mut(scope, |ctx| body.to_tokens(ctx));
                quote![|scope, #(#params),*| #body]
            }
            Expression::Verbatim { tokens, .. } => tokens.clone(),
            Expression::Block(block) => block.to_tokens(context),
            Expression::RuntimeMatch {
                expr,
                arms,
                default,
            } => {
                let branch = frontend_type("branch");

                let has_value = arms
                    .iter()
                    .next()
                    .is_some_and(|arm| arm.expr.needs_terminator());

                let match_ = match has_value {
                    true => quote![match_expand_expr],
                    false => quote![match_expand],
                };
                let comptime_finish = has_value.then(|| quote![use #branch::ComptimeMatchExpr as _;]);

                let expr = expr
                    .as_const(context)
                    .unwrap_or_else(|| expr.to_tokens(context));

                let arms = arms
                    .iter()
                    .map(|arm| arm.to_tokens(context, true, false))
                    .collect::<Vec<_>>();

                let default = default
                    .as_ref()
                    .map(|arm| arm.to_tokens(context, true, false))
                    .map(|(pat, block)| {
                        let bindings = runtime_pattern_bindings(&pat, quote![__ruda_match_value]);
                        quote![.default(scope, |scope, _value| { #bindings #block })]
                    });

                let cases = arms
                    .iter()
                    .enumerate()
                    .map(|(i, (pat, block))| -> Option<_> {
                        let _ = variant_name(pat)?;
                        let discriminant = format_ident!("_disc_{i}");
                        let bindings = runtime_pattern_bindings(pat, quote![__ruda_match_value]);
                        let block = quote! {{ #bindings #block }};
                        Some((discriminant, block))
                    })
                    .collect::<Option<Vec<_>>>();

                if let Some(mut cases) = cases {
                    let discriminants = arms.iter().enumerate().map(|(i, (pat, _))| {
                        let name = variant_name(pat).expect("Already checked");
                        let ident = format_ident!("_disc_{i}");
                        quote![let #ident = __ruda_match_value.discriminant_of_value(#name);]
                    });

                    // Needed so type inference can actually work
                    let (disc0, block0) = cases.remove(0);
                    let cases = cases
                        .iter()
                        .map(|(disc, block)| quote![.case(scope, #disc, |scope, _value| #block)]);
                    quote! {
                        {
                            let __ruda_match_value = (#expr).clone();
                            #(#discriminants)*
                            #comptime_finish
                            #branch::#match_(scope, __ruda_match_value.clone(), #disc0, |scope, _value| #block0)
                                #(#cases)*
                                #default
                                .finish(scope)
                        }
                    }
                } else {
                    let arms = arms.iter().map(|(pat, block)| quote![#pat => #block]);

                    quote! { match #expr { #(#arms,)* } }
                }
            }
            Expression::Match {
                runtime_variants,
                expr,
                arms,
            } => {
                let is_const = self.is_const();

                let expr = match expr.as_const(context) {
                    Some(expr) => expr,
                    None => {
                        let expr = expr.to_tokens(context);
                        let snapshot = snapshot_value(quote![__ruda_match_value]);
                        quote! {{
                            let __ruda_match_value = #expr;
                            #snapshot
                        }}
                    }
                };

                let arms = arms
                    .iter()
                    .map(|arm| arm.to_tokens(context, *runtime_variants, is_const))
                    .map(|(pat, block)| quote![#pat => #block])
                    .collect::<Vec<_>>();

                quote! { match #expr { #(#arms,)* } }
            }
            Expression::RuntimeIfLet {
                expr,
                arm,
                else_branch,
            } => {
                let name = variant_name(&arm.pat);
                let if_expand = prelude_type("if_expand");
                let if_else_expand = prelude_type("if_else_expand");
                let if_else_expr_expand = prelude_type("if_else_expr_expand");

                if let Some(name) = name {
                    let expr = expr
                        .as_const(context)
                        .unwrap_or_else(|| expr.to_tokens(context));

                    let (pat, block) = arm.to_tokens(context, true, false);

                    let bindings = runtime_pattern_bindings(&pat, quote![__ruda_match_value]);
                    let block = quote! {{ #bindings #block }};

                    let expand = match else_branch {
                        Some(else_branch) if else_branch.needs_terminator() => {
                            let else_branch = else_branch.to_tokens(context);
                            quote! {
                                #if_else_expr_expand(scope, __cond.into(), |scope| #block).or_else(scope, |scope| #else_branch)
                            }
                        }
                        Some(else_branch) => {
                            let else_branch = else_branch.to_tokens(context);
                            quote! {
                                #if_else_expand(scope, __cond.into(), |scope| #block).or_else(scope, |scope| #else_branch);
                            }
                        }
                        None => quote![#if_expand(scope, __cond.into(), |scope| #block);],
                    };

                    quote! {{
                        let __ruda_match_value = (#expr).clone();
                        let __disc = __ruda_match_value.discriminant_of_value(#name);
                        let __cond = eq::expand(scope, __ruda_match_value.discriminant(), __disc.into());

                        #expand
                    }}
                } else {
                    let is_const = self.is_const();
                    let expr = expr
                        .as_const(context)
                        .unwrap_or_else(|| expr.to_tokens(context));
                    let (pat, body) = arm.to_tokens(context, true, is_const);
                    let else_branch = else_branch
                        .as_ref()
                        .map(|it| it.to_tokens(context))
                        .map(|it| quote![else #it]);
                    quote! { if let #pat = #expr #body #else_branch }
                }
            }
            Expression::IfLet {
                runtime_variants,
                expr,
                arm,
                else_branch,
            } => {
                let is_const = self.is_const();
                let expr = match expr.as_const(context) {
                    Some(expr) => expr,
                    None => {
                        let expr = expr.to_tokens(context);
                        let snapshot = snapshot_value(quote![__ruda_match_value]);
                        quote! {{
                            let __ruda_match_value = #expr;
                            #snapshot
                        }}
                    }
                };
                let (pat, body) = arm.to_tokens(context, *runtime_variants, is_const);
                let else_branch = else_branch
                    .as_ref()
                    .map(|it| it.to_tokens(context))
                    .map(|it| quote![else #it]);
                quote! { if let #pat = #expr #body #else_branch }
            }
            Expression::Comment { content } => {
                let frontend_path = frontend_path();
                quote![#frontend_path::ruda_comment::expand(scope, #content)]
            }
            Expression::RustMacro { ident, tokens } => {
                quote![#ident!(#tokens)]
            }
            Expression::Terminate => {
                let frontend_path = frontend_path();
                quote![#frontend_path::branch::return_expand(scope);]
            }
            Expression::AssertConstant { inner } => inner.to_tokens(context),
            Expression::ExpressionMacro { ident, args } => {
                let frontend_path = frontend_path();
                let expand = format_ident!("{}_expand", ident);
                let args = args
                    .iter()
                    .map(|expr| expr.to_tokens(context))
                    .enumerate()
                    .map(|(i, arg)| {
                        let name = format_ident!("_{i}");
                        quote![let #name = #arg;]
                    });
                let arg_uses = (0..args.len()).map(|i| format_ident!("_{i}"));
                quote! {
                    {
                        #(
                            #args
                        )*
                        #frontend_path::#expand!(scope, #(#arg_uses),*);
                    }
                }
            }
        }
    }
}

impl MatchArm {
    pub fn to_tokens(
        &self,
        context: &mut Context,
        runtime_variants: bool,
        is_const: bool,
    ) -> (Pat, TokenStream) {
        let mut pat = self.pat.clone();

        // If using runtime variants, we need to replace the variant Name with
        // NameExpand.
        if runtime_variants {
            Self::expand_pat(&mut pat);
        }

        let expr = if is_const {
            self.expr.as_const(context).unwrap()
        } else {
            self.expr.to_tokens(context)
        };

        (pat, expr)
    }

    fn expand_pat(pat: &mut Pat) {
        match pat {
            Pat::Ident(ident) if ident.ident == "None" => {
                *pat = Pat::Path(parse_quote![OptionExpand::None]);
            }
            Pat::Ident(PatIdent {
                subpat: Some((_, pat)),
                ..
            }) => {
                Self::expand_pat(pat);
            }
            // Match simple ident like x in Enum::Variant(x).
            // Useful for recursive call.
            Pat::Ident(_) => {}
            // Match path::Enum::Ident
            Pat::Path(pat) => {
                let mut path = pat.path.clone();
                append_expand_to_enum_name(&mut path);
                pat.path = path;
            }
            // Match path::Enum::Variant {a, b, c}
            Pat::Struct(pat) => {
                let mut path = pat.path.clone();
                append_expand_to_enum_name(&mut path);
                pat.path = path;
            }
            // Match path::Enum::Variant(a, b, c)
            Pat::TupleStruct(pat) => {
                let mut path = pat.path.clone();
                append_expand_to_enum_name(&mut path);
                pat.path = path;
            }
            // Match Pat1 | Pat2 | ...
            Pat::Or(pat) => {
                pat.cases.iter_mut().for_each(Self::expand_pat);
            }
            // Match (Pat1, Pat2, ...)
            Pat::Tuple(pat) => {
                pat.elems.iter_mut().for_each(Self::expand_pat);
            }
            // Match the underscore pattern _
            Pat::Wild(_) => {}
            _ => {
                panic!("unsupported pattern in match for {pat:?}");
                // NOTE: From the documentation https://docs.rs/syn/latest/syn/enum.Pat.html
                //       I don't think we should support any other patterns.
                //       Users can always use a big if, else if, else pattern
                // instead.       Currently, the goal is to
                // support RudaType enums.
            }
        }
    }
}

// Replace something like `some_path::Enum::Variant` with
// `some_path::EnumExpand::Variant`.
fn append_expand_to_enum_name(path: &mut Path) {
    if path.segments.len() >= 2 {
        let segment = path.segments.get_mut(path.segments.len() - 2).unwrap(); // Safe because of the if
        segment.ident = Ident::new(&format!("{}Expand", segment.ident), Span::call_site());
    } else if path.is_ident("Some") || path.is_ident("None") {
        // Insert RudaOption for prelude use of `Some` and `None`
        *path = parse_quote!(OptionExpand::#path);
    } else {
        panic!("Found single path {path:?}");
    }
}

impl Block {
    pub fn to_tokens(&self, context: &mut Context) -> TokenStream {
        let inner: Vec<_> = self.inner.iter().map(|it| it.to_tokens(context)).collect();
        let ret = if let Some(ret) = self.ret.as_ref() {
            let as_const = ret.as_const(context);
            if let Some(as_const) = as_const {
                quote![#as_const]
            } else {
                ret.to_tokens(context)
            }
        } else {
            quote![()]
        };

        quote! {
            {
                #(#inner)*
                #ret
            }
        }
    }
}

fn split_generics(path: &Expression, context: &mut Context) -> (PathArguments, TokenStream) {
    let mut path = match path {
        Expression::Path { path, .. } => path.clone(),
        _ => return (PathArguments::None, path.to_tokens(context)),
    };
    let generics = if let Some(last) = path.segments.last_mut() {
        core::mem::replace(&mut last.arguments, PathArguments::None)
    } else {
        PathArguments::None
    };
    (generics, quote![#path])
}

pub(crate) fn snapshot_value(value: TokenStream) -> TokenStream {
    let snapshot = frontend_type("ValueSnapshot");
    let fallback = frontend_type("ValueSnapshotFallback");
    quote! {{
        use #fallback as _;
        #snapshot(#value).snapshot(scope)
    }}
}

fn map_args(args: &[Expression], context: &mut Context) -> (Vec<TokenStream>, Vec<TokenStream>) {
    let names: Vec<_> = (0..args.len()).map(|i| format_ident!("_arg_{i}")).collect();
    let values = names
        .iter()
        .zip(args.iter())
        .map(|(i, value)| {
            if matches!(value, Expression::Closure { .. }) {
                quote![]
            } else {
                if let Some(tokens) = value.as_const(context) {
                    quote_spanned![tokens.span()=> let #i = #tokens;]
                } else {
                    let tokens = value.to_tokens(context);
                    let snapshot = snapshot_value(quote![_value]);
                    quote![let #i = {
                        let _value = #tokens;
                        #snapshot
                    };]
                }
            }
        })
        .collect();
    let names = names
        .into_iter()
        .zip(args.iter())
        .map(|(name, value)| {
            if matches!(value, Expression::Closure { .. }) {
                value.to_tokens(context)
            } else {
                quote![#name.into()]
            }
        })
        .collect();
    (values, names)
}

fn init_fields<'a>(
    fields: &'a [(Member, Expression)],
    context: &'a mut Context,
) -> impl Iterator<Item = TokenStream> + 'a {
    fields.iter().map(|(pat, it)| {
        let it = if let Some(as_const) = it.as_const(context) {
            let it = quote_spanned![as_const.span()=> #as_const];
            return quote! {
                #pat: {
                    let _init = #it.clone();
                    _init.into()
                }
            };
        } else {
            it.to_tokens(context)
        };
        let snapshot = snapshot_value(quote![_init.clone()]);
        quote! {
            #pat: {
                let _init = #it;
                #snapshot
            }
        }
    })
}

fn with_span(context: &Context, span: Span, tokens: TokenStream) -> TokenStream {
    if context.debug_symbols {
        let debug_spanned = frontend_type("spanned_expand");
        quote_spanned! {span=>
            #debug_spanned(scope, line!(), column!(), |scope| #tokens)
        }
    } else {
        tokens
    }
}

fn with_debug_call(context: &Context, span: Span, tokens: TokenStream) -> TokenStream {
    if context.debug_symbols {
        let debug_call = frontend_type("debug_call_expand");
        quote_spanned! {span=>
            #debug_call(scope, line!(), column!(), |scope| #tokens)
        }
    } else {
        tokens
    }
}

fn variant_name(pat: &Pat) -> Option<String> {
    match pat {
        Pat::Ident(PatIdent { subpat: Some((_, pat)), .. }) => variant_name(pat),
        Pat::Ident(pat) => Some(pat.ident.to_string()),
        Pat::Paren(pat) => variant_name(&pat.pat),
        Pat::Path(PatPath { path, .. })
        | Pat::Struct(PatStruct { path, .. })
        | Pat::TupleStruct(PatTupleStruct { path, .. }) => {
            path.segments.last().map(|it| it.ident.to_string())
        }
        _ => None,
    }
}

fn runtime_pattern_bindings(pat: &Pat, value: TokenStream) -> TokenStream {
    use crate::ir::generate::{runtime_field_accessor, runtime_reverse_accessor};
    match pat {
        Pat::Paren(pat) => runtime_pattern_bindings(&pat.pat, value),
        Pat::TupleStruct(pat) => {
            let variant = &pat.path.segments.last().unwrap().ident;
            let rest = pat.elems.iter().position(|pat| matches!(pat, Pat::Rest(_)));
            let fields = pat.elems.iter().enumerate().filter_map(|(i, pat_inner)| {
                if matches!(pat_inner, Pat::Rest(_)) { return None; }
                let accessor = if rest.is_some_and(|rest| i > rest) {
                    runtime_reverse_accessor(variant, pat.elems.len() - i - 1)
                } else {
                    runtime_field_accessor(variant, &Member::Unnamed(syn::Index::from(i)))
                };
                Some(bind_runtime_value(pat_inner, quote![#value.#accessor()]))
            });
            quote![#(#fields)*]
        }
        Pat::Struct(pat) => {
            let variant = &pat.path.segments.last().unwrap().ident;
            let fields = pat.fields.iter().map(|field| {
                let accessor = runtime_field_accessor(variant, &field.member);
                bind_runtime_value(&field.pat, quote![#value.#accessor()])
            });
            quote![#(#fields)*]
        }
        Pat::Ident(PatIdent { subpat: Some((_, inner)), .. }) => {
            let mut binding = pat.clone();
            if let Pat::Ident(binding) = &mut binding { binding.subpat = None; }
            let outer = bind_runtime_value(&binding, quote![#value.clone()]);
            let inner = runtime_pattern_bindings(inner, value);
            quote![#outer #inner]
        }
        Pat::Ident(ident) if ident.ident != "None" => bind_runtime_value(pat, quote![#value.clone()]),
        _ => quote![],
    }
}

pub(crate) fn bind_runtime_value(pat: &Pat, value: TokenStream) -> TokenStream {
    fn bind(pat: &Pat, value: TokenStream, next: &mut usize) -> TokenStream {
        let into_mut = frontend_type("IntoMut");
        match pat {
            Pat::Paren(pat) => bind(&pat.pat, value, next),
            Pat::Reference(pat) => bind(&pat.pat, quote![#into_mut::expand_deref(#value, scope)], next),
            Pat::Ident(PatIdent { subpat: Some((_, inner)), .. }) => {
                let temp = format_ident!("__ruda_pattern_{}", *next);
                *next += 1;
                let mut outer = pat.clone();
                if let Pat::Ident(outer) = &mut outer { outer.subpat = None; }
                let outer = bind(&outer, quote![#temp.clone()], next);
                let inner = bind(inner, quote![#temp], next);
                quote![let #temp = #value; #outer #inner]
            }
            Pat::Ident(pat) => {
                let name = &pat.ident;
                let writable = pat.mutability.is_some();
                if pat.by_ref.is_some() {
                    quote![let #name = #into_mut::expand_reference(#value, scope, #writable);]
                } else if writable {
                    quote![let mut #name = #into_mut::into_mut(#value, scope);]
                } else {
                    let snapshot = snapshot_value(value);
                    quote![let #name = #snapshot;]
                }
            }
            Pat::Tuple(pat) => {
                let mut names = Vec::new();
                let mut fields = Vec::new();
                for field in &pat.elems {
                    if matches!(field, Pat::Rest(_)) {
                        names.push(quote![..]);
                    } else {
                        let temp = format_ident!("__ruda_pattern_{}", *next);
                        *next += 1;
                        names.push(quote![#temp]);
                        fields.push((field, temp));
                    }
                }
                let fields = fields.into_iter().map(|(field, temp)| bind(field, quote![#temp], next));
                quote![let (#(#names,)*) = #value; #(#fields)*]
            }
            Pat::Struct(pat) => {
                let temp = format_ident!("__ruda_pattern_{}", *next);
                *next += 1;
                let fields = pat.fields.iter().map(|field| {
                    let member = &field.member;
                    bind(&field.pat, quote![#temp.#member.clone()], next)
                });
                quote![let #temp = #value; #(#fields)*]
            }
            Pat::TupleStruct(pat) => {
                let temp = format_ident!("__ruda_pattern_{}", *next);
                *next += 1;
                let rest = pat.elems.iter().position(|pat| matches!(pat, Pat::Rest(_)));
                let fields = pat.elems.iter().enumerate().filter_map(|(i, field)| {
                    if matches!(field, Pat::Rest(_)) { return None; }
                    let field_value = if rest.is_some_and(|rest| i > rest) {
                        let accessor = format_ident!(
                            "__expand_tuple_field_reverse_{}", pat.elems.len() - i - 1,
                        );
                        quote![#temp.#accessor()]
                    } else {
                        let member = syn::Index::from(i);
                        quote![#temp.#member.clone()]
                    };
                    Some(bind(field, field_value, next))
                });
                quote![let #temp = #value; #(#fields)*]
            }
            Pat::Type(pat) => bind(&pat.pat, value, next),
            _ => quote![let #pat = #value;],
        }
    }
    bind(pat, value, &mut 0)
}

