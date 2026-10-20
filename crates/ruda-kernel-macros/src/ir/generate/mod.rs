use darling::usage::{GenericsExt, Purpose, UsesTypeParams};
use proc_macro2::TokenStream;
use syn::{
    Generics, Token, Type, WhereClause, parse_quote, punctuated::Punctuated, spanned::Spanned,
};

pub mod assign;
pub mod autotune;
pub mod ruda_impl;
pub mod ruda_trait;
pub mod ruda_type;
pub mod expression;
pub mod into_runtime;
pub mod kernel;
pub mod launch;
pub mod statement;

pub(crate) fn runtime_enum_uses_slots<'a>(
    variants: impl IntoIterator<Item = (bool, Vec<&'a Type>)>,
) -> bool {
    let mut first = None;
    for (structured, fields) in variants {
        if structured || fields.len() > 1 { return true; }
        if let Some(ty) = fields.first() {
            let ty = quote::quote![#ty].to_string();
            if first.as_ref().is_some_and(|first| first != &ty) { return true; }
            first = Some(ty);
        }
    }
    false
}

pub(crate) fn runtime_slot_name(variant: usize, field: usize) -> syn::Ident {
    quote::format_ident!("__ruda_variant_{variant}_{field}")
}

pub(crate) fn runtime_field_accessor(variant: &syn::Ident, field: &syn::Member) -> syn::Ident {
    let variant = variant.to_string();
    let field = match field {
        syn::Member::Named(name) => format!("named_{}", name.to_string().trim_start_matches("r#")),
        syn::Member::Unnamed(index) => format!("index_{}", index.index),
    };
    let variant = variant.trim_start_matches("r#");
    quote::format_ident!("__expand_field_{}_{}_{}", variant.len(), variant, field)
}

pub(crate) fn runtime_reverse_accessor(variant: &syn::Ident, index: usize) -> syn::Ident {
    let variant = variant.to_string();
    let variant = variant.trim_start_matches("r#");
    quote::format_ident!("__expand_field_{}_{}_reverse_{}", variant.len(), variant, index)
}

pub trait RuntimeField: Clone + UsesTypeParams {
    fn ty(self) -> Type;
}

impl RuntimeField for syn::Field {
    fn ty(self) -> Type {
        self.ty
    }
}

pub(crate) fn bounded_where_clause<T: RuntimeField>(
    generics: &Generics,
    runtime_fields: impl IntoIterator<Item = T>,
    bound: impl Fn(&Type) -> TokenStream,
) -> Option<WhereClause> {
    let params = generics.declared_type_params();
    let opts = Purpose::BoundImpl.into();

    let fields_params = runtime_fields
        .into_iter()
        .filter(|it| !it.clone().uses_type_params(&opts, &params).is_empty())
        .map(|it| it.ty())
        .collect::<Vec<_>>();

    if fields_params.is_empty() {
        return generics.where_clause.clone();
    }

    let span = generics.span();
    let mut where_clause = generics
        .where_clause
        .clone()
        .unwrap_or_else(|| WhereClause {
            where_token: Token![where](span),
            predicates: Punctuated::new(),
        });

    for field_ty in fields_params {
        let bound = bound(&field_ty);
        where_clause.predicates.push(parse_quote!(#bound));
    }

    Some(where_clause)
}
