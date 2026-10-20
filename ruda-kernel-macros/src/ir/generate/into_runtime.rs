use darling::{
    FromDeriveInput,
    ast::{Data, Style},
};
use proc_macro2::TokenStream;
use quote::{ToTokens, format_ident, quote};
use syn::{DeriveInput, Index, WhereClause};

use crate::ir::{
    generate::{bounded_where_clause, runtime_enum_uses_slots},
    parse::into_runtime::{IntoRuntime, IntoRuntimeVariant},
    paths::{core_type, prelude_type},
};

impl ToTokens for IntoRuntime {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        let into_runtime = core_type("IntoRuntime");
        let ruda_type = prelude_type("RudaType");
        let scope = prelude_type("Scope");

        let name = &self.ident;
        let (generics, generic_names, _) = self.generics.split_for_impl();
        let where_clause = self.where_clause();

        let init = match &self.data {
            Data::Enum(_) if self.runtime_variants.is_present() => self.init_runtime_enum(),
            Data::Enum(_) => self.init_comptime_enum(),
            Data::Struct(_) => self.init_struct(),
        };

        tokens.extend(quote! {
            impl #generics #into_runtime for #name #generic_names #where_clause {
                fn __expand_runtime_method(self, scope: &mut #scope) -> Self::ExpandType {
                    type _Ty #generic_names = <#name #generic_names as #ruda_type>::ExpandType;
                    #init
                }
            }
        });
    }
}

impl IntoRuntime {
    fn uses_runtime_slots(&self) -> bool {
        match &self.data {
            Data::Enum(variants) => runtime_enum_uses_slots(variants.iter().map(|variant| {
                (variant.fields.style == Style::Struct
                    || (variant.fields.style == Style::Tuple && variant.fields.is_empty()),
                    variant.fields.iter().map(|field| &field.ty).collect())
            })),
            _ => false,
        }
    }

    fn init_struct(&self) -> TokenStream {
        let into_runtime = core_type("IntoRuntime");
        let expand_name = self.name_expand.clone()
            .unwrap_or_else(|| format_ident!("{}Expand", self.ident));

        let struct_ = self.data.as_ref().take_struct().unwrap();

        let fields = struct_.fields.iter().enumerate().map(|(i, field)| {
            let index = Index::from(i);
            match &field.ident {
                Some(name) if field.comptime.is_present() => quote![#name: self.#name],
                Some(name) => {
                    quote![#name: #into_runtime::__expand_runtime_method(self.#name, scope)]
                }
                None if field.comptime.is_present() => quote![self.#index],
                None => quote![#into_runtime::__expand_runtime_method(self.#index, scope)],
            }
        });
        match struct_.style {
            Style::Tuple => quote![#expand_name(#(#fields,)*)],
            Style::Struct => quote![_Ty { #(#fields,)* }],
            Style::Unit => quote![#expand_name],
        }
    }

    fn init_comptime_enum(&self) -> TokenStream {
        let variants = self.data.as_ref().take_enum().unwrap();

        let inits = variants
            .iter()
            .map(|variant| self.init_comptime_variant(variant));

        quote![match self { #(#inits,)* }]
    }

    fn init_comptime_variant(&self, variant: &IntoRuntimeVariant) -> TokenStream {
        let into_runtime = core_type("IntoRuntime");

        let enum_name = &self.ident;
        let variant_name = &variant.ident;

        let field_names = variant
            .fields
            .iter()
            .enumerate()
            .map(|(i, field)| field.ident.clone().unwrap_or_else(|| format_ident!("_{i}")));

        let fields = variant.fields.iter().enumerate().map(|(i, field)| {
            let index = format_ident!("_{i}");
            match &field.ident {
                Some(name) if field.comptime.is_present() => quote![#name],
                Some(name) => {
                    quote![#name: #into_runtime::__expand_runtime_method(#name, scope)]
                }
                None if field.comptime.is_present() => quote![#index],
                None => quote![#into_runtime::__expand_runtime_method(#index, scope)],
            }
        });

        match variant.fields.style {
            Style::Tuple => {
                quote![#enum_name::#variant_name(#(#field_names,)*) => _Ty::#variant_name(#(#fields,)*)]
            }
            Style::Struct => {
                quote![#enum_name::#variant_name { #(#field_names,)* } => _Ty::#variant_name{ #(#fields,)* }]
            }
            Style::Unit => quote![#enum_name::#variant_name => _Ty::#variant_name],
        }
    }

    fn init_runtime_enum(&self) -> TokenStream {
        if self.uses_runtime_slots() { return self.init_runtime_slots(); }
        let into_runtime = core_type("IntoRuntime");

        let name = &self.ident;
        let expand_name = self.name_expand.clone()
            .unwrap_or_else(|| format_ident!("{name}Expand"));
        let (_, generic_names, _) = self.generics.split_for_impl();
        let generic_names = generic_names.as_turbofish();
        let variants = self.data.as_ref().take_enum().unwrap();

        let value_ty = variants
            .iter()
            .find_map(|v| match v.fields.style {
                Style::Struct => unimplemented!(),
                Style::Tuple => Some(v.fields.iter().next().unwrap().ty.clone()),
                Style::Unit => None,
            })
            .map(|ty| quote![#ty])
            .unwrap_or_else(|| quote![()]);

        let discriminants = variants.iter().map(|v| {
            let variant = &v.ident;
            let variant_name = variant.to_string();
            let discriminant = quote![#expand_name #generic_names::discriminant_of(#variant_name)];
            match v.fields.style {
                Style::Tuple => quote![#name::#variant(..) => #discriminant],
                Style::Struct => quote![#name::#variant { .. } => #discriminant],
                Style::Unit => quote![#name::#variant => #discriminant],
            }
        });

        let values = variants
            .iter()
            .map(|variant| self.runtime_variant_value(variant));

        let discriminant = quote! {
            let discriminant = match &self {
                #(#discriminants,)*
            };
        };
        let value = quote! {
            let value: #value_ty = match self {
                #(#values,)*
            };
        };

        quote! {
            #discriminant
            #value
            _Ty {
                discriminant: discriminant.into(),
                value: #into_runtime::__expand_runtime_method(value, scope),
            }
        }
    }

    fn init_runtime_slots(&self) -> TokenStream {
        let into_runtime = core_type("IntoRuntime");
        let name = &self.ident;
        let expand = self.name_expand.clone().unwrap_or_else(|| format_ident!("{}Expand", name));
        let (_, names, _) = self.generics.split_for_impl();
        let turbofish = names.as_turbofish();
        let variants = self.data.as_ref().take_enum().unwrap();
        let arms = variants.iter().map(|variant| {
            let variant_name = &variant.ident;
            let construct = format_ident!("__expand_runtime_{}", variant_name);
            let args: Vec<_> = variant.fields.iter().enumerate()
                .map(|(i, field)| field.ident.clone().unwrap_or_else(|| format_ident!("arg_{i}"))).collect();
            let pattern = match variant.fields.style {
                Style::Struct => quote![#name::#variant_name { #(#args,)* }],
                Style::Tuple => quote![#name::#variant_name(#(#args,)*)],
                Style::Unit => quote![#name::#variant_name],
            };
            quote! {
                #pattern => {
                    #(let #args = #into_runtime::__expand_runtime_method(#args, scope);)*
                    #expand #turbofish::#construct(scope, #(#args),*)
                }
            }
        });
        quote![match self { #(#arms,)* }]
    }

    fn runtime_variant_value(&self, variant: &IntoRuntimeVariant) -> TokenStream {
        let enum_name = &self.ident;
        let variant_name = &variant.ident;

        match variant.fields.style {
            Style::Tuple => {
                quote![#enum_name::#variant_name(value) => value]
            }
            Style::Struct => unimplemented!(),
            Style::Unit => quote![#enum_name::#variant_name => Default::default()],
        }
    }

    fn where_clause(&self) -> Option<WhereClause> {
        let into_runtime = prelude_type("IntoRuntime");
        let fields: Vec<_> = match &self.data {
            Data::Enum(variants) => variants
                .iter()
                .flat_map(|it| it.fields.fields.iter())
                .filter(|it| !it.comptime.is_present())
                .cloned()
                .collect(),
            Data::Struct(fields) => fields
                .iter()
                .filter(|it| !it.comptime.is_present())
                .cloned()
                .collect(),
        };
        if self.runtime_variants.is_present() && self.data.is_enum() && !self.uses_runtime_slots() {
            bounded_where_clause(
                &self.generics,
                fields,
                |param| quote![#param: #into_runtime + Default],
            )
        } else {
            bounded_where_clause(
                &self.generics,
                fields,
                |param| quote![#param: #into_runtime],
            )
        }
    }
}

pub fn generate_into_runtime(input: &DeriveInput) -> syn::Result<TokenStream> {
    let into_runtime = IntoRuntime::from_derive_input(input)?;
    Ok(into_runtime.into_token_stream())
}
